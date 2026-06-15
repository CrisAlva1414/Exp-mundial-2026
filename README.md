# Football World Cup Predictor 2026

**Predictor probabilístico de resultados de fútbol usando ML + Distribuciones Estadísticas**

---

## 📋 Resumen Ejecutivo

Modelo que predice marcadores de partidos (Mundial 2026) usando:
1. **Datos históricos** de football-data.org (matchs finalizados)
2. **XGBoost** para estimar probabilidades de victoria/empate/derrota
3. **Distribución de Poisson** para generar distribuciones de goles
4. **Monte Carlo** para simulaciones de marcadores con incertidumbre
5. **SHAP** para explicabilidad

**Salida**: Para cada partido, `{prob_local, prob_empate, prob_visitante, marcador_más_probable, rango_confianza}`

---

## 🗂️ Estructura del Directorio

```
football_predictor/
├── README.md                          # Este archivo
├── config.py                          # Configuración global (API keys, paths, competiciones)
├── requirements.txt                   # Dependencias Python
│
├── data/
│   ├── matches.csv                    # Histórico de partidos (football-data.org)
│   ├── model_xgboost.pkl              # Modelo entrenado (binario)
│   ├── model_scaler.pkl               # Scaler de features (StandardScaler)
│   └── features_metadata.json         # Metadata: nombres features, versión
│
├── fetchers/
│   ├── __init__.py
│   ├── base_fetcher.py                # Clase abstracta BaseFetcher
│   └── football_data_fetcher.py       # Implementación: descarga de football-data.org
│
├── preprocessing/
│   ├── __init__.py
│   ├── features.py                    # Feature engineering
│   └── validation.py                  # Validación de datos (NaN, outliers, etc)
│
├── models/
│   ├── __init__.py
│   ├── xgboost_classifier.py          # XGBoost wrapper + training
│   ├── poisson_simulator.py           # Simulador de Poisson + Monte Carlo
│   └── predictor.py                   # Orquestador: XGB → Poisson → resultados
│
├── orchestrator.py                    # Loop principal, ingestión + entrenamiento
├── predict.py                         # CLI: "dame la predicción para Argentina vs Francia"
│
└── logs/
    └── orchestrator.log               # Logs de ejecuciones
```

---

## 📊 Arquitecura de Datos y Fuentes

### Única Fuente: **football-data.org**

| Endpoint | Dato | Frecuencia | Rate Limit |
|---|---|---|---|
| `/competitions/WC/matches` | Partidos de Mundiales (2014, 2018, 2022, 2026) | Ingestión semanal | 10 req/min (free tier) |
| `/competitions/{CL,PL,PD,SA}/matches` | Ligas top (Champions, Premier, La Liga, Serie A) | Diariamente | Incluido en 10 req/min |

**Por qué solo football-data.org**: Simplifica credenciales, rate limits únicos, y tiene suficiente data histórica.

### CSV Resultante: `matches.csv`

```csv
match_id,competition,season,date,home_team,away_team,home_goals,away_goals,venue,stage
1234,WC,2022,2022-11-21,Argentina,Saudi Arabia,1,2,Lusail,Group Stage
1235,WC,2022,2022-11-21,Mexico,Poland,0,0,Stadium 974,Group Stage
```

---

## 🔧 Descripción de Archivos

### Core Ingestión

**`config.py`**
- API key de football-data.org, rate limits, paths
- Lista de competiciones + seasons a descargar
- Intervalos de ejecución (1 hora entre ciclos, ajustable)

**`fetchers/base_fetcher.py`**
- Clase abstracta `BaseFetcher` con métodos comunes
- `rate_limit()`: respeta 10 req/min (6 segundos entre calls)
- `run()`: orquesta fetch + save con manejo de errores
- Return: `{"success": bool, "new_records": int, "timestamp": ...}`

**`fetchers/football_data_fetcher.py`**
- Hereda de `BaseFetcher`
- `fetch()`: itera competiciones/seasons, llama API, parsea JSON
- `_parse_match()`: extrae campos relevantes, maneja NaN
- `save()`: deduplicación por `match_id`, append a `matches.csv`

**`orchestrator.py`**
- Loop principal con stop mode (Ctrl+C, SIGTERM)
- Modos: `once` (testing), `interval` (cada N min), `cron` (schedule)
- Pasos por ciclo:
  1. Ejecuta `FootballDataFetcher.run()`
  2. Si hay datos nuevos → entrenamiento automático
  3. Guarda estado en `state.json`

---

### Preprocessing y Features

**`preprocessing/features.py`**

Genera features antes de XGBoost:

```python
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: matches.csv sin features
    Output: Dataset con 20+ features para XGBoost
    
    Features categóricas (one-hot encoded):
    - home_team, away_team (selecciones nacionales)
    - competition (WC, CL, PL, etc)
    
    Features numéricas:
    - home_form_5: goles promedio en últimos 5 partidos (local)
    - away_form_5: goles promedio en últimos 5 partidos (visitante)
    - home_form_recent: forma últimos 3 partidos (W=3, D=1, L=0)
    - away_form_recent: análogo
    - home_goals_avg_season: promedio de goles en la temporada
    - away_goals_conceded_avg: promedio de goles en contra
    - days_since_last_match: descanso (lesiones, fatiga)
    - home_advantage: +1 si no es cancha neutral, 0 si lo es
    - elo_diff: Elo(home) - Elo(away) [sacado de eloratings.net manualmente]
    
    One-hot encoded binary:
    - is_knockout: True si es fase eliminatoria (Copa, CL)
    - is_neutral_venue: True si es cancha neutral
    """
```

**`preprocessing/validation.py`**
- Detección de NaN: rellena con media/forward-fill según contexto
- Outliers: caps a ±3σ en features numéricas
- Tipo de dato: asegura int/float/bool correcto

---

### Modelos ML

**`models/xgboost_classifier.py`**

```python
class XGBoostPredictor:
    """
    POR QUÉ XGBoost y no otro?
    
    1. CALIBRACIÓN: Necesitamos P(local), P(empate), P(visitante)
       XGBoost + CalibratedClassifierCV produce probabilidades confiables
       → Usamos monotonic_probabilistic_output + Isotonic Regression
    
    2. NO LINEAR: Relación entre features y resultado es NO-linear
       - Ventaja del local no es aditiva (depende de la liga)
       - Form reciente pesa más que form histórica (interacción)
       - Tree-based captura estas interacciones sin feature engineering manual
    
    3. ROBUSTO: Maneja missing values implícitamente
    
    4. EXPLICABLE: SHAP values muestran qué feature causó cada predicción
    """
    
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=1,  # balanceado: W~50%, D~25%, L~25%
        )
        self.calibrator = CalibratedClassifierCV(
            self.model,
            method='isotonic',  # mejor que sigmoid para 3 clases
            cv=5
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        X: (n_samples, n_features) → features normalizados [0,1]
        y: (n_samples,) → [0, 1, 2] = [Local, Empate, Visitante]
        
        Return: {accuracy, precision, recall, auc-ovo}
        """
        self.calibrator.fit(X, y)
        return self.evaluate(X, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return: (n_samples, 3) con [P(local), P(empate), P(visitante)]
        Suma a 1.0 por muestra
        """
        return self.calibrator.predict_proba(X)
```

**`models/poisson_simulator.py`**

```python
class PoissonSimulator:
    """
    POR QUÉ POISSON?
    
    1. GOLES SON RAROS: En fútbol, goles ocurren ~2.5-3 por partido
       → Proceso de Poisson es estándar de la industria (Maher 1982)
       → P(X=k goles) = e^(-λ) * λ^k / k!
    
    2. INDEPENDENCIA: Suponemos goles de local ⊥ goles de visitante
       (Xavier-Coles relaxa esto, por ahora OK)
    
    3. OUTPUT: Distribución completa de marcadores, no solo el resultado
       
    PASOS:
    1. XGBoost predice P(local gana) → estima λ_local (parámetro Poisson)
    2. Idem para λ_visitante
    3. Monte Carlo: simula 10,000 partidos con Poisson(λ_local), Poisson(λ_visitante)
    4. Cuenta qué % resulta en victoria/empate/derrota
    """
    
    def estimate_lambda(self, p_win: float, p_draw: float) -> tuple[float, float]:
        """
        Input: P(local gana), P(empate) desde XGBoost
        Output: λ_local, λ_visitante
        
        Usa Dixon-Coles (1997) para invertir probabilidades:
        Históricamente: λ ≈ 1.5 + (0.5 * p_win) para matches europeos
        """
        lambda_home = 1.5 + (0.5 * (p_win - p_draw * 0.5))
        lambda_away = 1.5 - (0.5 * (p_win - p_draw * 0.5))
        return max(lambda_home, 0.1), max(lambda_away, 0.1)
    
    def simulate(self, lambda_home: float, lambda_away: float, n_sims: int = 10000):
        """
        Simula n_sims partidos con Poisson
        Return: {
            "prob_home_win": float,
            "prob_draw": float,
            "prob_away_win": float,
            "most_likely_score": (2, 1),  # Argentina 2 - Marruecos 1
            "score_distribution": {...},  # {(2,1): 0.08, (1,1): 0.06, ...}
        }
        """
        home_goals = np.random.poisson(lambda_home, n_sims)
        away_goals = np.random.poisson(lambda_away, n_sims)
        
        # Contar resultados
        outcomes = np.where(
            home_goals > away_goals, 0,  # Local gana
            np.where(home_goals == away_goals, 1, 2)  # Empate o Visitante
        )
        
        prob_hw, prob_d, prob_aw = np.bincount(outcomes, minlength=3) / n_sims
        
        # Score más frecuente
        scores = list(zip(home_goals, away_goals))
        score_counts = Counter(scores)
        most_likely = score_counts.most_common(1)[0][0]
        
        return {
            "prob_home": prob_hw,
            "prob_draw": prob_d,
            "prob_away": prob_aw,
            "most_likely_score": most_likely,
            "score_distribution": {k: v/n_sims for k, v in score_counts.most_common(10)},
        }
```

**`models/predictor.py`**

Orquestador que une XGBoost + Poisson:

```python
class FootballPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.xgb = load_model(model_path)
        self.scaler = load_scaler(scaler_path)
        self.poisson = PoissonSimulator()
    
    def predict_match(self, home_team: str, away_team: str, 
                      competition: str, date: str) -> dict:
        """
        Input: Argentina, Francia, WC, 2026-12-18
        
        Pasos:
        1. Busca histórico de ambos equipos
        2. Genera 20+ features (form, Elo, cancha, etc)
        3. XGBoost → [P(local), P(empate), P(visitante)]
        4. Poisson Simulator → λ_local, λ_visitante
        5. Monte Carlo 10k iteraciones
        6. SHAP values para explicar
        
        Output:
        {
            "home_team": "Argentina",
            "away_team": "France",
            "xgb_probs": {
                "home_win": 0.52,
                "draw": 0.23,
                "away_win": 0.25
            },
            "poisson_simulation": {
                "most_likely_score": (2, 1),
                "confidence_interval_home": (0, 4),  # 95%
                "prob_home": 0.53,  # revisado con Poisson
                "prob_draw": 0.22,
                "prob_away": 0.25
            },
            "shap_explanation": {
                "home_form_5": +0.15,
                "elo_diff": +0.08,
                "home_advantage": +0.05,
                ...
            },
            "timestamp": "2026-12-15T18:45:00"
        }
        """
```

---

### CLI y Predicción

**`predict.py`**

```bash
python predict.py --home "Argentina" --away "France" --date "2026-12-18"
```

Output (JSON pretty-printed):
```json
{
  "match": "Argentina vs France (WC, 2026-12-18)",
  "predictions": {
    "probability": {
      "home_win": 0.525,
      "draw": 0.225,
      "away_win": 0.250
    },
    "most_likely_score": "Argentina 2 - France 1",
    "score_confidence": {
      "top_3_outcomes": [
        {"score": "2-1", "probability": 0.083},
        {"score": "1-1", "probability": 0.076},
        {"score": "1-0", "probability": 0.068}
      ]
    }
  },
  "explanation": "Argentina is favored due to: recent form (+0.15), Elo advantage (+0.08), home advantage (+0.05)"
}
```

---

## 🎯 Pipeline Completo: De Datos a Predicción

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INGESTIÓN (orchestrator.py)                              │
│    └─ football_data_fetcher.py                              │
│       └─ matches.csv ← API 10 req/min → deduplicado        │
└─────────────────────────────────────────────────────────────┘
                        ↓ (semanal)
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING (preprocessing/features.py)                 │
│    Input: matches.csv (crudo)                               │
│    ├─ Cálculo de form (5 últimos partidos)                  │
│    ├─ Elo diff (eloratings.net, manual por ahora)           │
│    ├─ One-hot encoding (teams, competition)                 │
│    ├─ Normalización (StandardScaler)                        │
│    Output: features.npy (n, 25), target.npy (n,)            │
└─────────────────────────────────────────────────────────────┘
                        ↓ (semanal)
┌─────────────────────────────────────────────────────────────┐
│ 3. ENTRENAMIENTO (models/xgboost_classifier.py)             │
│    ├─ Fit XGBClassifier(n_estimators=100, max_depth=6)      │
│    ├─ Calibración isotónica (3 clases)                      │
│    ├─ Cross-validation 5-fold                               │
│    ├─ Guardado: model_xgboost.pkl, model_scaler.pkl        │
│    Output: Model accuracy ~65-70% (baseline)                │
└─────────────────────────────────────────────────────────────┘
                        ↓ (por predicción)
┌─────────────────────────────────────────────────────────────┐
│ 4. PREDICCIÓN - PASO A: XGBoost (models/predictor.py)       │
│    Input: [Argentina, France, 2026-12-18]                   │
│    ├─ Lookup histórico (últimos 5 partidos)                 │
│    ├─ Genera features                                       │
│    ├─ XGBoost.predict_proba()                               │
│    Output: [P_home=0.52, P_draw=0.23, P_away=0.25]          │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. PREDICCIÓN - PASO B: Poisson + Monte Carlo               │
│    Input: [0.52, 0.23, 0.25]                                │
│    ├─ Invert: λ_home=1.8, λ_away=1.2                        │
│    ├─ Simulate 10,000 matches: Poisson(1.8), Poisson(1.2)  │
│    ├─ Count: goles home, goles away, outcomes               │
│    ├─ Extract: most likely score, confidence interval       │
│    Output: {                                                 │
│      "prob_home": 0.53,  (revisado)                         │
│      "most_likely_score": (2, 1),                           │
│      "score_dist": {...}                                    │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. EXPLICABILIDAD (SHAP)                                    │
│    Input: features del match, XGBoost model                 │
│    ├─ shap.TreeExplainer(model).shap_values(X)              │
│    Output: {"home_form_5": +0.15, "elo_diff": +0.08, ...}  │
│    Respuesta: "Argentina favorita porque: forma (+0.15)..."│
└─────────────────────────────────────────────────────────────┘
                        ↓
                   predict.py → JSON
```

---

## 💾 Datos Históricos Necesarios

Para entrenar el modelo necesitamos ~500+ partidos. Football-data.org ofrece:

| Competición | Seasons | Partidos Aprox |
|---|---|---|
| WC (World Cup) | 2014, 2018, 2022 | 192 |
| CL (Champions) | 2023-24, 2024-25 | 300+ |
| PL (Premier) | 2023-24, 2024-25 | 380 |
| La Liga | 2023-24, 2024-25 | 380 |
| **TOTAL** | | **~1,300** |

**Ingestión**: 1 semana para llenar `matches.csv` (respetando 10 req/min)

---

## 📈 Entrenamiento y Evaluación

### Estrategia Train/Test

```
Data split: 80% train, 20% test (temporalmente: entrena con 2014-2024, test con 2025+)

Métricas:
- Accuracy: ¿qué % de ganador/empate/perdedor predije correctamente?
- Macro Precision/Recall: performance por clase (W, D, L)
- ROC-AUC (OvR): separabilidad de clases

Baseline esperado: 65-70% accuracy
(porque fútbol es caótico; siempre hay upsets)
```

### Validación de Poisson

Para validar que Poisson es bueno:
1. Toma predicciones históricas
2. Compara "marcador predicho" vs "marcador real"
3. Metrics: MAE de goles, % dentro de ±1 gol

---

## 🔄 Ciclo de Operación

### Diariamente

```
03:00 UTC:
  └─ orchestrator.py --mode interval --interval 1440
     ├─ Ejecuta FootballDataFetcher
     ├─ Descarga ~10 partidos nuevos
     ├─ Actualiza matches.csv
     └─ Guarda timestamp en state.json
```

### Semanalmente (opcional)

```
Lunes 00:00 UTC:
  └─ Re-entrena modelo (si hay 100+ matches nuevos)
     ├─ Regenera features
     ├─ Fit XGBoost
     ├─ Guardaa model_xgboost.pkl
```

### On-Demand (CLI)

```
Usuario ejecuta:
  $ python predict.py --home "Argentina" --away "Brazil"
  
  Output: JSON con predicción + explicación
```

---

## 🛠️ Stack Técnico

```
Python 3.9+
├─ pandas      # Manejo de datos
├─ numpy       # Cálculos numerados
├─ scikit-learn # Scalers, CalibratedClassifier
├─ xgboost     # Modelo principal
├─ shap        # Explicabilidad
├─ requests    # HTTP a football-data.org
└─ python-schedule (opcional, para cron)
```

---

## 🚀 Cómo Empezar

### 1. Setup

```bash
cd football_predictor
pip install -r requirements.txt
export FD_API_KEY="tu_api_key_de_football_data"
```

### 2. Primera Ingestión (toma ~1 hora)

```bash
python orchestrator.py --mode once
```

Genera `data/matches.csv` con histórico.

### 3. Entrenar Modelo

```bash
python -c "from models.xgboost_classifier import XGBoostPredictor; 
           xgb = XGBoostPredictor(); 
           xgb.train_from_csv('data/matches.csv')"
```

Genera `data/model_xgboost.pkl` + `data/model_scaler.pkl`

### 4. Predecir

```bash
python predict.py --home "Argentina" --away "France" --date "2026-12-18"
```

### 5. Background Loop (opcional)

```bash
nohup python orchestrator.py --mode interval --interval 1440 > logs/orch.log 2>&1 &
echo $! > orch.pid
```

---

## 📖 Notas Técnicas

### Por qué XGBoost + Poisson (no solo XGBoost)

**XGBoost solo** → P(local gana), P(empate), P(visitante)
- ✅ Rápido, calibrado
- ❌ No da distribución de marcadores
- ❌ No captura incertidumbre inherente

**XGBoost + Poisson** → P(local gana), marcador probable, rango, incertidumbre
- ✅ Full distribución de goles
- ✅ Captura varianza del fútbol
- ✅ Intervalo de confianza (95%)
- ✅ Explora "upsets" (2-0, 0-2)

**Ejemplo**:
- XGBoost: Argentina tiene 52% de ganar
- Poisson: De esos 52%, mayormente 2-1 o 1-0, pero 5% chance de 3-2 loco

---

### Por qué Poisson es mejor que Normal

```
Normal Distribution:        Poisson Distribution:
└─ Goles pueden ser -1 ❌   └─ Goles ≥ 0 ✅
└─ Simétrica, no realista  └─ Asimétrica (eventos raros)
```

En fútbol, 0 goles es ~25%, 1 gol ~35%, 2+ ~40%.
Poisson captura esto. Normal no.

---

### Limitaciones Conocidas

1. **No captura lesiones de último minuto**: Solo usa histórico
2. **Motivación**: Un equipo ya clasificado tira menos
3. **Árbitros**: Inconsistencias en faltas/tarjetas no modeladas
4. **Clima**: Lluvia/viento/altitud no están en features
5. **Home Advantage**: Usamos cancha neutral flag, pero variabilidad es alta

→ Accuracy esperado: **65-70%** (porque fútbol es caótico)

---

## 📚 Referencias

- **Maher MJ (1982)**: "Modelling association football scores" — foundational
- **Dixon-Coles (1997)**: "Modelling association football match scores and inefficiencies"
- **XGBoost Paper**: Chen & Guestrin (2016)
- **SHAP**: Lundberg & Lee (2017)

---

## 📝 Notas para Retomar

Si cierras este README y necesitas recordar:

1. **¿Qué hace cada script?** → Ver sección "Descripción de Archivos"
2. **¿Cómo entreno?** → "Cómo Empezar" paso 3
3. **¿Cómo predigo?** → "Cómo Empezar" paso 4
4. **¿Por qué Poisson + XGBoost?** → "Notas Técnicas"
5. **¿Qué datos tengo?** → `data/matches.csv` después de `orchestrator.py --mode once`

---

## 🔗 Links Rápidos

- Football-Data.org: https://www.football-data.org/
- API Docs: https://www.football-data.org/documentation/quickstart
- XGBoost Docs: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/

---

**Última actualización**: 2026-06-15  
**Estado**: MVP — Ingestión + Features + XGBoost + Poisson  
**Siguiente**: Integración con n8n para alerts, Dashboard con resultados vs predicciones