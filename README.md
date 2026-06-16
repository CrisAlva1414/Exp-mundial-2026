# ⚽ FIFA World Cup 2026 Predictor

Predictor de partidos del Mundial 2026 usando XGBoost + simulación de Poisson, entrenado sobre datos históricos de fútbol internacional desde 1872.

---

## Arquitectura

```
run.py                        ← orquestador principal
│
├── fetchers/
│   ├── kaggle_fetcher.py     ← resultados históricos 1872–2025 (martj42/kaggle)
│   ├── openfootball_fetcher.py ← Mundiales 2010–2026 (openfootball/worldcup.json)
│   ├── eloratings_fetcher.py ← rankings ELO actuales (eloratings.net)
│   └── football_data_fetcher.py ← ligas y torneos (football-data.org API)
│
├── pipeline.py               ← merge + dedup + validación → data/matches.csv
├── model.py                  ← XGBoostPredictor + FeatureBuilder
├── train.py                  ← entrenamiento con split temporal
├── wc2026_predictor.py       ← predicción completa del Mundial 2026
└── data/
    ├── worldcup2026.json     ← fixture oficial WC 2026
    ├── matches.csv           ← dataset unificado (generado)
    └── elo_current.csv       ← rankings ELO actuales
```

---

## Flujo de ejecución

```bash
# Flujo completo (primera vez)
python run.py

# Si los CSVs ya existen, los fetchers se saltan automáticamente
python run.py --skip-fetch

# Solo predecir (modelo ya entrenado)
python run.py --only-predict

# Forzar reentrenamiento
python run.py --force-retrain

# Solo un fetcher específico
python run.py --only elo
```

El pipeline detecta automáticamente si `data/kaggle_results_current.csv`, `data/elo_current.csv` y `data/openfootball_current.csv` existen. Si los tres están presentes, salta directamente a `pipeline.py → train.py → wc2026_predictor.py`.

---

## Fuentes de datos

| Fuente | Contenido | Rows aprox. |
|--------|-----------|-------------|
| Kaggle (martj42) | Partidos internacionales 1872–2025 | ~46,000 |
| OpenFootball | Mundiales 2010–2026 con venues | ~250 |
| football-data.org | Ligas europeas + Brasileirao + EC 2024 | ~8,000 |
| eloratings.net | Rankings ELO actuales por selección | ~200 |

Pipeline de merge: prioridad `football_data > openfootball > kaggle`, dedup por clave `fecha|home|away`.

---

## Features del modelo (30)

| Feature | Descripción |
|---------|-------------|
| `home/away_form_goals_5` | Promedio de goles últimos 5 partidos |
| `home/away_form_pts_5` | Puntos acumulados últimos 5 (W=3, D=1, L=0) |
| `home/away_goals_avg_all` | Promedio histórico de goles |
| `home/away_conceded_avg` | Promedio de goles recibidos |
| `home/away_days_rest` | Días desde último partido |
| `elo_diff` | Diferencia de rating ELO (home − away) |
| `home_advantage` | 1 si hay local definido, 0 si cancha neutral |
| `is_neutral` | Complementario del anterior |
| `home/away_win_rate` | Tasa de victorias últimos 20 partidos |
| `head2head_home_wins` | Ratio de victorias en historial directo |
| `head2head_draws` | Ratio de empates en historial directo |
| `comp_*` (13) | One-hot encoding de competición |

---

## Modelo

**XGBoost + calibración isotónica (CalibratedClassifierCV, cv=5)**

```
n_estimators=300, max_depth=5, learning_rate=0.05
subsample=0.8, colsample_bytree=0.8, min_child_weight=3
```

Split temporal (no aleatorio): 80% train / 20% test, respetando la secuencia cronológica de partidos para evitar data leakage.

**Distribución de clases:**
- Victoria local: 43.3%
- Empate: 26.9%
- Victoria visitante: 29.8%

**Métricas (con datos sintéticos de prueba — mejorarán con datos reales):**

| Split | Accuracy | AUC OvO |
|-------|----------|---------|
| Train | 0.750 | 0.985 |
| Test  | 0.452 | 0.478 |

> Con los datos reales (~46k partidos históricos) se espera convergencia hacia ~0.54–0.58 en test, que es el rango típico para predicción de resultados de fútbol internacional.

---

## Simulación de Poisson

Las probabilidades XGBoost se convierten en lambdas de expected goals usando una aproximación Dixon-Coles (1997):

```python
lambda_home = 1.25 + (p_home - p_away) * 0.8
lambda_away = 1.25 - (p_home - p_away) * 0.8
```

Se simulan 50,000 partidos por Poisson para obtener:
- Distribución completa de resultados exactos
- Score más probable
- Top 8 marcadores con probabilidad

---

## Predicciones WC 2026

El fixture completo está en `data/worldcup2026.json` (104 partidos). El predictor opera sobre los **72 partidos con equipos reales definidos** (los knockouts con placeholders W/L se predicen cuando el fixture se complete).

Salida por partido:
```
2026-06-16  France                 vs  Senegal
           Local 58%  Empate 25%  Visita 17%  → France (58%) | ELO diff: +213
           Score más probable: 1-0  |  λ(1.66 - 0.84)
```

Más resumen de puntos esperados por grupo al final del output.

---

## Instalación

```bash
pip install xgboost scikit-learn pandas numpy kagglehub requests

# Variables de entorno opcionales
export FD_API_KEY="tu_key_football_data_org"   # free tier: 10 req/min
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
```

---

## Notas técnicas

- **Split temporal**: no se usa `train_test_split` aleatorio porque los partidos son secuencias temporales. Shuffle aleatorio introduce leakage (el modelo vería resultados futuros en entrenamiento).
- **ELO como feature**: `elo_diff` captura la calidad relativa de los equipos de forma continua, complementando los features de forma reciente.
- **Neutralidad de cancha**: el Mundial 2026 se juega en USA/México/Canadá, por lo que todos los partidos se marcan como `venue=neutral` en las predicciones.
- **Nombres de equipos**: `NAME_MAP` en `wc2026_predictor.py` normaliza las variantes del fixture a los nombres canónicos del historial (ej. "Bosnia & Herzegovina" → "Bosnia-Herzegovina").

---

## Ejemplo de salida (snapshot 2026-06-16)

Al momento de generar las predicciones del Mundial 2026:

```text
============================================================
FIFA WORLD CUP 2026 — PREDICCIONES COMPLETAS
============================================================
Partidos ya jugados: 16
Partidos a predecir: 56
```

### Salida por partido

```text
2026-06-16  France                 vs  Senegal
         Local 53%  Empate 33%  Visita 14%  → France (53%) | ELO diff: +203
         Score más probable: 1-0  |  λ(1.56 - 0.94)

2026-06-16  Argentina              vs  Algeria
         Local 75%  Empate 16%  Visita 9%  → Argentina (75%) | ELO diff: +343
         Score más probable: 1-0  |  λ(1.77 - 0.72)
```

Cada partido entrega:

- Probabilidad de victoria local, empate y victoria visitante.
- Equipo favorito.
- Diferencia ELO entre ambos equipos.
- Lambdas de goles esperados (`λ_home`, `λ_away`).
- Marcador exacto más probable obtenido mediante simulación de Poisson.

### Resumen por grupos

Al finalizar las predicciones se calcula una tabla de puntos esperados por grupo:

```text
Group J
1. Argentina   6.82 pts esperados (ELO 2115)
2. Austria     3.76 pts esperados (ELO 1830)
3. Algeria     3.48 pts esperados (ELO 1772)
4. Jordan      2.50 pts esperados (ELO 1680)

Group K
1. Portugal    6.00 pts esperados (ELO 1989)
2. Colombia    4.82 pts esperados (ELO 1982)
3. DR Congo    2.86 pts esperados (ELO 1652)
4. Uzbekistan  2.80 pts esperados (ELO 1714)
```

### Estado del pipeline

```text
fetch      ✓ skipped (CSVs present)
pipeline   ✓ 56,363 matches → matches.csv
train      ✓ skipped (model present)
predict    ✓ WC 2026 predictions done
```

```text
2026-06-16 00:56:23 [INFO] Predicted 56 matches
2026-06-16 00:56:23 [INFO] ✅ Done
```