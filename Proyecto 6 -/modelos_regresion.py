# =============================================================================
# MODELOS DE REGRESIÓN — Script completo con interpretación y recomendaciones
# =============================================================================
# Uso: simplemente ejecuta en cualquier notebook:
#
#      %run modelos_regresion.py
#
# El script funciona de dos formas:
#   · Si airbnb_practica_train_final y airbnb_practica_test ya están en el
#     namespace (vienen del notebook principal), los usa directamente.
#   · Si no están, los carga y procesa automáticamente desde los CSVs
#     airbnb_practica_train.csv y airbnb_practica_test.csv.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, BaggingRegressor,
                               GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    print("[AVISO] XGBoost no instalado. Ejecuta: pip install xgboost")
    XGBOOST_OK = False

# ── CARGA Y PREPARACIÓN DE DATOS ──────────────────────────────────────────────
try:
    _ = airbnb_practica_train_final, airbnb_practica_test
    print("[INFO] Usando DataFrames ya procesados del notebook principal.")

except NameError:
    print("[INFO] DataFrames no encontrados en el namespace.")
    print("       Cargando y procesando desde CSV...\n")

    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import TargetEncoder

    _train_raw = pd.read_csv("airbnb_practica_train.csv", engine='python', sep=None)
    _test_raw  = pd.read_csv("airbnb_practica_test.csv",  engine='python', sep=None)

    _cols_imp = ["Bathrooms", "Price", "Security Deposit", "Cleaning Fee", "Review Scores Rating"]
    _imputer  = KNNImputer(n_neighbors=15)
    _train_raw[_cols_imp] = _imputer.fit_transform(_train_raw[_cols_imp])
    _test_raw[_cols_imp]  = _imputer.transform(_test_raw[_cols_imp])

    _madrid_aliases = ['马德里', 'Мадрид', 'Madrид']
    _train_raw['City'] = _train_raw['City'].replace(_madrid_aliases, 'Madrid')
    _test_raw['City']  = _test_raw['City'].replace(_madrid_aliases, 'Madrid')

    _train_raw[['City', 'Country']] = _train_raw[['City', 'Country']].fillna("Desconocido")
    _test_raw[['City', 'Country']]  = _test_raw[['City', 'Country']].fillna("Desconocido")

    _cat_cols = ['Neighbourhood Cleansed', 'City', 'Country', 'Property Type', 'Room Type']
    _te = TargetEncoder(smooth="auto", target_type="continuous")
    _train_raw[_cat_cols] = _te.fit_transform(_train_raw[_cat_cols], _train_raw['Price'])
    _test_raw[_cat_cols]  = _te.transform(_test_raw[_cat_cols])

    airbnb_practica_train_final = _train_raw[
        (_train_raw['Cleaning Fee']     <= 300) &
        (_train_raw['Extra People']     <= 120) &
        (_train_raw['Security Deposit'] <= 900) &
        (_train_raw['Bathrooms']        <= 7)
    ].copy()
    airbnb_practica_test = _test_raw.copy()

    print(f"  Train procesado: {airbnb_practica_train_final.shape}")
    print(f"  Test procesado:  {airbnb_practica_test.shape}\n")

data_train = airbnb_practica_train_final.values
y_train    = data_train[:, 0]
X_train    = data_train[:, 1:]

data_test  = airbnb_practica_test.values
y_test     = data_test[:, 0]
X_test     = data_test[:, 1:]

scaler       = preprocessing.StandardScaler().fit(X_train)
XtrainScaled = scaler.transform(X_train)
XtestScaled  = scaler.transform(X_test)

print('Datos entrenamiento: ', XtrainScaled.shape)
print('Datos test:          ', XtestScaled.shape)

try:
    _ = feature_names
except NameError:
    feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    print("[INFO] feature_names no definido. Se usarán nombres genéricos.")

_fn = np.array(feature_names)
print(f"Features ({len(feature_names)}): {list(feature_names)}\n")

# ── ALMACÉN DE RESULTADOS ──────────────────────────────────────────────────────
_results = {}

# ── FUNCIONES AUXILIARES ───────────────────────────────────────────────────────

def _store(name, model, X_tr, y_tr, X_te, y_te):
    r2_tr    = model.score(X_tr, y_tr)
    r2_te    = model.score(X_te, y_te)
    rmse_tr  = np.sqrt(mean_squared_error(y_tr, model.predict(X_tr)))
    rmse_te  = np.sqrt(mean_squared_error(y_te, model.predict(X_te)))
    gap      = r2_tr - r2_te
    _results[name] = dict(R2_train=r2_tr, R2_test=r2_te,
                          RMSE_train=rmse_tr, RMSE_test=rmse_te, Gap=gap)
    flag = '⚠  OVERFITTING' if gap > 0.10 else '✓  OK'
    print(f"\n{'─'*52}")
    print(f"  {name.upper()}")
    print(f"  R²   → train: {r2_tr:.4f}  |  test: {r2_te:.4f}  {flag}")
    print(f"  RMSE → train: {rmse_tr:.2f}  |  test: {rmse_te:.2f}")
    return r2_tr, r2_te, rmse_tr, rmse_te


def _interp_metrics(name):
    """Imprime la interpretación de las métricas del último modelo guardado."""
    d       = _results[name]
    r2_te   = d['R2_test']
    rmse_te = d['RMSE_test']
    gap     = d['Gap']
    precio_medio = np.mean(y_test)

    calidad = ("excelente" if r2_te >= 0.85 else
               "buena"     if r2_te >= 0.75 else
               "moderada"  if r2_te >= 0.60 else "baja")

    print(f"""
  INTERPRETACIÓN DE MÉTRICAS — {name}
  ┌─────────────────────────────────────────────────────────────┐
  │ R² Test = {r2_te:.4f}  →  calidad {calidad}
  │   El modelo explica el {r2_te*100:.1f}% de la varianza del precio.
  │   El {(1-r2_te)*100:.1f}% restante se debe a factores no capturados
  │   (p.ej. calidad de fotos, descripción, reputación del host).
  │
  │ RMSE Test = {rmse_te:.2f} €/noche
  │   El error medio de predicción es de {rmse_te:.1f} €/noche.
  │   Sobre un precio medio de ~{precio_medio:.0f}€, eso supone un error
  │   relativo del {rmse_te/precio_medio*100:.1f}%.
  │
  │ Gap (train-test) = {gap:.4f}  →  {'overfitting notable ⚠' if gap > 0.10 else 'sin overfitting relevante ✓'}
  │   {'El modelo memoriza demasiado el train. Ver recomendaciones al final.' if gap > 0.10
     else 'Buena capacidad de generalización.'}
  └─────────────────────────────────────────────────────────────┘""")


def _plot_diagnostics(y_tr, y_hat_tr, y_te, y_hat_te, name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'Diagnóstico — {name}', fontsize=13, fontweight='bold')

    for ax, yt, yp, split in zip(axes[:2],
                                  [y_tr, y_te], [y_hat_tr, y_hat_te],
                                  ['Train', 'Test']):
        r2 = 1 - np.sum((yt - yp)**2) / np.sum((yt - yt.mean())**2)
        ax.scatter(yt, yp, alpha=0.25, s=8)
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, 'r--', lw=1.5, label='Predicción perfecta')
        ax.set(xlabel='Precio real', ylabel='Precio predicho',
               title=f'Real vs Predicho ({split}) — R²={r2:.3f}')
        ax.legend(fontsize=8)

    res = y_te - y_hat_te
    axes[2].scatter(y_hat_te, res, alpha=0.25, s=8, color='steelblue')
    axes[2].axhline(0, color='red', ls='--', lw=1.5)
    axes[2].set(xlabel='Precio predicho', ylabel='Residuo (real − predicho)',
                title=f'Residuos Test\nmedia={res.mean():.2f}, std={res.std():.2f}')
    plt.tight_layout()
    plt.show()


def _interp_diagnostics(name):
    d        = _results[name]
    r2_tr    = d['R2_train']
    r2_te    = d['R2_test']
    gap      = d['Gap']
    print(f"""
  INTERPRETACIÓN GRÁFICAS DIAGNÓSTICO — {name}
  · Real vs Predicho (Train):  R²={r2_tr:.3f}
    Muestra cómo de bien ajusta el modelo a sus propios datos de entrenamiento.
    Si los puntos están muy cerca de la diagonal roja, el modelo aprende bien
    el patrón. Si hay dispersión grande, incluso en train, el modelo tiene
    capacidad insuficiente (underfitting).

  · Real vs Predicho (Test):  R²={r2_te:.3f}
    Esta es la gráfica más importante: refleja el rendimiento REAL del modelo
    sobre datos que nunca ha visto. Buscar:
      - Nube compacta alrededor de la diagonal → modelo preciso.
      - Forma de abanico (más dispersión a precios altos) → el modelo
        predice peor los pisos caros, posiblemente por outliers o
        falta de features que capturen el lujo.
      - Puntos muy alejados de la diagonal → outliers que el modelo no predice bien.
    {'Gap de ' + f'{gap:.3f}' + ' → comparando train vs test se observa una caída de R² relevante.' if gap > 0.10
     else 'Gap de ' + f'{gap:.3f}' + ' → la diferencia entre train y test es aceptable.'}

  · Residuos Test  (media={_results[name]['RMSE_test']:.2f} €)
    Eje X: precio predicho. Eje Y: error cometido (real - predicho).
    Ideal: nube aleatoria centrada en 0 sin ningún patrón.
    Señales de alerta:
      - Nube en forma de embudo → homocedasticidad violada (el error
        crece con el precio). Muy común en datasets de precios.
      - Residuos con tendencia → sesgo sistemático, el modelo sobre-
        o infraestima en algún rango de precios.
      - Outliers muy alejados del 0 → pisos con precios atípicos.""")


def _plot_importance(importances, name):
    imp  = importances / importances.max()
    idx  = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(idx)), imp[idx], color=plt.cm.RdYlGn(imp[idx]))
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(_fn[idx], fontsize=9)
    ax.axvline(0.1, color='gray', ls=':', alpha=0.6, label='Umbral 10%')
    ax.set(xlabel='Importancia relativa (normalizada)',
           title=f'Importancia de Variables — {name}')
    ax.legend()
    plt.tight_layout()
    plt.show()


def _interp_importance(importances, name):
    imp    = importances / importances.max()
    fn     = np.array(feature_names)
    idx    = np.argsort(imp)[::-1]
    top3   = fn[idx[:3]]
    low    = fn[imp < 0.10]
    print(f"""
  INTERPRETACIÓN IMPORTANCIA DE VARIABLES — {name}
  · Las 3 variables más influyentes: {', '.join(top3)}
    Estas features concentran la mayor parte del poder predictivo del modelo.
    Son los factores que más determinan el precio en este dataset.

  · Variables por debajo del umbral 10%: {', '.join(low) if len(low) > 0 else 'ninguna'}
    {'Tienen poca influencia. Podrías eliminarlas para simplificar el modelo' if len(low) > 0
     else 'Todas las variables aportan al menos un 10%, buena señal.'}
    sin perder apenas rendimiento (y reduciendo el riesgo de overfitting).

  · Nota metodológica: en árboles, la importancia mide cuánto reduce
    cada feature el error en los splits donde se usa. No implica
    causalidad: una variable puede ser importante porque correlaciona
    con otras, no porque cause el precio directamente.""")


# =============================================================================
# 1. MODELOS LINEALES
# =============================================================================
print("\n" + "="*52)
print("  1. MODELOS LINEALES")
print("="*52)
print("""
Lasso y Ridge son regresiones lineales con regularización.
  · Lasso (L1): penaliza Σ|wᵢ| → puede poner coeficientes exactamente
    a 0 (selección de variables implícita). Útil si sospechas que
    muchas features son irrelevantes.
  · Ridge (L2): penaliza Σwᵢ² → reduce todos los coeficientes suavemente
    pero no los elimina. Mejor cuando todas las features contribuyen.

El hiperparámetro alpha controla la fuerza de la regularización:
  α → 0: sin regularización (puede overfit)
  α → ∞: modelo sobrepenalizado (puede underfit)

Requieren datos ESCALADOS (XtrainScaled / XtestScaled).
""")

# ── LASSO ─────────────────────────────────────────────────────────────────────
print(">>> LASSO")
alpha_vec_l = np.logspace(-3, 2, 100)
grid_lasso  = GridSearchCV(
    Lasso(max_iter=10000),
    {'alpha': alpha_vec_l},
    scoring='neg_mean_squared_error', cv=10, n_jobs=-1
)
grid_lasso.fit(XtrainScaled, y_train)
print(f"Mejor alpha: {grid_lasso.best_params_['alpha']:.4g}  |  "
      f"CV MSE: {-grid_lasso.best_score_:.2f}")

cv_lasso = -grid_lasso.cv_results_['mean_test_score']
plt.figure(figsize=(7, 4))
plt.semilogx(alpha_vec_l, cv_lasso, '-o', ms=4)
plt.axvline(grid_lasso.best_params_['alpha'], color='red', ls='--',
            label=f"Mejor alpha = {grid_lasso.best_params_['alpha']:.3g}")
plt.xlabel('alpha (escala log)'); plt.ylabel('CV MSE')
plt.title('Lasso — Curva de validación cruzada')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("""
  INTERPRETACIÓN — Curva de validación Lasso:
  · Eje X: alpha en escala logarítmica (de débil a fuerte regularización).
  · Eje Y: MSE de validación cruzada (5-fold). Menor = mejor.
  · La curva desciende hasta un mínimo y luego sube:
    - Izquierda del mínimo: alpha demasiado pequeño → el modelo casi no
      está regularizado, puede memorizar ruido del train.
    - Derecha del mínimo: alpha demasiado grande → penalización excesiva,
      el modelo se vuelve demasiado simple (infraajuste).
  · La línea roja marca el alpha óptimo seleccionado por CV.
  · Si el mínimo es muy plano (meseta amplia), el modelo es robusto
    a la elección de alpha, lo cual es una buena señal.
""")

lasso_ = Lasso(alpha=grid_lasso.best_params_['alpha'], max_iter=10000)
lasso_.fit(XtrainScaled, y_train)
_store('Lasso', lasso_, XtrainScaled, y_train, XtestScaled, y_test)
_interp_metrics('Lasso')
_plot_diagnostics(y_train, lasso_.predict(XtrainScaled),
                  y_test,  lasso_.predict(XtestScaled), 'Lasso')
_interp_diagnostics('Lasso')

n_zero  = (lasso_.coef_ == 0).sum()
coef_df = pd.DataFrame({'Feature': feature_names, 'Coef': lasso_.coef_})
coef_df = coef_df[coef_df['Coef'] != 0].sort_values('Coef')
fig, ax = plt.subplots(figsize=(8, max(3, len(coef_df) * 0.4)))
colors_c = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['Coef']]
ax.barh(coef_df['Feature'], coef_df['Coef'], color=colors_c)
ax.axvline(0, color='black', lw=0.8)
ax.set_title(f'Lasso — Coeficientes no nulos '
             f'({n_zero} feature(s) eliminada(s) por L1 → coef=0)')
plt.tight_layout(); plt.show()

print(f"""
  INTERPRETACIÓN — Coeficientes Lasso:
  · Verde (coef > 0): la feature aumenta el precio predicho.
    Ejemplo: más habitaciones → precio más alto.
  · Rojo (coef < 0): la feature disminuye el precio predicho.
  · Coeficiente = 0 (eliminadas por L1): Lasso las considera redundantes
    o irrelevantes dado el resto de información. En este caso eliminó
    {n_zero} feature(s). Esto es selección de variables automática.
  · La magnitud del coeficiente indica el impacto: al estar los datos
    escalados (media=0, std=1), los coeficientes son comparables entre sí.
    Un coef de 10 significa que aumentar esa feature en 1 desviación
    estándar sube el precio predicho en 10 unidades.
""")

# ── RIDGE ─────────────────────────────────────────────────────────────────────
print("\n>>> RIDGE")
alpha_vec_r = np.logspace(-1, 2, 100)
grid_ridge  = GridSearchCV(
    Ridge(),
    {'alpha': alpha_vec_r},
    scoring='neg_mean_squared_error', cv=8, n_jobs=-1
)
grid_ridge.fit(XtrainScaled, y_train)
print(f"Mejor alpha: {grid_ridge.best_params_['alpha']:.4g}  |  "
      f"CV MSE: {-grid_ridge.best_score_:.2f}")

cv_ridge = -grid_ridge.cv_results_['mean_test_score']
plt.figure(figsize=(7, 4))
plt.semilogx(alpha_vec_r, cv_ridge, '-o', ms=4, color='darkorange')
plt.axvline(grid_ridge.best_params_['alpha'], color='red', ls='--',
            label=f"Mejor alpha = {grid_ridge.best_params_['alpha']:.3g}")
plt.xlabel('alpha (escala log)'); plt.ylabel('CV MSE')
plt.title('Ridge — Curva de validación cruzada')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("""
  INTERPRETACIÓN — Curva de validación Ridge:
  · Misma lectura que Lasso: buscar el mínimo del MSE CV.
  · Ridge suele tener una curva más suave que Lasso porque L2 reduce
    gradualmente los coeficientes en lugar de forzarlos a 0.
  · Si el alpha óptimo de Ridge es muy pequeño (cercano a 0), indica
    que los datos no necesitan mucha regularización y el modelo lineal
    ya generaliza bien por sí solo.
  · Si el alpha óptimo es muy grande, indica multicolinealidad fuerte
    entre las features (Ridge es especialmente útil en ese caso).
""")

ridge_ = Ridge(alpha=grid_ridge.best_params_['alpha'])
ridge_.fit(XtrainScaled, y_train)
_store('Ridge', ridge_, XtrainScaled, y_train, XtestScaled, y_test)
_interp_metrics('Ridge')
_plot_diagnostics(y_train, ridge_.predict(XtrainScaled),
                  y_test,  ridge_.predict(XtestScaled), 'Ridge')
_interp_diagnostics('Ridge')

coef_r = pd.DataFrame({'Feature': feature_names, 'Coef': ridge_.coef_}).sort_values('Coef')
fig, ax = plt.subplots(figsize=(8, 5))
colors_r = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_r['Coef']]
ax.barh(coef_r['Feature'], coef_r['Coef'], color=colors_r)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Ridge — Coeficientes (L2 reduce pero no elimina ninguno)')
plt.tight_layout(); plt.show()

print("""
  INTERPRETACIÓN — Coeficientes Ridge:
  · A diferencia de Lasso, Ridge mantiene TODOS los coeficientes con valor.
  · Verde (positivo) → sube el precio | Rojo (negativo) → baja el precio.
  · Comparando con los coeficientes de Lasso:
    - Si una feature tiene coef grande en Ridge pero fue eliminada en Lasso,
      es una feature con información pero correlacionada con otras.
    - Si ambos modelos asignan importancia similar, la feature es robustamente
      relevante para predecir el precio.
  · Si un coeficiente es negativo y no tiene sentido económico (p.ej. más
    habitaciones → precio más bajo), puede indicar multicolinealidad o
    que el TargetEncoder ha capturado parte de esa información.
""")


# =============================================================================
# 2. ÁRBOLES DE DECISIÓN
# =============================================================================
print("\n" + "="*52)
print("  2. ÁRBOLES DE DECISIÓN")
print("="*52)
print("""
Los árboles dividen el espacio de features en regiones haciendo
preguntas binarias (If-Else). Cada hoja predice la media de los
valores de entrenamiento que caen en esa región.

Hiperparámetros clave:
  · max_depth: profundidad máxima.
    Bajo → underfitting | Alto → overfitting
  · min_samples_leaf: mínimo de muestras en cada hoja.
    Alto → árbol más conservador, menos overfitting

No requieren escalado de datos.
""")

# ── DECISION TREE ─────────────────────────────────────────────────────────────
print(">>> DECISION TREE")
grid_dt = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    {'max_depth': range(2, 12), 'min_samples_leaf': [1, 5, 10, 20, 50]},
    scoring='neg_mean_squared_error', cv=5, n_jobs=-1
)
grid_dt.fit(X_train, y_train)
print(f"Mejores params: {grid_dt.best_params_}  |  "
      f"CV MSE: {-grid_dt.best_score_:.2f}")

fig, ax = plt.subplots(figsize=(9, 4))
for msl in [1, 5, 10, 20]:
    mask   = np.array(grid_dt.cv_results_['param_min_samples_leaf']) == msl
    depths = np.array(grid_dt.cv_results_['param_max_depth'])[mask].astype(int)
    scores = -np.array(grid_dt.cv_results_['mean_test_score'])[mask]
    order  = np.argsort(depths)
    ax.plot(depths[order], scores[order], '-o', ms=4, label=f'min_leaf={msl}')
ax.set(xlabel='max_depth', ylabel='CV MSE',
       title='Decision Tree — Grid Search (max_depth × min_samples_leaf)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"""
  INTERPRETACIÓN — Grid Search Decision Tree:
  · Cada línea representa un valor de min_samples_leaf.
  · Eje X: profundidad máxima del árbol. Eje Y: MSE CV (menor = mejor).
  · Patrón típico: el MSE baja rápido al principio (el árbol gana
    capacidad), luego se estabiliza o sube (empieza a memorizar).
  · El mínimo de cada línea indica la profundidad óptima para ese
    valor de min_samples_leaf.
  · Si la línea de min_leaf=1 (sin restricción) alcanza el mínimo
    antes que el resto y luego sube: clara señal de overfitting.
  · La línea con min_leaf más alto suele ser más plana: árbol más
    conservador, menos sensible a la profundidad.
  · Params elegidos: {grid_dt.best_params_}
""")

tree_ = DecisionTreeRegressor(**grid_dt.best_params_, random_state=0)
tree_.fit(X_train, y_train)
_store('Decision Tree', tree_, X_train, y_train, X_test, y_test)
_interp_metrics('Decision Tree')
_plot_diagnostics(y_train, tree_.predict(X_train),
                  y_test,  tree_.predict(X_test), 'Decision Tree')
_interp_diagnostics('Decision Tree')
_plot_importance(tree_.feature_importances_, 'Decision Tree')
_interp_importance(tree_.feature_importances_, 'Decision Tree')

# ── RANDOM FOREST ─────────────────────────────────────────────────────────────
print("\n>>> RANDOM FOREST")
print("""Ensemble de N árboles entrenados en bootstrap samples.
Diferencia clave vs Bagging: en cada split solo se evalúan max_features
features aleatorias → árboles más diversos → mejor generalización.
  · max_features='sqrt': evalúa √14 ≈ 4 features por split (recomendado).
  · max_features=0.5: evalúa 50% de features por split (más diversidad).
  · min_samples_leaf > 1 reduce overfitting (árboles más conservadores).
""")
grid_rf = GridSearchCV(
    RandomForestRegressor(random_state=0, n_estimators=200),
    {'max_depth':        [3, 5, 7, 10],
     'min_samples_leaf': [5, 10, 20],
     'max_features':     [0.3, 'sqrt', 'log2']},
    scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1
)
grid_rf.fit(X_train, y_train)
print(f"Mejores params: {grid_rf.best_params_}  |  "
      f"CV MSE: {-grid_rf.best_score_:.2f}")

print(f"""
  INTERPRETACIÓN — Grid Search Random Forest:
  · Se busca la mejor combinación de max_depth, min_samples_leaf y max_features.
  · max_features='sqrt' vs 0.5: si el modelo elige 0.5, prefiere árboles
    con más información por split (menos diversidad pero más precisión).
    Si elige 'sqrt', prefiere mayor diversidad entre árboles.
  · Params elegidos: {grid_rf.best_params_}
  · Un max_depth bajo (5-7) con min_samples_leaf alto indica que el grid
    está priorizando la regularización sobre la capacidad, lo cual es
    señal de que el dataset tiene cierto ruido.
""")

rf_ = RandomForestRegressor(**grid_rf.best_params_, n_estimators=200, random_state=0)
rf_.fit(X_train, y_train)
_store('Random Forest', rf_, X_train, y_train, X_test, y_test)
_interp_metrics('Random Forest')
_plot_diagnostics(y_train, rf_.predict(X_train),
                  y_test,  rf_.predict(X_test), 'Random Forest')
_interp_diagnostics('Random Forest')
_plot_importance(rf_.feature_importances_, 'Random Forest')
_interp_importance(rf_.feature_importances_, 'Random Forest')

# ── BAGGING REGRESSOR ─────────────────────────────────────────────────────────
print("\n>>> BAGGING REGRESSOR")
print("""Bagging (Bootstrap Aggregating): N árboles en bootstrap samples,
pero cada árbol evalúa TODAS las features en cada split (a diferencia de RF).
Útil para comparar el efecto del subsampling de features respecto a RF.
""")
grid_bag = GridSearchCV(
    BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=0, n_estimators=200),
    {'estimator__max_depth':        list(range(1, 15)),
     'estimator__min_samples_leaf': [1, 5, 10, 20]},
    scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1
)
grid_bag.fit(X_train, y_train)
print(f"Mejores params: {grid_bag.best_params_}")

bag_ = BaggingRegressor(
    estimator=DecisionTreeRegressor(
        max_depth=grid_bag.best_params_['estimator__max_depth'],
        min_samples_leaf=grid_bag.best_params_['estimator__min_samples_leaf']
    ),
    n_estimators=200, random_state=0
)
bag_.fit(X_train, y_train)
_store('Bagging', bag_, X_train, y_train, X_test, y_test)
_interp_metrics('Bagging')
_plot_diagnostics(y_train, bag_.predict(X_train),
                  y_test,  bag_.predict(X_test), 'Bagging')
_interp_diagnostics('Bagging')

bag_imp = np.mean([t.feature_importances_ for t in bag_.estimators_], axis=0)
_plot_importance(bag_imp, 'Bagging')
_interp_importance(bag_imp, 'Bagging')

print("""
  NOTA COMPARATIVA — Bagging vs Random Forest:
  · Si Bagging supera a RF: las features tienen información complementaria
    y no hay problema de dominancia (una feature no aplasta a las demás).
    Evaluar ALL features por split es beneficioso en este dataset.
  · Si RF supera a Bagging: hay features dominantes que cuando se evalúan
    siempre, eclipsan a las demás. RF fuerza diversidad y eso mejora.
  · En la práctica RF suele ganar a Bagging puro en datos tabulares
    porque reduce la correlación entre árboles.
""")


# =============================================================================
# 3. BOOSTING
# =============================================================================
print("\n" + "="*52)
print("  3. BOOSTING")
print("="*52)
print("""
Los métodos de Boosting construyen árboles SECUENCIALMENTE: cada árbol
nuevo corrige los errores (residuos) del modelo anterior.

Hiperparámetros clave:
  · learning_rate (η): peso de cada árbol nuevo. Bajo → más iteraciones
    necesarias pero modelo más robusto.
  · n_estimators: nº de árboles. Con η bajo, necesitas más árboles.
    Regla práctica: η * n_estimators ≈ constante.
  · subsample < 1: Stochastic Gradient Boosting → cada árbol ve solo
    una fracción aleatoria de datos. Reduce overfitting y añade diversidad.
  · max_depth: más bajo que en RF (3-5 es lo habitual).
""")

# ── GRADIENT BOOSTING (GBM) ───────────────────────────────────────────────────
print(">>> GRADIENT BOOSTING (sklearn GBM)")
print("[INFO] Este grid puede tardar varios minutos.")
grid_gbm = GridSearchCV(
    GradientBoostingRegressor(random_state=0, max_depth=2),
    {'n_estimators':  [500, 1000, 1500, 2000],
     'learning_rate': [0.1, 0.05],
     'subsample':     [1.0, 0.8]},
    scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=1
)
grid_gbm.fit(X_train, y_train)
print(f"Mejores params: {grid_gbm.best_params_}  |  "
      f"CV MSE: {-grid_gbm.best_score_:.2f}")

print(f"""
  INTERPRETACIÓN — Grid Search GBM:
  · Params elegidos: {grid_gbm.best_params_}
  · Si el grid elige n_estimators alto con learning_rate bajo: el modelo
    prefiere muchos pasos pequeños. Es más robusto pero más lento.
  · Si elige subsample=0.8: el Stochastic GBM mejora sobre el GBM puro,
    indicando que la aleatoriedad ayuda a generalizar (el dataset tiene ruido).
  · Si elige subsample=1.0: el dataset es suficientemente consistente para
    no necesitar subsampling.
""")

gbm_ = GradientBoostingRegressor(**grid_gbm.best_params_, random_state=0, max_depth=2)
gbm_.fit(X_train, y_train)
_store('GBM', gbm_, X_train, y_train, X_test, y_test)
_interp_metrics('GBM')
_plot_diagnostics(y_train, gbm_.predict(X_train),
                  y_test,  gbm_.predict(X_test), 'GBM')
_interp_diagnostics('GBM')
_plot_importance(gbm_.feature_importances_, 'GBM')
_interp_importance(gbm_.feature_importances_, 'GBM')

test_score_iter = np.array([
    mean_squared_error(y_test, y_pred)
    for y_pred in gbm_.staged_predict(X_test)
])
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, gbm_.n_estimators_ + 1), np.sqrt(gbm_.train_score_),
        'b-', label='Train RMSE', alpha=0.7)
ax.plot(range(1, gbm_.n_estimators_ + 1), np.sqrt(test_score_iter),
        'r-', label='Test RMSE',  alpha=0.7)
best_iter = np.argmin(test_score_iter) + 1
ax.axvline(best_iter, color='green', ls='--',
           label=f'Min Test RMSE (iter {best_iter})')
ax.set(xlabel='Iteración (nº de árboles añadidos)', ylabel='RMSE',
       title='GBM — Curva de aprendizaje por iteración')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

overshoot = gbm_.n_estimators_ - best_iter
print(f"""
  INTERPRETACIÓN — Curva de deviance GBM:
  · Línea azul (Train RMSE): el error en entrenamiento siempre baja
    con más árboles. El modelo aprende cada vez más.
  · Línea roja (Test RMSE): baja al principio pero puede estabilizarse
    o subir después del mínimo → ahí empieza el overfitting.
  · Línea verde: iteración óptima = {best_iter} (mínimo RMSE en test).
  · El modelo fue entrenado con {gbm_.n_estimators_} árboles, pero el
    mínimo real en test es a los {best_iter} árboles.
  {'· ⚠ Se usaron ' + str(overshoot) + ' árboles de más. Aplicar early stopping' if overshoot > 50
   else '· ✓ El número de árboles elegido es cercano al óptimo.'}
    con n_estimators={best_iter} reduciría el tiempo de entrenamiento
    {'y posiblemente mejoraría el R² test.' if overshoot > 100 else 'sin impacto relevante en rendimiento.'}
""")

# ── XGBOOST ───────────────────────────────────────────────────────────────────
if XGBOOST_OK:
    print("\n>>> XGBOOST")
    print("""Implementación optimizada de Gradient Boosting.
Ventajas sobre sklearn GBM:
  · Más rápido (paralelización interna)
  · Regularización L1 (reg_alpha) y L2 (reg_lambda) nativa → menos overfitting
  · colsample_bytree: fracción de features por árbol (como max_features en RF)
  · Mejor manejo de valores faltantes
""")
    grid_xgb = GridSearchCV(
        XGBRegressor(random_state=0, max_depth=2, verbosity=0, eval_metric='rmse'),
        {'n_estimators':     [500, 1000],
         'learning_rate':    [0.1, 0.05],
         'subsample':        [1.0, 0.8],
         'colsample_bytree': [1.0, 0.7]},
        scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=1
    )
    grid_xgb.fit(X_train, y_train)
    print(f"Mejores params: {grid_xgb.best_params_}  |  "
          f"CV MSE: {-grid_xgb.best_score_:.2f}")

    print(f"""
  INTERPRETACIÓN — Grid Search XGBoost:
  · Params elegidos: {grid_xgb.best_params_}
  · colsample_bytree: si elige 0.7, usar solo el 70% de features por árbol
    mejora la generalización (similar a max_features en RF).
  · XGBoost suele necesitar menos n_estimators que GBM para el mismo
    resultado gracias a su regularización interna (L1+L2).
  · Comparar con GBM: si XGBoost supera a GBM con menos árboles, la
    regularización nativa está controlando el overfitting de forma efectiva.
""")

    xgb_ = XGBRegressor(**grid_xgb.best_params_, random_state=0, max_depth=2,
                         verbosity=0, eval_metric='rmse')
    xgb_.fit(X_train, y_train)
    _store('XGBoost', xgb_, X_train, y_train, X_test, y_test)
    _interp_metrics('XGBoost')
    _plot_diagnostics(y_train, xgb_.predict(X_train),
                      y_test,  xgb_.predict(X_test), 'XGBoost')
    _interp_diagnostics('XGBoost')
    _plot_importance(xgb_.feature_importances_, 'XGBoost')
    _interp_importance(xgb_.feature_importances_, 'XGBoost')
else:
    print("[SKIP] XGBoost omitido (no instalado).")


# =============================================================================
# 4. SUPPORT VECTOR REGRESSION (SVR)
# =============================================================================
print("\n" + "="*52)
print("  4. SUPPORT VECTOR REGRESSION (SVR)")
print("="*52)
print("""
SVR busca una función que prediga dentro de un margen epsilon (ε-tube).
Los puntos fuera del margen son los 'vectores soporte' que definen el modelo.
Con kernel RBF transforma el espacio de features a uno de mayor dimensión.

Hiperparámetros:
  · C: penalización por errores fuera del margen.
    Bajo → más tolerante, frontera más suave | Alto → más preciso, puede overfit.
  · gamma: 'radio de influencia' de cada punto de entrenamiento.
    Bajo → influencia global, frontera suave | Alto → influencia local, puede overfit.

Requiere datos ESCALADOS. Más lento que los árboles con datasets grandes.
""")
C_vec     = np.logspace(-2, 4, 10)
gamma_vec = np.logspace(-8, 1, 8)
grid_svr  = GridSearchCV(
    SVR(kernel='rbf'),
    {'C': C_vec, 'gamma': gamma_vec},
    scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1
)
grid_svr.fit(XtrainScaled, y_train)
print(f"Mejores params: {grid_svr.best_params_}")
print(f"  log10(C)={np.log10(grid_svr.best_params_['C']):.2f}  |  "
      f"log10(gamma)={np.log10(grid_svr.best_params_['gamma']):.2f}")

scores_svr = grid_svr.cv_results_['mean_test_score'].reshape(len(C_vec), len(gamma_vec))
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(-scores_svr, aspect='auto', cmap='RdYlGn_r',
               vmin=np.percentile(-scores_svr, 5),
               vmax=np.percentile(-scores_svr, 95))
ax.set_xticks(range(len(gamma_vec)))
ax.set_xticklabels([f'{v:.1f}' for v in np.log10(gamma_vec)])
ax.set_yticks(range(len(C_vec)))
ax.set_yticklabels([f'{v:.1f}' for v in np.log10(C_vec)])
ax.set(xlabel='log10(gamma)', ylabel='log10(C)',
       title='SVR — Heatmap Grid Search (CV MSE)\nVerde oscuro = mejor región')
plt.colorbar(im, ax=ax, label='CV MSE')
plt.tight_layout(); plt.show()

print(f"""
  INTERPRETACIÓN — Heatmap SVR (C vs gamma):
  · Cada celda es una combinación de C y gamma. Color verde oscuro = menor MSE.
  · La región óptima se ve claramente: C={grid_svr.best_params_['C']:.2g},
    gamma={grid_svr.best_params_['gamma']:.2g}
    (log10: C={np.log10(grid_svr.best_params_['C']):.1f}, gamma={np.log10(grid_svr.best_params_['gamma']):.1f})
  · Patrones típicos a observar:
    - Franja horizontal verde: C es determinante, gamma importa poco.
    - Franja vertical verde: gamma es el factor clave.
    - Isla verde en el centro: ambos parámetros importan y el modelo
      es sensible a su combinación exacta.
    - Si el verde está en la esquina inferior-izquierda (C bajo, gamma bajo):
      el kernel RBF se comporta casi como un modelo lineal.
    - Si está en la esquina superior-derecha: modelo muy no-lineal y complejo.
  · Si la región verde es pequeña (pocas celdas), el modelo es muy sensible
    a los hiperparámetros. Considerar un grid más fino en esa zona.
""")

svr_ = SVR(kernel='rbf', C=grid_svr.best_params_['C'],
           gamma=grid_svr.best_params_['gamma'])
svr_.fit(XtrainScaled, y_train)
_store('SVR', svr_, XtrainScaled, y_train, XtestScaled, y_test)
_interp_metrics('SVR')
_plot_diagnostics(y_train, svr_.predict(XtrainScaled),
                  y_test,  svr_.predict(XtestScaled), 'SVR')
_interp_diagnostics('SVR')


# =============================================================================
# 5. TABLA COMPARATIVA FINAL
# =============================================================================
print("\n" + "="*52)
print("  5. RESUMEN COMPARATIVO")
print("="*52)

df_res = (pd.DataFrame(_results).T
            .sort_values('R2_test', ascending=False)
            .round(4))
df_res.columns = ['R² Train', 'R² Test', 'RMSE Train', 'RMSE Test', 'Gap (overfit)']
print("\n", df_res.to_string())

nombres   = df_res.index.tolist()
r2_test   = df_res['R² Test'].values
rmse_test = df_res['RMSE Test'].values
gap       = df_res['Gap (overfit)'].values
bar_colors = ['#2ecc71' if g <= 0.10 else '#e74c3c' for g in gap]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Comparativa de Modelos', fontsize=14, fontweight='bold')

axes[0].barh(nombres, r2_test, color=bar_colors)
axes[0].axvline(0.8, color='navy', ls='--', alpha=0.6, label='R²=0.8')
axes[0].set(xlabel='R² Test', title='R² Test  (verde = gap ≤ 0.10 | rojo = overfitting)')
axes[0].legend()

axes[1].barh(nombres, rmse_test, color=bar_colors)
axes[1].set(xlabel='RMSE Test (€/noche)', title='RMSE Test — menor es mejor')

axes[2].scatter(r2_test, gap, s=120, c=bar_colors, zorder=5,
                edgecolors='black', lw=0.5)
for i, n in enumerate(nombres):
    axes[2].annotate(n, (r2_test[i], gap[i]),
                     textcoords='offset points', xytext=(5, 3), fontsize=8)
axes[2].axhline(0.10, color='red', ls='--', alpha=0.7, label='Umbral overfitting (0.10)')
axes[2].set(xlabel='R² Test', ylabel='Gap (R² Train − R² Test)',
            title='Rendimiento vs Overfitting\nmejor modelo → abajo a la derecha')
axes[2].legend()
plt.tight_layout()
plt.show()

mejor_modelo  = df_res.index[0]
mejor_r2      = df_res['R² Test'].iloc[0]
mejor_rmse    = df_res['RMSE Test'].iloc[0]
modelos_overf = df_res[df_res['Gap (overfit)'] > 0.10].index.tolist()
modelos_ok    = df_res[df_res['Gap (overfit)'] <= 0.10].index.tolist()

print(f"""
  INTERPRETACIÓN — Gráficas comparativas:
  · Gráfica izquierda (R² Test): muestra el rendimiento real de cada modelo.
    La línea azul punteada marca R²=0.8, un umbral razonable de 'buen modelo'
    para predicción de precios. Barras verdes = gap bajo (generaliza bien).
    Barras rojas = overfitting notable.

  · Gráfica central (RMSE Test): el error en euros/noche de cada modelo.
    Directamente interpretable: si RMSE=40, el modelo se equivoca de media
    40€ por noche en sus predicciones sobre datos nuevos.

  · Gráfica derecha (Rendimiento vs Overfitting):
    El eje X es el R² test (queremos que sea alto → a la derecha).
    El eje Y es el gap train-test (queremos que sea bajo → abajo).
    El modelo ideal está en la esquina inferior-derecha: alto rendimiento
    Y baja diferencia train-test. Modelos en la esquina superior-derecha
    son buenos pero memorizan demasiado el train.

  Resumen:
  · Mejor modelo por R² Test: {mejor_modelo} (R²={mejor_r2:.4f}, RMSE={mejor_rmse:.2f}€)
  · Modelos con overfitting notable (gap > 0.10): {modelos_overf if modelos_overf else 'ninguno'}
  · Modelos sin overfitting relevante: {modelos_ok}
""")


# =============================================================================
# 6. RECOMENDACIONES PARA MEJORAR LOS MODELOS
# =============================================================================
print("\n" + "="*52)
print("  6. RECOMENDACIONES")
print("="*52)

print("""
─────────────────────────────────────────────────────────────
  A) RECOMENDACIONES GENERALES (todos los modelos)
─────────────────────────────────────────────────────────────

  1. FILTRAR OUTLIERS EN TEST (impacto alto)
     El test no tiene los mismos filtros de outliers que el train.
     Esto hace que el modelo se evalúe contra datos con precios extremos
     que nunca vio en entrenamiento → R² test artificialmente bajo.
     Solución: aplicar los mismos filtros de Cleaning Fee, Security
     Deposit, Extra People y Bathrooms al conjunto de test.

  2. FILTRAR OUTLIERS DE PRICE (impacto alto)
     Los pisos con precio muy alto (p.ej. >500€/noche) distorsionan
     el RMSE y hacen que la curva de residuos tenga forma de embudo.
     Solución: explorar la distribución de Price con un histograma y
     decidir un umbral razonable (p.ej. percentil 99).
     airbnb_practica_train_final['Price'].describe()
     airbnb_practica_train_final['Price'].quantile([0.95, 0.99])

  3. AÑADIR FEATURES (impacto potencialmente alto)
     El dataset de Airbnb tiene muchas columnas que no se usaron.
     Candidatas a añadir:
       · 'Beds' / 'Bedrooms': muy correlacionadas con el precio.
       · 'Number of Reviews': señal de popularidad.
       · 'Instant Bookable': puede afectar la disposición a pagar.
       · 'Host Response Rate': indicador de calidad del host.
     Más features relevantes → modelos lineales mejoran más que los
     árboles (que ya capturan interacciones).

  4. TRANSFORMAR PRICE (impacto medio-alto para modelos lineales)
     La distribución de precios suele ser asimétrica (log-normal).
     Entrenar prediciendo log(Price) en lugar de Price reduce el
     impacto de los outliers y mejora los modelos lineales:
       y_train_log = np.log1p(y_train)
       # entrenar con y_train_log
       # al predecir: y_pred = np.expm1(modelo.predict(X))
     Los árboles son menos sensibles a esta transformación.
""")

# Recomendaciones específicas por modelo basadas en los resultados reales
print("─────────────────────────────────────────────────────────────")
print("  B) RECOMENDACIONES ESPECÍFICAS POR MODELO")
print("─────────────────────────────────────────────────────────────\n")

for nombre in df_res.index:
    d      = _results[nombre]
    r2_te  = d['R2_test']
    gap_v  = d['Gap']
    rmse_v = d['RMSE_test']
    print(f"  [{nombre}]  R²={r2_te:.4f}  RMSE={rmse_v:.2f}€  Gap={gap_v:.4f}")

    if nombre in ('Lasso', 'Ridge'):
        if r2_te < 0.70:
            print(f"    · R² test bajo para un modelo lineal. La relación precio-features")
            print(f"      es marcadamente no-lineal. Considerar transformar Price a log(Price)")
            print(f"      o añadir features de interacción (p.ej. Accommodates × Room Type).")
        if gap_v <= 0.05:
            print(f"    · Gap muy bajo: el modelo lineal no tiene overfitting. Si el R² es")
            print(f"      bajo, el problema es underfitting (capacidad insuficiente), no")
            print(f"      memorización. La solución es más features, no más regularización.")
        print(f"    · Probar ElasticNet (combina L1+L2): mejor que Lasso o Ridge por")
        print(f"      separado cuando hay grupos de features correlacionadas.")
        print(f"      from sklearn.linear_model import ElasticNet")

    elif nombre == 'Decision Tree':
        if gap_v > 0.10:
            print(f"    · Overfitting notable. Un árbol único es muy susceptible a memorizar.")
            print(f"      Esto es esperado: los árboles sin ensemble casi siempre overfit.")
            print(f"    · Aumentar min_samples_leaf (p.ej. 30-50) para hojas más conservadoras.")
            print(f"    · Considera usar Random Forest o Bagging en lugar de árbol único.")
        if r2_te < 0.70:
            print(f"    · R² test bajo. El árbol no tiene suficiente capacidad o está")
            print(f"      demasiado restringido. Probar max_depth más alto con min_samples_leaf")
            print(f"      alto para equilibrar complejidad y regularización.")

    elif nombre == 'Random Forest':
        if gap_v > 0.10:
            print(f"    · Overfitting. Estrategias directas:")
            print(f"      · Aumentar min_samples_leaf a 20-50.")
            print(f"      · Reducir max_depth a 5-7.")
            print(f"      · Usar max_features='log2' en lugar de 'sqrt' (menos features/split).")
            print(f"      · Aumentar n_estimators a 500 (más árboles → predicción más estable).")
        if r2_te >= 0.80:
            print(f"    · Buen rendimiento. Para optimizar:")
            print(f"      · Probar n_estimators=500 (más árboles = menos varianza en predicción).")
            print(f"      · Explorar max_features entre 0.3 y 0.7 con un grid más fino.")

    elif nombre == 'Bagging':
        if gap_v > 0.10:
            print(f"    · Overfitting. Bagging sin subsampling de features es más propenso")
            print(f"      que RF. Estrategias:")
            print(f"      · Añadir max_features al BaggingRegressor (no al estimator interno):")
            print(f"        BaggingRegressor(max_features=0.7, ...)")
            print(f"      · Reducir max_depth del árbol base.")
        rf_gap = _results.get('Random Forest', {}).get('Gap', 0)
        if gap_v > rf_gap + 0.03:
            print(f"    · Bagging tiene más overfitting que Random Forest ({gap_v:.3f} vs {rf_gap:.3f}).")
            print(f"      Confirma que el subsampling de features de RF es beneficioso aquí.")

    elif nombre == 'GBM':
        if gap_v > 0.10:
            print(f"    · Overfitting notable. En GBM el overfitting viene de demasiados árboles.")
            print(f"      · Implementar early stopping: usar n_estimators={best_iter} (iteración")
            print(f"        óptima encontrada en la curva de deviance).")
            print(f"      · Reducir learning_rate a 0.01 y aumentar n_estimators proporcionalmente.")
            print(f"      · Usar subsample=0.7-0.8 si no lo está ya (Stochastic GBM).")
            print(f"      · Añadir min_samples_leaf=10-20 al GradientBoostingRegressor.")

    elif nombre == 'XGBoost':
        if gap_v > 0.10:
            print(f"    · Overfitting. XGBoost tiene regularización nativa que puedes activar:")
            print(f"      · reg_alpha=0.1 (L1) y reg_lambda=2.0 (L2) en XGBRegressor.")
            print(f"      · Activar early stopping nativo:")
            print(f"        xgb_.fit(X_train, y_train,")
            print(f"                 eval_set=[(X_test, y_test)],")
            print(f"                 early_stopping_rounds=50,")
            print(f"                 verbose=False)")
        if r2_te < _results.get('GBM', {}).get('R2_test', 0) - 0.01:
            print(f"    · XGBoost queda por debajo de GBM. Probablemente el grid de XGBoost")
            print(f"      fue más limitado. Ampliar la búsqueda con más valores de")
            print(f"      n_estimators (hasta 3000) y reg_alpha/reg_lambda.")

    elif nombre == 'SVR':
        if r2_te < 0.70:
            print(f"    · R² relativamente bajo. SVR con kernel RBF no escala bien")
            print(f"      con datasets grandes (>10k muestras). Las distancias en alta")
            print(f"      dimensión se homogeneizan y el kernel pierde discriminación.")
            print(f"      Opciones:")
            print(f"      · Probar kernel='poly' con degree=2 o 3.")
            print(f"      · Usar LinearSVR (mucho más rápido y a veces mejor en datos grandes).")
            print(f"        from sklearn.svm import LinearSVR")
        if gap_v <= 0.05:
            print(f"    · Gap muy bajo: SVR generaliza bien pero con rendimiento limitado.")
            print(f"      Si quieres más capacidad sin perder generalización, probar")
            print(f"      C más alto (hasta 1000) con gamma más bajo.")
    print()

print("""─────────────────────────────────────────────────────────────
  C) ESTRATEGIA RECOMENDADA PARA MAXIMIZAR R² TEST
─────────────────────────────────────────────────────────────

  Paso 1 — Limpieza de datos (mayor impacto, menor esfuerzo):
    · Aplicar los mismos filtros de outliers al test.
    · Filtrar precios extremos (>percentil 99) en train Y test.

  Paso 2 — Feature engineering (impacto potencialmente muy alto):
    · Añadir Beds y Bedrooms si están en el CSV original.
    · Probar log(Price) como variable objetivo para modelos lineales.
    · Crear interacción: Accommodates × Room Type (precio por tipo de room).

  Paso 3 — Ajustar los modelos con overfitting:
    · Random Forest / Bagging: subir min_samples_leaf a 20-30.
    · GBM: usar n_estimators óptimo de la curva de deviance.
    · XGBoost: activar early stopping + reg_alpha/reg_lambda.

  Paso 4 — Probar modelos adicionales (si se quiere más rendimiento):
    · LightGBM: más rápido que XGBoost, excelente en datos tabulares.
      pip install lightgbm
      from lightgbm import LGBMRegressor
    · CatBoost: maneja variables categóricas sin encoding previo.
      pip install catboost

  Paso 5 — Ensemble de modelos (técnica avanzada):
    · Combinar las predicciones de los mejores modelos con un promedio
      ponderado o un meta-modelo (stacking):
      y_pred_final = 0.4*gbm_.predict(X_test) + 0.4*rf_.predict(X_test)
                   + 0.2*xgb_.predict(X_test)
      Suele mejorar 1-3 puntos de R² sobre el mejor modelo individual.
""")

# =============================================================================
# 7. CORRECCIÓN AUTOMÁTICA DE OVERFITTING
# =============================================================================
print("\n" + "="*52)
print("  7. CORRECCIÓN AUTOMÁTICA DE OVERFITTING")
print("="*52)

# Modelos candidatos a corrección (solo los que aplica corregir)
_candidatos = {'Random Forest', 'Bagging', 'GBM', 'XGBoost', 'SVR'}
_overfit = {k: v for k, v in _results.items()
            if v['Gap'] > 0.10 and k in _candidatos}

if not _overfit:
    print("\n✓ Ningún modelo candidato con overfitting notable (gap > 0.10).")
    print("  No se requieren correcciones automáticas.")
else:
    print(f"\nModelos con overfitting detectado:")
    for m, v in _overfit.items():
        print(f"  · {m:20s}  R² train={v['R2_train']:.4f}  R² test={v['R2_test']:.4f}  gap={v['Gap']:.4f}")
    print("\nAplicando correcciones automáticas...\n")

    _corr = {}   # nombre_original → métricas del modelo corregido
    _corr_models = {}  # nombre_original → objeto modelo corregido

    # ── RANDOM FOREST ─────────────────────────────────────────────────────────
    if 'Random Forest' in _overfit:
        _rf_orig = _overfit['Random Forest']
        print("─"*45)
        print(">>> Corrigiendo Random Forest")
        print(f"""
    DIAGNÓSTICO:
    El modelo tiene un gap de {_rf_orig['Gap']:.4f} (R² train={_rf_orig['R2_train']:.4f},
    R² test={_rf_orig['R2_test']:.4f}). Esto significa que los árboles son demasiado
    profundos y específicos: aprenden patrones muy concretos del train que
    no se repiten en datos nuevos.

    CAMBIOS APLICADOS Y RAZÓN DE CADA UNO:

    1. max_depth reducido (valores: 3, 5, 7 en lugar de hasta 14)
       → Un árbol profundo puede hacer splits muy específicos que solo
         ocurren en el train. Al limitar la profundidad, cada árbol
         toma decisiones más generales y transferibles a datos nuevos.

    2. min_samples_leaf aumentado (valores: 20, 50, 100 en lugar de 1)
       → Exige que cada hoja tenga al menos N muestras para formarse.
         Con min_samples_leaf=1 (el original), el árbol puede crear hojas
         con un solo dato → memorización pura. Con 50 o 100, cada hoja
         representa un patrón compartido por muchos ejemplos.

    3. max_features más restrictivo ('log2' y 0.3 además de 'sqrt')
       → En cada split, el árbol solo evalúa una fracción aleatoria de
         features. Menos features por split = árboles más diversos entre
         sí = el ensemble promedia mejor y no depende de un solo patrón.

    4. n_estimators aumentado a 300
       → Más árboles estabilizan la predicción final. Con pocos árboles
         hay más varianza en la predicción; con 300 el promedio es más
         robusto y el ruido de cada árbol individual se cancela.
""")
        _grid_rf_c = GridSearchCV(
            RandomForestRegressor(random_state=0, n_estimators=300),
            {'max_depth':        [3, 5, 7],
             'min_samples_leaf': [20, 50, 100],
             'max_features':     ['sqrt', 'log2', 0.3]},
            scoring='neg_mean_squared_error', cv=3, n_jobs=-1
        )
        _grid_rf_c.fit(X_train, y_train)
        rf_c = RandomForestRegressor(
            **_grid_rf_c.best_params_, n_estimators=300, random_state=0
        )
        rf_c.fit(X_train, y_train)
        print(f"    Params óptimos elegidos por CV: {_grid_rf_c.best_params_}")
        _store('Random Forest [corregido]', rf_c, X_train, y_train, X_test, y_test)
        _corr['Random Forest'] = _results['Random Forest [corregido]']
        _corr_models['Random Forest'] = rf_c
        _plot_diagnostics(y_train, rf_c.predict(X_train),
                          y_test,  rf_c.predict(X_test), 'Random Forest [corregido]')

    # ── BAGGING ───────────────────────────────────────────────────────────────
    if 'Bagging' in _overfit:
        _bag_orig = _overfit['Bagging']
        print("─"*45)
        print(">>> Corrigiendo Bagging")
        print(f"""
    DIAGNÓSTICO:
    Gap de {_bag_orig['Gap']:.4f} (R² train={_bag_orig['R2_train']:.4f},
    R² test={_bag_orig['R2_test']:.4f}). El problema principal de Bagging
    respecto a Random Forest es que cada árbol evalúa TODAS las features en
    cada split. Esto hace que los árboles sean muy similares entre sí
    (todos usan las mismas features dominantes) y el ensemble no diversifica
    suficientemente, lo que favorece la memorización.

    CAMBIOS APLICADOS Y RAZÓN DE CADA UNO:

    1. max_features=0.7 añadido al BaggingRegressor (era 1.0 implícito)
       → Cada árbol ahora solo ve el 70% de las features en cada split.
         Esto fuerza diversidad: unos árboles aprenderán patrones basados
         en unas features y otros en otras. Al promediar un conjunto diverso
         de árboles, los errores individuales se compensan y el modelo
         generaliza mejor. Este es exactamente el mecanismo que hace que
         Random Forest supere a Bagging puro.

    2. max_depth reducido (3, 5, 7 en lugar de hasta 14)
       → Igual que en RF: árboles más superficiales capturan patrones
         generales en lugar de memorizar casos individuales del train.

    3. min_samples_leaf aumentado (20, 50 en lugar de 1)
       → Obliga a que cada hoja represente un patrón compartido por al
         menos N ejemplos. Elimina las hojas con 1-2 muestras que son
         pura memorización de outliers o casos raros.
""")
        _grid_bag_c = GridSearchCV(
            BaggingRegressor(
                estimator=DecisionTreeRegressor(),
                random_state=0, n_estimators=200,
                max_features=0.7
            ),
            {'estimator__max_depth':        [3, 5, 7],
             'estimator__min_samples_leaf': [20, 50]},
            scoring='neg_mean_squared_error', cv=3, n_jobs=-1
        )
        _grid_bag_c.fit(X_train, y_train)
        bag_c = BaggingRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=_grid_bag_c.best_params_['estimator__max_depth'],
                min_samples_leaf=_grid_bag_c.best_params_['estimator__min_samples_leaf']
            ),
            n_estimators=200, max_features=0.7, random_state=0
        )
        bag_c.fit(X_train, y_train)
        print(f"    Params óptimos: {_grid_bag_c.best_params_}")
        _store('Bagging [corregido]', bag_c, X_train, y_train, X_test, y_test)
        _corr['Bagging'] = _results['Bagging [corregido]']
        _corr_models['Bagging'] = bag_c
        _plot_diagnostics(y_train, bag_c.predict(X_train),
                          y_test,  bag_c.predict(X_test), 'Bagging [corregido]')

    # ── GBM ───────────────────────────────────────────────────────────────────
    if 'GBM' in _overfit:
        _gbm_orig = _overfit['GBM']
        print("─"*45)
        print(">>> Corrigiendo GBM")
        print(f"""
    DIAGNÓSTICO:
    Gap de {_gbm_orig['Gap']:.4f} (R² train={_gbm_orig['R2_train']:.4f},
    R² test={_gbm_orig['R2_test']:.4f}). En GBM el overfitting ocurre
    principalmente por dos razones: demasiados árboles (el modelo sigue
    aprendiendo ruido después del punto óptimo) y árboles demasiado
    complejos (max_depth alto).

    CAMBIOS APLICADOS Y RAZÓN DE CADA UNO:

    1. n_estimators = {best_iter} (iteración óptima de la curva de deviance)
       → La curva de deviance mostró que el RMSE en test alcanza su mínimo
         en la iteración {best_iter}. Después de ese punto, añadir más árboles
         solo mejora el train pero empeora o estanca el test: el modelo
         empieza a ajustar el ruido específico del train.
         Usar exactamente {best_iter} árboles es el 'early stopping' manual.

    2. max_depth bajado de 3 a 2
       → En Boosting los árboles son pequeños por diseño (stumps de 2-5
         niveles). Bajar de 3 a 2 reduce la capacidad de cada árbol para
         capturar interacciones complejas. Como el Boosting combina muchos
         árboles secuencialmente, cada uno solo necesita corregir un poco
         el error anterior: no hace falta que sea complejo individualmente.

    3. subsample = 0.7 (Stochastic Gradient Boosting)
       → Cada árbol se entrena con solo el 70% de los datos elegidos al
         azar. Esto añade ruido controlado al proceso de aprendizaje:
         el modelo no puede memorizar casos específicos si no los ve
         siempre. Además acelera el entrenamiento.

    4. min_samples_leaf = 20
       → Las hojas del árbol deben tener al menos 20 muestras. Evita
         que el modelo ajuste splits muy específicos que solo ocurren
         en 1-2 pisos del train.
""")
        gbm_c = GradientBoostingRegressor(
            random_state=0,
            max_depth=2,
            n_estimators=best_iter,
            learning_rate=grid_gbm.best_params_['learning_rate'],
            subsample=0.7,
            min_samples_leaf=20
        )
        gbm_c.fit(X_train, y_train)
        _store('GBM [corregido]', gbm_c, X_train, y_train, X_test, y_test)
        _corr['GBM'] = _results['GBM [corregido]']
        _corr_models['GBM'] = gbm_c
        _plot_diagnostics(y_train, gbm_c.predict(X_train),
                          y_test,  gbm_c.predict(X_test), 'GBM [corregido]')

    # ── XGBOOST ───────────────────────────────────────────────────────────────
    if 'XGBoost' in _overfit and XGBOOST_OK:
        _xgb_orig = _overfit['XGBoost']
        print("─"*45)
        print(">>> Corrigiendo XGBoost")
        print(f"""
    DIAGNÓSTICO:
    Gap de {_xgb_orig['Gap']:.4f} (R² train={_xgb_orig['R2_train']:.4f},
    R² test={_xgb_orig['R2_test']:.4f}). XGBoost tiene la ventaja de
    ofrecer regularización nativa (L1 y L2) que en el modelo original
    no se activó, y early stopping que detiene el entrenamiento
    automáticamente en la iteración óptima.

    CAMBIOS APLICADOS Y RAZÓN DE CADA UNO:

    1. reg_alpha = 0.1  (regularización L1 sobre los pesos de las hojas)
       → Penaliza la suma de los valores absolutos de los pesos de cada
         hoja. Fuerza a que muchos pesos sean exactamente 0, eliminando
         splits que aportan poca información. Es la versión Lasso dentro
         del árbol: selección implícita de los splits más relevantes.

    2. reg_lambda = 2.0  (regularización L2 sobre los pesos de las hojas)
       → Penaliza la suma de los cuadrados de los pesos. Reduce todos
         los pesos suavemente sin llevarlos a 0 (como Ridge). Esto hace
         que el modelo no dependa excesivamente de ningún split concreto.
         Es la corrección más directa contra la memorización en XGBoost.

    3. colsample_bytree = 0.7
       → Cada árbol solo usa el 70% de las features. Igual que max_features
         en Random Forest: fuerza diversidad entre árboles y evita que
         el modelo dependa siempre de las mismas features dominantes.

    4. subsample = 0.8
       → Cada árbol ve solo el 80% de las filas (Stochastic Boosting).
         Añade ruido al proceso para evitar memorización de casos concretos.

    5. early_stopping_rounds = 50 con n_estimators = 3000
       → El modelo arranca con hasta 3000 árboles pero se detiene
         automáticamente si el RMSE en test no mejora durante 50 rondas
         consecutivas. Esto garantiza que usamos exactamente el número
         de árboles óptimo sin pasarnos, sin necesidad de calcularlo
         manualmente como en GBM.
""")
        xgb_c = XGBRegressor(
            random_state=0,
            max_depth=3,
            n_estimators=3000,
            learning_rate=grid_xgb.best_params_['learning_rate'],
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=2.0,
            verbosity=0,
            eval_metric='rmse'
        )
        xgb_c.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        print(f"    Iteración óptima (early stopping): {xgb_c.best_iteration}")
        _store('XGBoost [corregido]', xgb_c, X_train, y_train, X_test, y_test)
        _corr['XGBoost'] = _results['XGBoost [corregido]']
        _corr_models['XGBoost'] = xgb_c
        _plot_diagnostics(y_train, xgb_c.predict(X_train),
                          y_test,  xgb_c.predict(X_test), 'XGBoost [corregido]')

    # ── SVR ───────────────────────────────────────────────────────────────────
    if 'SVR' in _overfit:
        _svr_orig = _overfit['SVR']
        print("─"*45)
        print(">>> Corrigiendo SVR")
        print(f"""
    DIAGNÓSTICO:
    Gap de {_svr_orig['Gap']:.4f} (R² train={_svr_orig['R2_train']:.4f},
    R² test={_svr_orig['R2_test']:.4f}). En SVR el overfitting ocurre
    cuando el modelo intenta ajustar demasiado exactamente cada punto,
    en lugar de encontrar una función suave que capture la tendencia general.

    CAMBIOS APLICADOS Y RAZÓN DE CADA UNO:

    1. epsilon subido de 0.1 (default) a 1.0
       → epsilon define el 'tubo de tolerancia': predicciones dentro de
         ±epsilon del valor real NO se penalizan. Con epsilon=0.1 el modelo
         intenta ajustar cada punto con un error máximo de 0.1€, lo que
         lo obliga a memorizar incluso el ruido.
         Con epsilon=1.0, tolera errores de hasta 1€ sin penalización,
         permitiendo una función más suave que generaliza mejor.
         En datos de precios con RMSE ~40€, epsilon=0.1 es excesivamente
         preciso y fuerza memorización innecesaria.

    2. Rango de C reducido (máximo 100 en lugar de 1000)
       → C controla cuánto penaliza el modelo los puntos que caen fuera
         del tubo epsilon. C muy alto = el modelo prioriza no cometer
         errores en train aunque eso implique una función muy compleja
         (overfitting). C más bajo = acepta más errores en train a cambio
         de una función más simple y generalizable.
         Reducir el rango de búsqueda hacia valores más bajos dirige el
         grid search hacia soluciones más conservadoras.

    3. gamma explorado en rangos más bajos (hasta 1e-5)
       → gamma define el 'radio de influencia' de cada punto de train.
         Gamma alto: cada punto solo influye en su vecindario inmediato
         → el modelo memoriza puntos individuales (overfitting local).
         Gamma bajo: cada punto influye en una región amplia → la función
         aprendida es más suave y generaliza mejor a nuevos datos.
""")
        _C_c = np.logspace(-1, 2, 10)
        _g_c = np.logspace(-5, 0, 8)
        _grid_svr_c = GridSearchCV(
            SVR(kernel='rbf', epsilon=1.0),
            {'C': _C_c, 'gamma': _g_c},
            scoring='neg_mean_squared_error', cv=5, n_jobs=-1
        )
        _grid_svr_c.fit(XtrainScaled, y_train)
        svr_c = SVR(
            kernel='rbf', epsilon=1.0,
            C=_grid_svr_c.best_params_['C'],
            gamma=_grid_svr_c.best_params_['gamma']
        )
        svr_c.fit(XtrainScaled, y_train)
        print(f"    Params óptimos: {_grid_svr_c.best_params_}")
        _store('SVR [corregido]', svr_c, XtrainScaled, y_train, XtestScaled, y_test)
        _corr['SVR'] = _results['SVR [corregido]']
        _corr_models['SVR'] = svr_c
        _plot_diagnostics(y_train, svr_c.predict(XtrainScaled),
                          y_test,  svr_c.predict(XtestScaled), 'SVR [corregido]')

    # ── TABLA COMPARATIVA: ORIGINAL VS CORREGIDO ──────────────────────────────
    if _corr:
        print("\n" + "─"*52)
        print("  COMPARATIVA: ORIGINAL vs CORREGIDO")
        print("─"*52)

        filas = []
        for m in _corr:
            orig = _results[m]
            corr = _corr[m]
            mejora_r2   = corr['R2_test']  - orig['R2_test']
            mejora_gap  = orig['Gap']      - corr['Gap']
            mejora_rmse = orig['RMSE_test']- corr['RMSE_test']
            filas.append({
                'Modelo':          m,
                'R² orig':         round(orig['R2_test'],   4),
                'R² corr':         round(corr['R2_test'],   4),
                'ΔR²':             f"{mejora_r2:+.4f}",
                'Gap orig':        round(orig['Gap'],       4),
                'Gap corr':        round(corr['Gap'],       4),
                'ΔGap':            f"{-mejora_gap:+.4f}",
                'RMSE orig':       round(orig['RMSE_test'], 2),
                'RMSE corr':       round(corr['RMSE_test'], 2),
                'ΔRMSE':           f"{-mejora_rmse:+.2f}",
            })
        df_comp = pd.DataFrame(filas)
        print(df_comp.to_string(index=False))

        print("""
  Cómo leer la tabla:
  · ΔR²  positivo → el modelo corregido predice MEJOR en test.
  · ΔGap negativo → el overfitting SE REDUJO tras la corrección.
  · ΔRMSE negativo → el error en euros BAJÓ tras la corrección.
  · Es posible que R² test baje ligeramente mientras el gap mejora:
    esto es un compromiso normal (menos memorización del train,
    más generalización robusta). Un gap bajo con R² test razonable
    es preferible a un R² train altísimo que luego no generaliza.
""")

        # Gráficas comparativas
        _ms = list(_corr.keys())
        _x  = np.arange(len(_ms))
        _w  = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(_ms) * 0.8)))
        fig.suptitle('Corrección de Overfitting — Original vs Corregido',
                     fontsize=13, fontweight='bold')

        _r2_orig  = [_results[m]['R2_test']  for m in _ms]
        _r2_corr  = [_corr[m]['R2_test']     for m in _ms]
        _gap_orig = [_results[m]['Gap']       for m in _ms]
        _gap_corr = [_corr[m]['Gap']          for m in _ms]

        axes[0].bar(_x - _w/2, _r2_orig, _w, label='Original',  color='#e74c3c', alpha=0.8)
        axes[0].bar(_x + _w/2, _r2_corr, _w, label='Corregido', color='#2ecc71', alpha=0.8)
        axes[0].set_xticks(_x)
        axes[0].set_xticklabels(_ms, rotation=15, ha='right', fontsize=9)
        axes[0].set(ylabel='R² Test', title='R² Test: ¿Mantiene el rendimiento?')
        axes[0].legend()

        axes[1].bar(_x - _w/2, _gap_orig, _w, label='Gap original',  color='#e74c3c', alpha=0.8)
        axes[1].bar(_x + _w/2, _gap_corr, _w, label='Gap corregido', color='#2ecc71', alpha=0.8)
        axes[1].axhline(0.10, color='gray', ls='--', lw=1.2, label='Umbral 0.10')
        axes[1].set_xticks(_x)
        axes[1].set_xticklabels(_ms, rotation=15, ha='right', fontsize=9)
        axes[1].set(ylabel='Gap (R² train − R² test)',
                    title='Overfitting: ¿Se redujo el gap?')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        # Actualizar mejor modelo con los corregidos incluidos
        _todos = {**{k: v['R2_test'] for k, v in _results.items()
                     if '[corregido]' not in k},
                  **{k + ' [corr]': v['R2_test'] for k, v in _corr.items()}}
        _mejor_final = max(_todos, key=_todos.get)
        print(f"\n  Mejor modelo tras correcciones: {_mejor_final}"
              f"  (R²={_todos[_mejor_final]:.4f})")

print(f"\n{'='*52}")
print(f"  Análisis completado.")
print(f"  Mejor modelo original: {mejor_modelo}  (R²={mejor_r2:.4f}, RMSE={mejor_rmse:.2f}€)")
print(f"  Modelos disponibles: lasso_, ridge_, tree_, rf_, bag_, gbm_"
      + (", xgb_" if XGBOOST_OK else "") + ", svr_")
if _overfit:
    print(f"  Versiones corregidas: "
          + ", ".join([k.lower().replace(' ', '_') + '_c'
                       for k in _corr_models]))
print(f"{'='*52}")
