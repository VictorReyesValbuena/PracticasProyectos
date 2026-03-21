# 🏠 Airbnb Price Prediction — Práctica Machine Learning

Práctica de Machine Learning sobre un dataset real de Airbnb con el objetivo de predecir el precio de los alojamientos. El proyecto se ha resuelto de **dos formas distintas**: una aproximación completamente manual y una segunda, a modo prueba y por estudiarlo, implementada con IA (Claude), con el fin de comparar ambos enfoques metodológicos (básicamente para probar la resolucion del problema con IA). En esa versión con IA, le he ido dando instrucciones a Claude para que fuera desarrolando tanto el eda_profiler como el documento modelos_regresion.py.

EL IMPORTANTE ES EL MANUAL.

---

## 📁 Estructura del proyecto

```
├── **SIN IA/ Y LA QUE ES PARA CORREGIR!!**
│   ├── **OK_EDA_y_limpieza_datos_manual.ipynb**     # Análisis exploratorio y limpieza manual paso a paso
│   └── **OK_ResolucionPractica.ipynb**              # Modelado completo: preprocesamiento + 7 algoritmos
│
├── CON IA/
│   ├── eda_profiler.py                       # Script de EDA automatizado generado con Claude
│   ├── Exploracion_de_datos.ipynb            # Notebook para ejecutar eda_profiler.py
│   ├── EDA_Airbnb.html                       # Informe HTML interactivo resultado del EDA automático
│   ├── modelos_regresion.py                  # Script de modelado automatizado generado con Claude
│   └── ResolucionPracticaIA.ipynb            # Ejecución de modelos_regresion.py sobre los datos procesados
│
└── README.md
```

---

## ⚙️ Requisitos

Python 3.8+ y las siguientes librerías:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

| Librería | Uso |
|---|---|
| `numpy` / `pandas` | Manipulación de datos |
| `matplotlib` / `seaborn` | Visualización |
| `scikit-learn` | Preprocesamiento, modelos y métricas |
| `xgboost` | Gradient Boosting optimizado |

---

## 🔍 Descripción de la resolución

### 1. División train / test
Lo primero antes de cualquier transformación: split 80/20 con `train_test_split` para evitar *data leakage*. Todas las transformaciones posteriores se aprenden **solo sobre train** y se replican en test (`KNNImputer`, `TargetEncoder` y `StandardScaler`)

### 2. Selección de variables
Del dataset original se seleccionaron 15 variables relevantes para la predicción del precio:
`Neighbourhood Cleansed`, `City`, `Country`, `Property Type`, `Room Type`, `Accommodates`, `Bathrooms`, `Security Deposit`, `Cleaning Fee`, `Guests Included`, `Extra People`, `Amenities`, `Availability 365`, `Review Scores Rating`.

### 3. Análisis exploratorio (EDA)
- Revisión de tipos de datos, nulos y distribuciones
- Detección y tratamiento de outliers en `Cleaning Fee`, `Extra People`, `Security Deposit` y `Bathrooms`
- Matriz de correlación para identificar multicolinealidad (se eliminaron columnas de `Review` correlacionadas entre sí y `Beds` por su correlación con `Accommodates`)
- Feature selection con **F-Test** y **Mutual Information**: `Cleaning Fee` y `Security Deposit` son las variables con mayor poder predictivo

### 4. Preprocesamiento
- **Imputación de nulos**: `KNNImputer` sobre variables numéricas; valor `"Desconocido"` para `City` y `Country`
- **Codificación categórica**: `TargetEncoder` sobre variables `object`
- **Escalado**: `StandardScaler` con `fit` solo en train (aplicado a modelos que lo requieren)

### 5. Modelado

Se han entrenado y comparado **7 algoritmos** con optimización de hiperparámetros mediante `GridSearchCV` y validación cruzada:

| Modelo | Escalado | Hiperparámetros principales |
|---|---|---|
| **Lasso** | ✅ Sí | `alpha` (GridSearch) |

Defino el alpha vector, los paso por GridSearch, obtengo los parámetro óptimos y hago fit con los datos escalados. El R2 de train y test es bastante parecido y podemos ver que no existe Overfitting.

| Modelo | Escalado | Hiperparámetros principales |
| **Ridge** | ✅ Sí | `alpha` (rango 10⁻² a 10⁴) |

Defino alpha con un rango alto (el que copié) y empiezo a trabajar sobre él. Veo que el mejor rango para observar el mejor alpha está entre 10-1 y 10-2. Concretamente, me fija el 7.054. Hago el modelo con el valor optimo y  pinto los R2. No es un R2 muy bueno, pero si es cierto que podemos observar que no hay Overfitting.

| Modelo | Escalado | Hiperparámetros principales |
| **Decision Tree** | ❌ No | `max_depth`, `min_samples_leaf` |

Utilizo datos sin escalar. Sigo el mismo proceso, defino max_depth, le paso por validación cruzada para sacar el mejor parámetro, hago fit sobre ese valor y saco los R2. En los primeros intentos, veo que tiene un Overfitting considerable. Lo que hago es introducir el dato min_sample_leaf, calcular su optimo y volver ha hacer fit para sacar los R2. Aprovecho y saco la importancia de las variables y vemos como "Property Type", "Guests Included" y "Extra People", son las "peores" de hecho la primera de estas ni se ha tenido en cuenta, es un punto negativo de este algoritmo que tiende a no contar con las variables "peores". "Cleaning Fee es la mejor"

| Modelo | Escalado | Hiperparámetros principales |
| **Random Forest** | ❌ No | `n_estimators`, `max_depth` |

Utilizo datos sin escalar. Antes de realizar la validación cruzada, defino los parametros de mi param_grid, obtengo los mejores parametros y hago la prediccion. Además, grafico la importancia de las Features donde se puede observar que, utilizar todas ellas (al contrario que el arbol de decisión individual que dejaba una sin utilizar), y la que más importancia tienen a la hora de explicar el precio es Cleaning Fee.

| Modelo | Escalado | Hiperparámetros principales |
| **Bagging Regressor** | ❌ No | `n_estimators` |

Utilizo datos sin escalar. Hago lo mismo, defino mi estimador como DecisionTreeRegressor, saco los parametros optimos y ejecuto el algoritmo con ellos. Obtengo las variables más importantes y la ganadora vuelve a ser Cleaning Fee.

| Modelo | Escalado | Hiperparámetros principales |
| **Gradient Boosting (GBM)** | ❌ No | `n_estimators`, `learning_rate` |

Defino las NIterations y el learningRate e implemento validación cruzada para encontrar los valores óptimos. Una vez hecho eso, ejecuto el algoritmo, imprimo los r2 de train y test y grafico las variables por importancia. Vuelve a ganar Cleaning fee y se puede ver como también usa todas las variables.

| Modelo | Escalado | Hiperparámetros principales |
| **SVR** | ✅ Sí | `C`, `gamma` (kernel RBF) |

---

## 📝Comentario alumno:

Antes de pasar a las conclusiones, quiero comentar algunas cosas:

 -- En cuanto a la metodología a seguir para el desarrollo de la práctica, me he centrado en ejecutar "los pasos correctos" en cada momento. Probablemente no sea la práctica más top a nivel técnico pero he intentado ir paso a paso seguiendo el orden que nos has ido diciendo en las clases sobre cómo se debe implementar el proceso de manera 100% correcta. He quitado algunas columnas de primeras que no tenían ningún valor, he hecho la división train y test, he hecho imputaciones, codificación de variables categoricas (en ambas solo haciendo el fit en train), limpieza general, graficación de variables para el estudio de Outliers, eliminación de los mismos y escalado. He intentado ser conservador en ese sentido, por eso decía que quizás no es la mejor práctica a nivel técnico ya que me he centrado en el proceso.

 -- Posteriormente, he implementado cada uno de los algoritmos vistos en clase intentado entender cada uno de los parámetros. Basicamente, he ido copiando cada uno de los código para cada algoritmo (no al 100% pero si la estructura general) y he ido modificando aspectos importantes para que el modelo tuviera mejor resultado, (alpha_vector, max_depth, min_samples_leaf, max_features, n_estimator y learning_rate entre otros). También he utilizando en los métodos de árboles, la graficación de la importancia de cada variable (incluso llegué a eliminar una tras ejecutar por primera vez los algoritmos). En esta parte, básicamente he intentado repasar y profundizar en el funcionamiento de cada uno de los algoritmos.

 -- Al final del todo, le pedí a Claude que me hiciera una tabla comparativa con las métricas **R²** y **RMSE** sobre train y test, incluyendo el gap de overfitting (ΔR²).

## 📊 Conclusiones

### Modelos lineales (Lasso / Ridge):

Tienen un rendimiento modesto, por lo que podemos decir que la relación entre las features y la variable objetivo no se puede ajustar bien a una relación lineal. A cambio de esto, podemos ver que no existe prácticamente Overfitting.

### Modelos basados en árboles:

GBM es el modelo más recomendable: mejor equilibrio entre rendimiento (R² test = 0.843) y generalización (ΔR² = 0.047). En la práctica sería la primera elección.
Luego iría Random Forest y Decision Tree ya que Bagging tiene claro Overfitting (probablemente podría retocar algo el modelo o los parámetros)

### Modelo SVM:

Presenta overfitting sin embargo es por los parámetros que le he definido (con los "habituales" llevaba ejecutándose 2h...)

*Práctica realizada en el marco del Bootcamp de Machine Learning — KeepCoding*