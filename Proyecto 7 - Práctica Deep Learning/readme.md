# Proyecto 7 — Clasificación Multimodal de Lesiones Cutáneas (HAM10000)

Proyecto final del módulo de **Deep Learning & Computer Vision**. El objetivo es predecir la condición médica de un paciente a partir de dos fuentes de información —una **imagen dermatoscópica** y unos **datos tabulares clínicos**— empleando redes neuronales profundas y estrategias de **fusión multimodal**.

---

## 📌 Problema

El dataset [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) contiene **10.015 imágenes de lesiones cutáneas** clasificadas en 7 categorías, junto con metadatos de cada paciente (edad, sexo, localización de la lesión).

Se plantea un problema de **clasificación multiclase** sobre las siguientes categorías:

| Código | Enfermedad |
|---|---|
| `akiec` | Queratosis actínicas / carcinoma de células escamosas |
| `bcc` | Carcinoma de células basales |
| `bkl` | Lesiones benignas de queratosis |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Nevus melanocítico |
| `vasc` | Lesiones vasculares |

El dataset está **fuertemente desbalanceado** (la clase `nv` representa ~67% de las muestras), por lo que la métrica principal de evaluación es el **F1-macro**, además del accuracy.

---

## 🎯 Hitos implementados

El enunciado exige la implementación de cuatro modelos con enfoques distintos:

1. **Modelo 1D (tabular)** — red densa que utiliza únicamente los metadatos clínicos.
2. **Modelo 2D (imágenes)** — redes convolucionales que utilizan únicamente la imagen.
3. **Estrategia late-fusion** — combinación aprendida de las **predicciones** de los modelos 1D y 2D.
4. **Estrategia early-fusion** — combinación aprendida de las **representaciones intermedias** (features) de los modelos 1D y 2D.

---

## 🛠️ Pipeline de trabajo

### Preprocesado
- Descarga automática del dataset desde Kaggle.
- Imputación de valores nulos en `age` con la mediana y normalización al rango `[0, 1]`.(lo pensaba dividir entre la edad máxima, pero finalmente normalicé dividiendo entre 100)
- Codificación *one-hot* de `sex` y `localization` 
- Reshape de las imágenes aplanadas a tensores `(N, 28, 28, 3)` 
- **Split estratificado por índices**, garantizando que datos tabulares, imágenes y etiquetas permanecen alineados en train / val / test (64% / 16% / 20%). 
	[La parte de hacer el índice me costó bastante verlo al principio y dado que soy "nuevo" en tema Python/Programación en generalm tuve que tirar un poco de IA para ver todo y entenderlo]

### Modelo 1D (tabular)
Red densa con API funcional: 4 capas ocultas de 32 y 16 neuronas, con activaciones ReLU/SELU y softmax final de 7 clases. Se nombra explícitamente la capa h3 - `tabular_features` para su reutilización en early-fusion.

### Modelo 2D (imágenes)
Se implementan y comparan **cuatro arquitecturas**. Como lo he preparado en KERAS, lo hice así para practicar lo máximo posible:
- **CNN propia**: incluí dos bloques Conv2D + BatchNorm + ReLU, Global Max Pooling y una capa densa. IMPORTANTE: aquí si divido entre 255. En las demás, no hace falta ya que tenemos el componente "preprocess_input".
- **VGG16** preentrenada en ImageNet, congelada como extractor de features.
- **ResNet50** preentrenada en ImageNet, congelada.
- **InceptionV3** preentrenada en ImageNet, congelada.

Las redes preentrenadas se usan en modo **feature extraction** (`trainable=False`, `training=False`), añadiendo encima un clasificador denso. Las imágenes 28×28 se redimensionan a la resolución esperada por cada red (224×224 o 299×299) mediante una capa `Resizing`.

- A cada capa densa final la voy nombrando como `vision_features_VGG`, `vision_features_REST` y `vision_features_Inc`.

### Estrategia late-fusion
Se construyen **4 modelos** de late-fusion, uno por cada red convolucional anterior. En cada uno:
- Se congelan los sub-modelos ya entrenados.
- Se concatenan sus **probabilidades softmax**.
- Un clasificador denso final aprende a combinar esas decisiones.

### Estrategia early-fusion
Se construyen **4 modelos** de early-fusion, uno por cada red convolucional. En cada uno:
- Se extraen las **representaciones intermedias** (features) de cada sub-modelo mediante `get_layer('tabular_features').output` y análogos en visión. (Por eso el nombrar las capas anteriormente)
- Se concatenan los vectores de features.
- Un clasificador más profundo (Dense 128 → 64 → 7) aprende directamente sobre la representación fusionada, permitiéndole capturar **interacciones cruzadas** entre modalidades que late-fusion no puede detectar.

### Evaluación
# Análisis de resultados — Práctica HAM10000 (segunda ejecución)
 
## 1. Tabla comparativa con ranking
 
Resultados ordenados por F1-macro descendente sobre el conjunto de test (2003 muestras):
 
| Posición | Modelo            | Accuracy | F1-macro | Categoría     |
|----------|-------------------|----------|----------|---------------|
| 🥇 1     | H4 - EF ResNet    | 0.803    | **0.611** | Early fusion  |
| 🥈 2     | H4 - EF CNN       | 0.805    | 0.608    | Early fusion  |
| 🥉 3     | H4 - EF VGG       | 0.798    | 0.557    | Early fusion  |
| 4        | H2 - ResNet50     | 0.764    | 0.530    | Visual solo   |
| 5        | H4 - EF Inception | 0.772    | 0.509    | Early fusion  |
| 6        | H2 - CNN manual   | 0.776    | 0.496    | Visual solo   |
| 7        | H2 - VGG16        | 0.764    | 0.493    | Visual solo   |
| 8        | H2 - InceptionV3  | 0.745    | 0.458    | Visual solo   |
| 9        | H3 - LF VGG       | 0.753    | 0.331    | Late fusion   |
| 10       | H3 - LF Inception | 0.722    | 0.244    | Late fusion   |
| 11       | H3 - LF ResNet    | 0.705    | 0.210    | Late fusion   |
| 12       | H1 - Tabular      | 0.698    | 0.198    | Tabular solo  |
| 13       | H3 - LF CNN       | 0.697    | 0.180    | Late fusion   |
 
---
 
## 2. El mejor modelo: H4 - EF ResNet
 
El **early-fusion combinando MLP tabular con ResNet50** se confirma como la mejor configuración, con un F1-macro de **0.611** y una accuracy de **0.803**. La diferencia con el segundo (EF CNN) es muy pequeña (0.003 puntos de F1), lo cual sugiere que el ranking de los dos primeros podría reordenarse en una nueva ejecución por simple variabilidad estocástica del entrenamiento. La conclusión sólida no es "ResNet es el mejor backbone", sino "**early fusion con un buen extractor visual** es la mejor estrategia".
 
**Recall por clase del modelo ganador (EF ResNet):**
 
| Clase | Recall | Lectura |
|-------|--------|---------|
| nv    | 0.93   | Excelente sin ser 1.00 — el modelo no colapsa |
| bcc   | 0.54   | Aceptable |
| vasc  | 0.61   | Bueno pese a 28 muestras |
| bkl   | 0.58   | Aceptable |
| mel   | **0.58** | Mejor que en la 1ª ejecución (0.53) |
| akiec | 0.42   | Mejorable |
| df    | 0.35   | Limitado por la escasez de datos (23) |
 
El detalle clínicamente más relevante: el recall de **melanoma sube de 0.53 a 0.58**. En un sistema de cribado dermatológico, este es el indicador que más importa, porque cada melanoma no detectado es un falso negativo de cáncer agresivo.
 
---
 
## 3. Patrones que se confirman entre ejecuciones
 
Los hechos que se mantienen estables entre la primera y la segunda ejecución son los hallazgos académicamente defendibles en una memoria.
 
### 3.1. Early fusion >> Late fusion (sistemáticamente)
 
| Backbone   | LF (F1-m) | EF (F1-m) | Diferencia |
|------------|-----------|-----------|------------|
| CNN manual | 0.180     | 0.608     | **+0.428** |
| VGG16      | 0.331     | 0.557     | +0.226     |
| ResNet50   | 0.210     | 0.611     | +0.401     |
| Inception  | 0.244     | 0.509     | +0.265     |
 
La superioridad de early fusion no es un artefacto de una ejecución concreta: es un fenómeno **reproducible y de gran magnitud**. Este es el resultado más sólido del experimento.
 
### 3.2. Late fusion está colapsando hacia la clase mayoritaria
 
Mira el `recall` de `nv` en los cuatro late-fusion: 1.00, 0.99, 1.00, 0.99. Es decir, predicen **siempre o casi siempre `nv`**. Las clases minoritarias (`akiec`, `df`, `vasc`) tienen recall 0.00 en varios casos. Esto confirma la interpretación de la ejecución anterior: el clasificador de late fusion (una única `Dense(softmax)` sobre 14 logits, con `lr=1e-4` durante 30 épocas) encuentra el óptimo trivial "predice nv siempre" antes de aprender nada útil.
 
Es importante destacar esto en la defensa: **late fusion no está infrautilizando la información, está colapsando**. La solución correcta sería `class_weight='balanced'` y/o un meta-clasificador con más capacidad. Sin esos cambios, la comparación EF vs LF parte sesgada.
 
### 3.3. El modelo tabular solo es muy débil pero no inútil
 
H1 da F1-macro 0.198, prácticamente lo mismo que predecir siempre `nv`. Sin embargo, cuando se combina con cualquier backbone visual en early fusion, **el resultado supera al backbone visual aislado**:
 
| Comparación              | Visual solo (F1-m) | + Tabular EF (F1-m) | Aporte |
|--------------------------|--------------------|---------------------|--------|
| CNN manual               | 0.496              | 0.608               | +0.112 |
| VGG16                    | 0.493              | 0.557               | +0.064 |
| ResNet50                 | 0.530              | 0.611               | +0.081 |
| InceptionV3              | 0.458              | 0.509               | +0.051 |
 
Los datos clínicos (edad, sexo, localización) son **señales débiles individualmente pero complementarias** a la imagen. Es la justificación empírica de plantear la práctica como un problema multimodal.
 
### 3.4. El ranking de backbones se mantiene
 
ResNet ≈ CNN manual (en EF) ≥ VGG > Inception. Inception sigue siendo el peor, consistente con la hipótesis de que las imágenes interpoladas desde 28×28 a 299×299 no contienen la información multiescala que las ramas paralelas de Inception fueron diseñadas para captar.
 
---
 
## 4. Diferencias entre la primera y la segunda ejecución
 
| Métrica                     | 1ª ejecución | 2ª ejecución | Δ      |
|-----------------------------|--------------|--------------|--------|
| Mejor F1-macro (EF ResNet)  | 0.621        | 0.611        | -0.010 |
| Mejor accuracy (EF ResNet)  | 0.810        | 0.803        | -0.007 |
| Recall melanoma (EF ResNet) | 0.53         | 0.58         | **+0.05** |
| Posición 1 vs 2             | Diferencia 0.008 | Diferencia 0.003 | — |
 
Las diferencias absolutas son pequeñas (<0.02 en F1-macro) y atribuibles a la **estocasticidad** del entrenamiento (inicialización de pesos, orden de batches, dropout estocástico). Esto te da munición para una observación importante en la defensa: **una única ejecución no es suficiente para distinguir entre modelos cuyas métricas difieren en menos de 0.02 puntos**. Lo correcto formalmente sería ejecutar varias semillas y reportar media ± desviación. En la práctica, lo que sí puedes afirmar con seguridad es que:
 
- Early fusion supera consistentemente a late fusion y a los modelos unimodales.
- El ganador siempre es algún EF, aunque el backbone concreto puede variar.
---
 
## 5. Limitaciones del estudio
 
Una memoria honesta debe incluir esta sección. Estas son las debilidades reales del experimento:
 
1. **Late fusion no se ha optimizado** (sin `class_weight`, meta-clasificador trivial). La comparación EF vs LF está sesgada a favor de EF.
2. **Una sola ejecución por modelo**. Diferencias menores de ~0.02 en F1-macro no son distinguibles del ruido.
3. **Imágenes a 28×28** redimensionadas al alza para los modelos preentrenados. Esto desperdicia la capacidad de redes diseñadas para imágenes nítidas (especialmente Inception).
4. **La imputación de la mediana se calcula sobre todo el dataset**, no solo sobre train. Es un *data leakage* menor pero existente.
5. **No se evalúa la incertidumbre**. Para una aplicación clínica real, el modelo debería poder decir "no estoy seguro" en lugar de forzar siempre una clase.
6. **Sin fine-tuning** de los backbones. Se utilizan en feature-extraction puro. Un fine-tuning suave con `lr` muy bajo en las últimas capas convolucionales podría aportar algo más.
---
 
## 6. Conclusión defendible para la memoria
 
> Los experimentos confirman tres conclusiones reproducibles entre ejecuciones. Primero, la estrategia de **early fusion supera consistentemente a late fusion** en todos los backbones probados, con diferencias en F1-macro entre +0.23 y +0.43 puntos. Segundo, los modelos de **late fusion implementados con un meta-clasificador minimalista colapsan hacia la clase mayoritaria**, lo que deja su comparación con early fusion sesgada y sugiere como mejora futura el uso de `class_weight='balanced'` y de un meta-clasificador con mayor capacidad. Tercero, los **datos tabulares aportan una señal complementaria** a la imagen: aunque por sí solos son insuficientes (F1-macro 0.20), su combinación en early fusion mejora cualquier modelo unimodal visual entre 5 y 11 puntos de F1-macro. El mejor modelo obtenido es el **early fusion con ResNet50** (F1-macro 0.611, accuracy 0.803), aunque su diferencia con el segundo (EF CNN, 0.608) está dentro del margen de variabilidad estocástica.
---

## 📚 Conceptos clave trabajados

- Diseño multimodal con Keras Functional API.
- Transfer learning y feature extraction con modelos pre-entrenados (VGG, ResNet, Inception).
- Alineación de modalidades mediante split estratificado por índices.
- Manejo de flujos de datos heterogéneos (tabular + imagen) desde Kaggle a TensorFlow.
- Extracción de capas intermedias mediante `Model(inputs=..., outputs=get_layer(name).output)`.
- Diferencia conceptual y práctica entre **feature extraction**, **fine-tuning**, **late fusion** y **early fusion**.

---

## 🧰 Stack tecnológico

- **Python 3** · **TensorFlow / Keras**
- **NumPy**, **pandas**, **scikit-learn**
- **matplotlib**, **seaborn**
- **Google Colab** como entorno de ejecución (GPU/TPU)
- **Kaggle API** para la descarga del dataset

---

## 📁 Estructura del proyecto