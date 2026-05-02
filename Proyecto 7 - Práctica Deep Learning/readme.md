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
Todos los modelos se evalúan sobre el **set de test** nunca usado durante el entrenamiento, con:
- Accuracy y F1-macro.
- Classification report detallado por clase.
- Tabla comparativa ordenada por F1-macro.
- Matriz de confusión del mejor modelo.

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