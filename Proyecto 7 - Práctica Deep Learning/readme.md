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

---

## 🎯 Cuestiones a resolver:

El enunciado exige la implementación de cuatro modelos con enfoques distintos:

1. **Modelo 1D (tabular)** — red densa que utiliza únicamente los metadatos clínicos.
2. **Modelo 2D (imágenes)** — redes convolucionales que utilizan únicamente la imagen.
3. **Estrategia late-fusion** — combinación aprendida de las **predicciones** de los modelos 1D y 2D.
4. **Estrategia early-fusion** — combinación aprendida de las **representaciones intermedias** (features) de los modelos 1D y 2D.

---

## 🛠️ Desarrollo del trabajo:

### Preprocesado
- Descarga automática del dataset desde Kaggle.
- Imputación de valores nulos en `age` con la mediana y normalización al rango `[0, 1]`.(lo pensaba dividir entre la edad máxima, pero finalmente normalicé dividiendo entre 100)
- Codificación *one-hot* de `sex` y `localization` 
- Reshape de las imágenes aplanadas a tensores `(N, 28, 28, 3)` 
- **Split estratificado por índices**, garantizando que datos tabulares, imágenes y etiquetas permanecen alineados en train / val / test (64% / 16% / 20%). 
	[La parte de hacer el índice me costó bastante verlo al principio y dado que soy "nuevo" en tema Python/Programación en general, tuve que tirar un poco de IA para ver todo y entenderlo]

### Modelo 1D (tabular)
Red densa con API funcional: 4 capas ocultas de 128 neuronas, con activaciones ReLU/SELU y softmax final de 7 clases. Se nombra explícitamente la capa h3 - `tabular_features` para su reutilización en early-fusion.

### Modelo 2D (imágenes)
Se implementan y comparan **cuatro arquitecturas**. Como lo he preparado en KERAS, lo hice así para practicar lo máximo posible:
- **CNN propia**: incluí dos bloques Conv2D + BatchNorm + ReLU, Global Max Pooling y una capa densa. IMPORTANTE: aquí si divido entre 255. En las demás, no hace falta ya que tenemos el componente "preprocess_input".
- **VGG16** 
- **ResNet50** 
- **InceptionV3** 

Las redes preentrenadas se usan en modo **feature extraction**, añadiendo encima un clasificador denso. Las imágenes 28×28 se redimensionan a la resolución esperada por cada red (224×224 o 299×299) mediante una capa `Resizing`.

- A cada capa densa final la voy nombrando como `vision_features_VGG`, `vision_features_REST` y `vision_features_Inc` para su posterior utilizanción en las fusiones.

### Estrategia late-fusion
Se construyen **4 modelos** de late-fusion, uno por cada red convolucional anterior. En cada uno:
- Se congelan los sub-modelos ya entrenados.
- Se concatenan sus **probabilidades**.
- Se añade un clasificador denso final.

### Estrategia early-fusion
Se construyen **4 modelos** de early-fusion, uno por cada red convolucional. En cada uno:
- Se extraen las **representaciones intermedias** (features) de cada sub-modelo mediante `get_layer('tabular_features').output` y análogos en visión. (Por eso el nombrar las capas anteriormente)
- Se concatenan los vectores de features.
- Un clasificador más profundo (Dense 128 → 64 → 7) aprende directamente sobre la representación fusionada, permitiéndole capturar **interacciones cruzadas** entre modalidades que late-fusion no puede detectar.

### Evaluación

## 1. Tabla comparativa 
[Para esto, le pedi a Claude que me hicera un cuadro comparativo de los resultados de todos los modelos entrenados, supuse que por encima de 0.05 existía Overfitting]
 
 | # | Modelo | accuracy | val_accuracy | gap | Diagnóstico |
|:-:|--------|:--------:|:------------:|:-----:|:-----------:|
| 🥇 1 | **H4 - EF CNN**       | 0.8540 | **0.8085** | +0.0455 | ✅ Ajuste correcto |
| 🥈 2 | **H4 - EF VGG**       | 0.8048 | 0.7954     | +0.0094 | ✅ Ajuste correcto |
| 🥉 3 | **H4 - EF ResNet**    | 0.8399 | 0.7954     | +0.0445 | ✅ Ajuste correcto |
| 4 | H4 - EF Inception     | 0.7942 | 0.7723     | +0.0219 | ✅ Ajuste correcto |
| 5 | H2 - CNN manual       | 0.7984 | 0.7717     | +0.0267 | ✅ Ajuste correcto |
| 6 | H2 - ResNet50         | 0.7919 | 0.7642     | +0.0277 | ✅ Ajuste correcto |
| 7 | H3 - LF VGG           | 0.7514 | 0.7611     | −0.0097 | ⚠️ **Underfitting** |
| 8 | H2 - VGG16            | 0.7681 | 0.7592     | +0.0089 | ✅ Ajuste correcto |
| 9 | H2 - InceptionV3      | 0.7644 | 0.7386     | +0.0258 | ✅ Ajuste correcto |
| 10 | H3 - LF Inception    | 0.7327 | 0.7243     | +0.0084 | ✅ Ajuste correcto |
| 11 | H1 - Tabular         | 0.7010 | 0.7137     | −0.0127 | ⚠️ **Underfitting** |
| 12 | H3 - LF ResNet       | 0.7165 | 0.7118     | +0.0047 | ✅ Ajuste correcto |
| 13 | H3 - LF CNN          | 0.7074 | 0.7024     | +0.0050 | ✅ Ajuste correcto |

---

## 🎯 COMENTARIO ALUMNO

	Hola Diego, 

	Como comentario final a todo el trabajo desarrollado, decir varias cosas:

	- He desarrollado la práctica en Keras ya que, es más fácil a nivel "código" pero sobre todo he querido detenerme en cada concepto para llevarme de este módulo todos los punto clave bien aprendidos y tenerlo todo claro. Sin embargo, lo estoy empezando a hacer con Pytorch paralelamente siguiendo tu consejo de las clases. 
	- En cuanto al propio archivo, he intentado optimizar al máximo los modelos. Me surgieron problemas de overfitting en muchos de ellos, he "jugado" con los hiperparámetros (Dropout, learning rate, batch), con las capas de las redes añadiendo capas de BatchNormalization, sustituyendo la capa flatten por GlobalAveragePooling, etc.
	- Basicamente, me he centrado en presentar unos modelos consistentes y en entender el funcionamiento de late y early fusión además de las formas de "conectar" ambas partes (la tabular y las de imágenes). Se que no es un trabajo "espectacular" pero ya te digo que en lo que me he centrado es en tener los conceptos claros.
	- El readme, lo he hecho al 50% con IA para seguir el formato de todo el repositorio y para que quedara más vistoso.


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