# Sistema de Cierre Fiscal Inteligente

> Automatización de informes de cierre contable-fiscal mediante Prompt Engineering e integración con la API de Google Gemini.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gemini API](https://img.shields.io/badge/Google-Gemini%20API-4285F4.svg)](https://ai.google.dev/)
[![pandas](https://img.shields.io/badge/pandas-data--processing-150458.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/status-functional-success.svg)]()

---

## 📑 Tabla de contenidos

- [Sobre el proyecto](#-sobre-el-proyecto)
- [Caso de uso real](#-caso-de-uso-real)
- [Arquitectura y flujo](#-arquitectura-y-flujo)
- [Características](#-características)
- [Stack técnico](#-stack-técnico)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Requisitos previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Configuración de la API](#-configuración-de-la-api)
- [Uso](#-uso)
- [Formato esperado del CSV](#-formato-esperado-del-csv)
- [Metodología CO-STAR](#-metodología-co-star)
- [Iteraciones del proyecto](#-iteraciones-del-proyecto)
- [Ejemplo de salida](#-ejemplo-de-salida)
- [Roadmap](#-roadmap)
- [Autor](#-autor)

---

## 📌 Sobre el proyecto

`Informe_mail_Contabilidad.py` es un script en Python que **automatiza la redacción de los mails informativos de cierre provisional del Impuesto sobre Sociedades** que un asesor contable-fiscal envía a sus clientes a finales de cada ejercicio.

A partir de un CSV exportado del ERP **A3Eco** con la cuenta de Pérdidas y Ganancias del cliente (ejercicio en curso vs. ejercicio anterior), el sistema:

1. Limpia y agrega los datos por código de cuenta del Plan General Contable (PGC).
2. Construye un resumen estructurado con las partidas relevantes para el cierre.
3. Lo envía a **Google Gemini** mediante llamada a la API con un prompt diseñado bajo el framework **CO-STAR**.
4. Recibe y guarda un informe-mail listo para enviar al cliente, con análisis de variaciones, alertas de riesgo fiscal y conclusión narrativa.

El proyecto nace como **práctica académica de Usos de la IA** y resuelve un problema profesional real: lo que normalmente toma una mañana o dos por cliente queda reducido a minutos, dejando solo la matización personalizada por cliente al asesor.

---

## 🎯 Caso de uso real

> En la asesoría se manda un mail "genérico e informativo" a finales de diciembre o principios de enero, con el resultado estimado del ejercicio, la cifra orientativa a pagar de Impuesto sobre Sociedades y una comparativa con el año anterior, intentando explicar la razón de pagar más o menos en dicho impuesto.

El mail sigue siempre la misma plantilla y debe cubrir un set fijo de partidas (ventas, aprovisionamientos, otros gastos de explotación, amortizaciones, deterioros, gastos financieros, resultado e impuesto). El script reproduce ese flujo y permite **cribar clientes**:

- **Clientes con actividad estable** → mail genérico provisional.
- **Clientes con variaciones relevantes** → se identifican rápidamente para agendar reunión y cierre detallado.

---

## 🏗 Arquitectura y flujo

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  CSV de A3Eco    │───▶│  Limpieza con    │───▶│  Diccionario     │
│  (P&G 2025/24)   │    │  pandas          │    │  resumen         │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                          │
                                                          ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Informe_Final   │◀───│  Respuesta       │◀───│  Prompt CO-STAR  │
│  _Gemini.txt     │    │  Gemini API      │    │  + datos         │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

---

## ✨ Características

- ✅ **Lectura robusta de CSV** con separador `;` y decimales `,` (formato A3Eco / locale español).
- ✅ **Limpieza automática** de números (puntos de miles, comas decimales, separadores residuales).
- ✅ **Agrupación por código raíz de cuenta** (ej. todas las subcuentas que empiecen por `6230`).
- ✅ **Detección automática del modelo Gemini disponible** según la API Key del usuario, priorizando la familia *flash* por gratuidad y velocidad.
- ✅ **Prompt estructurado por secciones obligatorias** que garantiza un informe completo y homogéneo.
- ✅ **Alertas de riesgo fiscal incorporadas**: falsos autónomos (alta `6230` con `640/642 = 0`), variaciones relevantes en arrendamientos, suministros, amortizaciones y gastos financieros.
- ✅ **Tipo impositivo dinámico** (23% reducido para facturación < 1 M€ / 25% general).
- ✅ **Salida en `.txt`** lista para copiar y pegar en el cliente de correo.

---

## 🛠 Stack técnico

| Componente | Uso |
|------------|-----|
| **Python 3.10+** | Lenguaje base |
| **pandas** | Lectura y limpieza del CSV |
| **google-generativeai** | Cliente oficial de la API de Gemini |
| **Gemini 2.5 Flash** (auto-detectado) | Modelo LLM para la redacción |
| **Framework CO-STAR** | Estructura del prompt |

---

## 📁 Estructura del repositorio

```
.
├── Informe_mail_Contabilidad.py     # Script principal
├── datos_cliente_2.csv              # CSV de ejemplo (formato A3Eco)
├── modelo.csv                       # Plantilla vacía con la estructura de cuentas
├── Informe_Final_Gemini.txt         # Salida generada por el script
├── docs/
│   └── Practica_Usos_de_la_IA.pdf   # Dossier académico del proyecto
└── README.md
```

---

## 📋 Requisitos previos

- Python **3.10** o superior
- Una **API Key de Google Gemini** (gratuita) — se obtiene en [Google AI Studio]
- CSV de Pérdidas y Ganancias exportado desde el ERP en el formato esperado (ver más abajo)

---

## ⚙️ Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/VictorReyesValbuena/<nombre-del-repo>.git
cd <nombre-del-repo>

# 2. (Recomendado) Crear y activar un entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Instalar dependencias
pip install pandas google-generativeai
```

---

## 🔑 Configuración de la API

> ⚠️ **Nunca subas tu API Key al repositorio.** El campo está deliberadamente vacío y debe rellenarse en local o mediante una variable de entorno.

Abre `Informe_mail_Contabilidad.py` y rellena la constante:

```python
# =================================================================
# CONFIGURACIÓN DE SEGURIDAD
# =================================================================
API_KEY = "TU_API_KEY_AQUÍ"
genai.configure(api_key=API_KEY)
```
---

## 🚀 Uso

Desde la terminal, en el directorio donde se encuentran el script y el CSV:

```bash
python Informe_mail_Contabilidad.py
```

El programa pedirá interactivamente el nombre del CSV a analizar:

```
--- SISTEMA DE CIERRE FISCAL INTELIGENTE v7.1 ---
Nombre del CSV (ej. datos.cliente.csv): datos_cliente_2.csv
Buscando modelos disponibles para tu clave...
Modelo detectado y listo: models/gemini-2.5-flash
Redactando informe inteligente...

[ÉXITO] Informe generado en 'Informe_Final_Gemini.txt'
```

El fichero `Informe_Final_Gemini.txt` aparecerá en el mismo directorio.

---

## 📊 Formato esperado del CSV

El CSV debe seguir la estructura estándar de la cuenta de Pérdidas y Ganancias del PGC tal y como la exporta **A3Eco**:

- **Separador**: `;` (punto y coma)
- **Decimales**: `,` (coma)
- **Miles**: `.` (punto)
- **Encoding**: UTF-8
- **Columna identificadora**: `Cuenta de Pérdidas y Ganancias`
- **Columnas de datos**: `AÑO 2025` y `AÑO 2024`

Ejemplo (extracto):

```csv
Cuenta de Pérdidas y Ganancias;AÑO 2025;AÑO 2024
1. Importe neto de la cifra de negocios;679.861,10;540.678,43
  7000 Ventas mercaderías;...;...
4. Aprovisionamientos;...;...
7. Otros gastos de explotación;...;...
  6230 Servicios de profesionales independientes;196.833,26;155.050,98
8. Amortización de inmovilizado;27.854,11;23.308,81
...
C) RESULTADO ANTES DE IMPUESTOS;145.125,55;67.228,21
Impuesto sobre sociedades;33.378,88;15.462,49
D) RESULTADO DEL EJERCICIO;111.746,67;51.765,72
```

El script extrae automáticamente las siguientes secciones del resumen:

| Clave | Patrón de búsqueda |
|-------|-------------------|
| Ventas | `1. Importe neto` |
| Gastos | `7. Otros gastos` |
| Profesionales (6230) | `6230` |
| Salarios | `640\|642` |
| Amortizaciones | `8. Amortización de inmovilizado` |
| Arrendamientos | `621` |
| Suministros | `628` |
| Gastos financieros | `15. Gastos financieros` |
| Impuesto | `Impuesto sobre sociedades` |
| Resultado | `D) RESULTADO DEL` |

---

## 🧠 Metodología CO-STAR

El prompt enviado a Gemini está construido bajo el framework **CO-STAR** (Sheila Teo) — Context, Objective, Style, Tone, Audience, Response — y exige una estructura de salida con secciones marcadas entre corchetes que el modelo debe respetar siempre, haya o no incidencias en cada una:

```
[RESULTADO E IMPUESTO]
[ALERTA FALSOS AUTÓNOMOS]
[ARRENDAMIENTOS]
[SUMINISTROS]
[AMORTIZACIONES]
[GASTOS FINANCIEROS]
[CONCLUSIÓN DEL CIERRE CONTABLE]
[FINAL DEL MAIL]
```

Esta restricción estructural fue clave para conseguir informes homogéneos entre clientes. La temperatura del modelo se fija en `0.4` para equilibrar precisión técnica y naturalidad redactiva.

---

## 🔄 Iteraciones del proyecto

El proyecto evolucionó en cuatro versiones, documentadas en detalle en el [dossier académico](docs/Practica_Usos_de_la_IA.pdf):

| Versión | Plataforma | Aprendizaje clave |
|---------|------------|-------------------|
| **v1** | Claude (con Skill personalizada) | Un objetivo demasiado amplio sin ejemplo de input real produce resultados genéricos. Skill descartada. |
| **v1 CO-STAR** | Gemini (web) | Funciona bien con CO-STAR, pero el formato del CSV de A3Eco causó iteraciones extra de prompt. |
| **v2 CO-STAR** | NotebookLM + Gems + API Gemini | Generación del propio script vía prompts. Lección: ser conciso da mejores prompts que sobreexplicar. |
| **v3** | Refinamiento dentro del `.py` | Estructura por secciones obligatorias `[BLOQUE]`. Resultado homogéneo y aprovechable como borrador real. |

---

## 📤 Ejemplo de salida

Extracto del `Informe_Final_Gemini.txt` generado a partir de `datos_cliente_2.csv`:

```
Estimado cliente,

[RESULTADO E IMPUESTO]
Las Ventas han experimentado un notable incremento, pasando de 540.678,43 €
en 2024 a 679.861,10 € en 2025, lo que representa un aumento del 25,74%.
Los Gastos también han aumentado de 243.856,74 € a 293.751,27 €, un 20,46%.
A pesar del aumento de gastos, el incremento de las ventas ha sido
proporcionalmente mayor, resultando en un Resultado significativamente
superior, pasando de 67.228,21 € en 2024 a 145.125,55 € en 2025.
[...]

[ALERTA FALSOS AUTÓNOMOS]
La cuenta de "Profesionales (6230)" muestra un importe elevado de
196.833,26 € en 2025 (frente a 155.050,98 € en 2024), mientras que la
cuenta de "Salarios" se mantiene en 0 € en ambos ejercicios. Esta
situación podría ser objeto de revisión por parte de la Inspección de
Trabajo y Seguridad Social, que podría recalificar a estos profesionales
como "falsos autónomos" [...]

[GASTOS FINANCIEROS]
Los gastos financieros han aumentado significativamente, pasando de
1.503,17 € en 2024 a 3.856,76 € en 2025. Este incremento de más del 150%
puede ser una señal de una mayor financiación ajena [...]

[CONCLUSIÓN DEL CIERRE CONTABLE]
El principal motivo del aumento del impuesto sobre sociedades es el
significativo incremento del resultado antes de impuestos obtenido en
el ejercicio 2025.

Para cualquier consulta, quedamos a su total disposición [...]
Un saludo y gracias.
```

---

## 🛣 Roadmap

Mejoras previstas para próximas versiones:

- [ ] **Procesamiento por lotes**: ejecutar el script sobre todos los CSV de una carpeta y generar un informe por cliente en una sola pasada.
- [ ] **Gestión segura de la API Key** mediante `python-dotenv` y `.env` (con `.gitignore`).
- [ ] **Salida en múltiples formatos** (`.docx`, `.html`, `.md`) además de `.txt`.
- [ ] **Plantilla de tabla resumen** con las partidas clave al inicio del mail.
- [ ] **Logging estructurado** (en lugar de `print`) y manejo de errores granular.
- [ ] **Tests unitarios** sobre las funciones de limpieza y agregación.
- [ ] **Configuración externa** (YAML/JSON) para los patrones de cuentas y secciones del prompt.
- [ ] **Integración opcional con n8n** para disparar el script al recibir un CSV por correo.

---

## 👤 Autor

**Víctor Reyes Valbuena**
Asesor contable-fiscal · Estudiante de Máster en Asesoría Fiscal · Full Stack AI

[![GitHub](https://img.shields.io/badge/GitHub-VictorReyesValbuena-181717?logo=github)](https://github.com/VictorReyesValbuena)

---

## 📚 Contexto académico

Este repositorio forma parte de la práctica final de la asignatura **Usos de la Inteligencia Artificial**, dentro de un Bootcamp Full Stack AI. La memoria completa del proceso, con cada iteración del prompt, capturas y reflexiones, está disponible en [`docs/Practica_Usos_de_la_IA.pdf`](docs/Practica_Usos_de_la_IA.pdf).
