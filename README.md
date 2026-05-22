# 🫀 Heart Disease EDA — Dashboard Interactivo

> **Sistema de Soporte a la Decisión Clínica para la Detección Temprana de Patologías Cardiovasculares**  
> Análisis Exploratorio de Datos (EDA) + Constructor Interactivo + Predicción por ML sobre el dataset Cleveland Heart Disease (UCI)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18-3F4F75?logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3?logo=bootstrap&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Cleveland-orange)
![Observations](https://img.shields.io/badge/Observaciones-303-green)
![Variables](https://img.shields.io/badge/Variables-14-green)

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Dataset](#-dataset)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Visualizaciones (21 gráficas)](#-visualizaciones-21-gráficas)
- [Constructor Interactivo de Gráficas](#-constructor-interactivo-de-gráficas)
- [Predicción con Machine Learning](#-predicción-con-machine-learning)
  - [Algoritmo Seleccionado](#algoritmo-seleccionado-regresión-logística)
  - [Comparación de Algoritmos](#comparación-de-algoritmos)
  - [API de Predicción](#api-de-predicción)
- [Instalación y Ejecución](#-instalación-y-ejecución)
  - [Ejecución Local (Python)](#opción-1-ejecución-local-python)
  - [Ejecución con Docker](#opción-2-ejecución-con-docker-recomendado)
- [API Endpoints](#-api-endpoints)
- [Tecnologías](#-tecnologías)
- [Variables del Dataset](#-variables-del-dataset)
- [Equipo](#-equipo)

---

## 📌 Descripción del Proyecto

Este proyecto es un **dashboard web interactivo** de Análisis Exploratorio de Datos (EDA) construido con **Flask + Plotly + scikit-learn**, completamente Dockerizado. Analiza el dataset *Cleveland Heart Disease* del repositorio UCI con el objetivo de identificar patrones clínicos y demográficos asociados a la presencia de enfermedad cardíaca.

El dashboard incluye **21 gráficas EDA interactivas** organizadas en **8 secciones temáticas**, más un **constructor personalizado de gráficas** y un **módulo de predicción por aprendizaje automático**, con diseño médico profesional utilizando Bootstrap 5.

**Objetivo clínico:** Facilitar la detección temprana de patologías cardiovasculares a través del análisis visual de variables como presión arterial, colesterol, frecuencia cardíaca máxima, tipo de dolor torácico y resultados electrocardiográficos, con capacidad de predicción individual para nuevos pacientes.

---

## 📊 Dataset

| Propiedad            | Detalle                                              |
|----------------------|------------------------------------------------------|
| **Nombre**           | Cleveland Heart Disease Dataset                      |
| **Fuente**           | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |
| **Observaciones**    | 303 pacientes                                        |
| **Variables**        | 14 (8 nominales + 5 numéricas + 1 objetivo)          |
| **Valores faltantes**| Ninguno                                              |
| **Tarea**            | Clasificación binaria (0: sin enfermedad / 1: enfermedad) |

### Variables del Dataset

| Variable    | Tipo     | Descripción                                                                 | Valores                                                        |
|-------------|----------|-----------------------------------------------------------------------------|----------------------------------------------------------------|
| `age`       | Numérico | Edad del paciente (años)                                                    | 29 – 77                                                        |
| `sex`       | Nominal  | Sexo biológico                                                              | 0 = Femenino, 1 = Masculino                                    |
| `cp`        | Nominal  | Tipo de dolor torácico                                                      | 0 = Angina Típica, 1 = Angina Atípica, 2 = Dolor No Anginoso, 3 = Asintomático |
| `trestbps`  | Numérico | Presión arterial en reposo (mmHg)                                           | —                                                              |
| `chol`      | Numérico | Colesterol sérico (mg/dl)                                                   | —                                                              |
| `fbs`       | Nominal  | Glucosa en ayunas > 120 mg/dl                                               | 0 = Normal (≤120), 1 = Alto (>120)                             |
| `restecg`   | Nominal  | Resultado del electrocardiograma en reposo                                  | 0 = Normal, 1 = Anomalía ST-T, 2 = HVI (Estes)                |
| `thalach`   | Numérico | Frecuencia cardíaca máxima alcanzada (bpm)                                  | —                                                              |
| `exang`     | Nominal  | Angina inducida por ejercicio                                               | 0 = No, 1 = Sí                                                 |
| `oldpeak`   | Numérico | Depresión del segmento ST inducida por ejercicio respecto al reposo         | —                                                              |
| `slope`     | Nominal  | Pendiente del segmento ST en el pico del ejercicio                          | 0 = Ascendente, 1 = Plano, 2 = Descendente                     |
| `ca`        | Nominal  | Número de vasos principales coloreados por fluoroscopía                     | 0 – 3                                                          |
| `thal`      | Nominal  | Resultado del examen de talasemia (flujo sanguíneo)                         | 1 = Flujo Normal, 2 = Defecto Fijo, 3 = Defecto Reversible    |
| `target`    | Nominal  | **Variable objetivo:** presencia de enfermedad cardíaca                     | 0 = Sin Enfermedad, 1 = Enfermedad Cardíaca                    |

---

## 📁 Estructura del Proyecto

```
heart-disease-eda/
│
├── app.py                          # Flask: 23 funciones de gráficas Plotly + pipeline ML + 3 rutas
│
├── templates/
│   └── index.html                  # Dashboard HTML con Bootstrap 5 (10 secciones)
│
├── static/
│   └── css/
│       └── style.css               # Tema médico personalizado
│
├── data/
│   └── Heart_disease_cleveland_new.csv   # Dataset UCI Cleveland (303 obs)
│
├── Dockerfile                      # Imagen Python 3.11-slim
├── docker-compose.yml              # Orquestación del contenedor
├── requirements.txt                # Dependencias Python (incl. scikit-learn)
└── README.md
```

---

## 📈 Visualizaciones (21 gráficas)

El dashboard está dividido en **8 secciones** temáticas:

### 1. 🔍 Resumen General
| Gráfica | Tipo | Descripción |
|---------|------|-------------|
| Distribución de la Variable Objetivo | Donut chart | Balance de clases: pacientes con/sin enfermedad |

### 2. 👥 Análisis Demográfico
| Gráfica | Tipo | Descripción |
|---------|------|-------------|
| Distribución de Edad por Diagnóstico | Histograma overlay | Comparación de rangos etarios entre grupos |
| Distribución por Sexo y Diagnóstico | Barras agrupadas | Prevalencia de enfermedad por género |

### 3. 🩺 Variables Clínicas
| Gráfica | Tipo | Descripción |
|---------|------|-------------|
| Distribuciones de Variables Numéricas | 4 histogramas (subplots) | Presión, colesterol, FC máxima, Oldpeak |
| Box Plots de Variables Numéricas | 5 box plots (subplots) | Mediana, IQR y outliers por diagnóstico |
| FC Máxima — Violin Plot | Violin plot | Distribución completa de la frecuencia cardíaca |

### 4. 🔗 Análisis de Correlación
| Gráfica | Tipo | Descripción |
|---------|------|-------------|
| Mapa de Calor de Correlación | Heatmap RdBu | Correlaciones entre todas las variables |
| Correlación con Variable Objetivo | Barras horizontales | Coeficientes de Pearson ordenados |

### 5. 📊 Análisis Bivariado — Variables Nominales
| Gráfica | Tipo | Descripción |
|---------|------|-------------|
| Tipo de Dolor Torácico vs Diagnóstico | Barras apiladas | `cp` vs `target` |
| Talasemia vs Diagnóstico | Barras apiladas | `thal` vs `target` |
| Pendiente del Segmento ST vs Diagnóstico | Barras apiladas | `slope` vs `target` |
| Número de Vasos Coloreados vs Diagnóstico | Barras apiladas | `ca` vs `target` |
| ECG en Reposo vs Diagnóstico | Barras apiladas | `restecg` vs `target` |
| Angina Inducida por Ejercicio vs Diagnóstico | Barras apiladas | `exang` vs `target` |
| Glucosa en Ayunas vs Diagnóstico | Barras apiladas | `fbs` vs `target` |

### 6. 🔵 Distribución Multivariable
| Gráfica | Tipo | Descripción |
|---------|------|-------------|
| Edad vs FC Máxima | Scatter + línea OLS | Tendencia inversa edad–frecuencia cardíaca |
| Edad vs Colesterol (tamaño = Oldpeak) | Scatter burbuja | Relación tridimensional con Oldpeak como tamaño |

### 7. 📉 Tasas de Enfermedad
| Gráfica | Tipo | Descripción |
|---------|------|-------------|
| Tasa de Enfermedad por Grupo de Edad | Barras | % de enfermedad en grupos 20–40, 40–50, 50–60, 60–70, 70+ |
| Tasa de Enfermedad por Sexo | Barras | Comparación de tasas hombre/mujer |
| Tasa de Enfermedad por Tipo de Dolor | Barras horizontales | Impacto del tipo de `cp` en la prevalencia |

### 8. 🔀 Vista Multidimensional
| Gráfica | Tipo | Descripción |
|---------|------|-------------|
| Coordenadas Paralelas | Parallel coordinates | Trayectorias multivariables de 6 variables clave |

---

## 🛠️ Constructor Interactivo de Gráficas

La **Sección 9** del dashboard permite a cualquier usuario construir gráficas personalizadas sin escribir código:

- **Eje X / Eje Y:** Selecciona cualquiera de las 14 variables del dataset.
- **Tipo de gráfica:** Dispersión (`scatter`), Barras (`bar`), Caja (`box`), Violín (`violin`), Histograma (`histogram`).
- **Color por:** Colorea los puntos/barras por `target` (diagnóstico), `sex`, `cp`, `thal` u otras variables nominales.
- **Línea de tendencia OLS:** Cuando ambas variables son numéricas y el tipo es `scatter`, se agrega automáticamente la línea de regresión.

La gráfica se genera en tiempo real al hacer clic en **"Generar Gráfica"** mediante una llamada `fetch` al endpoint `/api/chart-builder`.

---

## 🤖 Predicción con Machine Learning

### Algoritmo Seleccionado: Regresión Logística

Después de comparar **4 algoritmos de clasificación** mediante validación cruzada estratificada de 5 pliegues, el modelo ganador fue la **Regresión Logística**.

#### ¿Qué es la Regresión Logística?

La Regresión Logística es un modelo de clasificación que estima la **probabilidad** de pertenecer a una clase (en este caso, tener enfermedad cardíaca). Aunque se llama "regresión", es un clasificador:

1. Combina linealmente las variables de entrada: `z = β₀ + β₁·x₁ + β₂·x₂ + ... + β₁₃·x₁₃`  
2. Aplica la función sigmoide para convertir `z` en una probabilidad: `P(enfermedad) = 1 / (1 + e⁻ᶻ)`  
3. Si `P ≥ 0.5` → predice **Enfermedad Cardíaca**; si `P < 0.5` → predice **Sin Enfermedad**

Los coeficientes `β` se interpretan como **log-odds** (logaritmo de la razón de probabilidades). Un coeficiente positivo grande significa que esa variable aumenta significativamente el riesgo.

En el pipeline implementado, las variables se normalizan primero con **StandardScaler** (media 0, desviación 1) para que los coeficientes sean comparables entre sí.

#### ¿Por qué Regresión Logística para este dataset?

| Razón | Explicación |
|-------|-------------|
| **Dataset pequeño (303 filas)** | Los modelos complejos (Random Forest, Gradient Boosting) sobreajustan con tan pocos datos |
| **Variables clínicas con relación lineal** | Variables como `ca`, `thal`, `cp` tienen una relación casi monotónica con el diagnóstico |
| **Alta interpretabilidad** | Los coeficientes explican directamente el aporte de cada variable al riesgo |
| **AUC-ROC de 95.13%** | Confirma excelente separabilidad entre clases |
| **Menor varianza en CV** | ±4.96% vs ±6-8% en los otros modelos |

#### Variables más importantes (coeficientes normalizados)

| Variable | Coeficiente | Interpretación |
|----------|-------------|----------------|
| `ca` | +1.089 | Más vasos bloqueados → mayor riesgo |
| `thal` | +0.686 | Defecto reversible → mayor riesgo |
| `sex` | +0.650 | Sexo masculino → mayor riesgo observado |
| `cp` | +0.539 | Tipo asintomático → mayor riesgo paradójico |
| `exang` | +0.378 | Angina por ejercicio → mayor riesgo |
| `thalach` | −0.507 | Mayor FC máxima → menor riesgo |
| `oldpeak` | +0.407 | Mayor depresión ST → mayor riesgo |

### Comparación de Algoritmos

| Algoritmo | CV Accuracy (5-fold) | Test Accuracy | AUC-ROC |
|-----------|---------------------|---------------|---------|
| **Regresión Logística** ⭐ | **83.16% ± 4.96%** | **86.89%** | **95.13%** |
| Random Forest | 81.54% ± 5.82% | 85.25% | 93.47% |
| SVM | 82.18% ± 5.31% | 85.25% | 92.88% |
| Gradient Boosting | 78.54% ± 6.14% | 80.33% | 90.12% |

### Módulo de Predicción (Sección 10)

La sección de predicción permite ingresar los 13 valores clínicos de un paciente y obtener:

- ✅ Diagnóstico: **Enfermedad Cardíaca** o **Sin Enfermedad**
- 📊 Probabilidad de enfermedad (en %)
- 🔴 Nivel de riesgo: **Bajo** (< 30%), **Moderado** (30–70%), **Alto** (> 70%)
- 📈 Gráficas de Importancia de Variables y Matriz de Confusión del modelo

---

## 🚀 Instalación y Ejecución

### Opción 1: Ejecución Local (Python)

**Requisitos:** Python 3.9+

```bash
# 1. Clonar el repositorio
git clone https://github.com/YeisonBue/Heart-disease-eda.git
cd Heart-disease-eda

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Iniciar la aplicación
python app.py
```

Abrir en el navegador: **http://localhost:5000**

---

### Opción 2: Ejecución con Docker (Recomendado)

**Requisitos:** Docker + Docker Compose instalados

```bash
# 1. Clonar el repositorio
git clone https://github.com/YeisonBue/Heart-disease-eda.git
cd Heart-disease-eda

# 2. Construir y levantar el contenedor
docker-compose up --build

# Para ejecutar en segundo plano
docker-compose up --build -d
```

Abrir en el navegador: **http://localhost:5000**

**Detener el contenedor:**
```bash
docker-compose down
```

**Ver logs:**
```bash
docker-compose logs -f
```

---

## 🔌 API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Dashboard principal |
| `GET` | `/api/chart-builder` | Genera una gráfica Plotly personalizada |
| `POST` | `/api/predict` | Predice el diagnóstico de un paciente |

### GET `/api/chart-builder`

**Parámetros query:**

| Parámetro | Tipo | Valores posibles | Descripción |
|-----------|------|-----------------|-------------|
| `x` | string | cualquier variable | Variable para el eje X |
| `y` | string | cualquier variable | Variable para el eje Y |
| `chart_type` | string | `scatter`, `bar`, `box`, `violin`, `histogram` | Tipo de gráfica |
| `color_by` | string | `target`, `sex`, `cp`, `thal`, `none` | Variable de color |

**Ejemplo:**
```
GET /api/chart-builder?x=age&y=chol&chart_type=scatter&color_by=target
```

**Respuesta:** JSON con estructura Plotly (`data` + `layout`), listo para `Plotly.newPlot()`.

---

### POST `/api/predict`

**Body JSON:**
```json
{
  "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
  "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
  "exang": 0, "oldpeak": 2.3, "slope": 1, "ca": 1, "thal": 2
}
```

**Respuesta:**
```json
{
  "label": "Enfermedad Cardíaca",
  "probability": 86.2,
  "risk_level": "Alto",
  "model": "Logistic Regression"
}
```

---

## 🛠️ Tecnologías

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **Python** | 3.11 | Lenguaje base |
| **Flask** | ≥3.0 | Servidor web y enrutamiento |
| **Plotly** | ≥5.18 | Gráficas interactivas |
| **Pandas** | ≥2.1 | Manipulación y análisis de datos |
| **NumPy** | ≥1.26 | Operaciones numéricas |
| **scikit-learn** | ≥1.4 | Pipeline ML, modelos de clasificación, métricas |
| **statsmodels** | ≥0.14 | Regresión OLS para líneas de tendencia |
| **Bootstrap** | 5 | Framework CSS responsivo |
| **Docker** | — | Contenerización y despliegue |
| **Gunicorn** | ≥21.2 | Servidor WSGI de producción (Docker) |

---

## 👥 Equipo

Proyecto de **Electiva de Grado** — Ingeniería de Sistemas

| Nombre | Rol |
|--------|-----|
| **Yeison Buelvas** | Desarrollo |
| **Juan Díaz** | Desarrollo |
| **Jeison Díaz** | Desarrollo |
| **Juan Pájaro** | Desarrollo |

---

## 📚 Referencias

- Dataset original: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Referencia EDA: [Kaggle — Heart Disease EDA by Spandan Ghodke](https://www.kaggle.com/code/spandanghodke/heartdisease-dataset-eda)
- Detro, R. H., & Jankowski, T. A. (1989). *Cleveland Clinic Foundation Dataset*

---

<p align="center">
  Hecho con ❤️ para el análisis clínico cardiovascular
</p>
