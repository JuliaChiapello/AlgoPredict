# AlgoPredict 📊🔥
> **Predicción y análisis del tiempo de ejecución de algoritmos de ordenamiento y búsqueda.**

Plataforma experimental e interactiva para medir, modelar y predecir el comportamiento temporal de algoritmos clásicos, combinando benchmarking empírico, teoría de algoritmos y Machine Learning.

Desarrollado con **Python, Flask y MongoDB**, con un enfoque estricto en ingeniería de datos, criterio algorítmico y diseño experimental reproducible.

---

## 💡 Motivación del Proyecto
En la práctica profesional, la complejidad algorítmica rara vez se comporta exactamente como indica la teoría asintótica. Factores del entorno real como el tipo de datos, el estado de ordenamiento previo, el *overhead* del lenguaje de programación y las interrupciones del hardware/sistema operativo provocan que el **tiempo real de ejecución** difiera de la cota teórica esperada.

**AlgoPredict** nace para solucionar esta brecha:
*   **Medir** tiempos reales de ejecución bajo escenarios controlados.
*   **Modelar** tiempos teóricos basados en complejidad matemática (O(n), \(O(n \log n)\), O(n²)).
*   **Comparar** la discrepancia empírica vs. teórica.
*   **Predecir** tiempos de ejecución para tamaños de entrada no observados.

---

## 🧠 Enfoque de Modelado y Machine Learning

El sistema implementa una **arquitectura de doble modelo**, separando la naturaleza de los datos como una decisión de ingeniería consciente:

### 🔹 1. Predicción de Tiempos Reales (Modelo Empírico)
*   **Algoritmo:** `HistGradientBoostingRegressor`
*   **Técnicas adicionales:** Transformación logarítmica del *target* y optimización de hiperparámetros mediante `GridSearchCV`.
*   **Justificación:** Captura relaciones no lineales complejas, demuestra robustez ante el ruido intrínseco del hardware, escala eficientemente y procesa de forma nativa variables categóricas mixtas. *El Machine Learning se aplica aquí donde la teoría abstracta no puede parametrizar el entorno.*

### 🔹 2. Predicción de Tiempos Teóricos (Modelo Matemático)
*   **Algoritmo:** Regresión Polinómica + Regularización `Ridge`
*   **Técnicas adicionales:** Generación de *features* polinómicas adaptadas al tamaño de la entrada (n).
*   **Justificación:** Dado que el crecimiento algorítmico tiene una forma matemática conocida, se prioriza la interpretabilidad estricta sobre la complejidad. La regularización evita el sobreajuste y permite al modelo aprender los coeficientes de crecimiento reales. *El ML acompaña a la teoría, no la reemplaza.*

---

## ⚙️ Ingeniería de Datos y Generación del Dataset

El conjunto de datos se construye de forma determinística, parametrizada y reproducible bajo los siguientes estándares de diseño experimental:

*   **Diversidad de Escenarios:** Cobertura de algoritmos iterativos/recursivos, búsqueda/ordenación, tamaños de entrada variables y diferentes estados de ordenamiento inicial (*sorted*, *reverse*, *random*).
*   **Mitigación de Ruido:** Medición de precisión mediante `time.perf_counter()` utilizando la **mediana** de múltiples ejecuciones consecutivas para neutralizar picos aislados de consumo de CPU.
*   **Rendimiento:** Paralelización de las pruebas de carga mediante el módulo `multiprocessing` de Python.
*   **Persistencia:** Almacenamiento masivo indexado en **MongoDB** para consultas dinámicas y soporte de reentrenamiento.

---

## 🛠️ Stack Tecnológico

| Capa | Tecnologías Utilizadas |
| :--- | :--- |
| **Backend & Core** | Python 3.11+, Flask, PyMongo, Jinja2 |
| **Data Science / ML** | NumPy, Pandas, Scikit-learn |
| **Base de Datos** | MongoDB |
| **Frontend & UI** | HTML5, TailwindCSS (Dark Mode Nativo) |

---

## 🚀 Funcionalidades Principales

1.  **Predicción Interactiva Inteligente:** Interfaz web donde el usuario parametriza una consulta y el sistema decide internamente qué modelo invocar (teórico o empírico) según las características y rango del tamaño de entrada.
2.  **Exploración del Dataset Avanzada:** Tabla de datos masiva totalmente navegable con filtros dinámicos independientes y persistentes por columna que operan directamente sobre la base de datos sin romper la paginación optimizada.
3.  **Procesos Asíncronos en Background:** Módulos aislados para la regeneración del dataset y el reentrenamiento de los modelos predictivos con bloqueos seguros de rutas críticas y auditoría mediante *logs* de estado.

---

## 📁 Estructura del Proyecto

```text
AlgoPredict/
│
├── app/
│   ├── algorithms.py       # Implementación y benchmarking de algoritmos clásicos
│   └── model.py            # Lógica de entrenamiento y predicción de la arquitectura de ML
│
├── templates/              # Vistas HTML renderizadas con Jinja2 (TailwindCSS)
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── train.html
│   ├── generate_dataset.html
│   └── dataset.html
│
├── app.py                  # Punto de entrada de la aplicación Flask
├── .env.example            # Configuración de variables de entorno (MongoDB Connection)
├── dualModelTrain.pkl      # Serialización de los modelos entrenados
├── requirements.txt        # Dependencias del proyecto
└── README.md
```

---

## 💻 Instalación y Despliegue Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/juliachiapello/AlgoPredict.git
cd AlgoPredict
```

### 2. Configurar el Entorno Virtual
```bash
python -m venv venv
# Activar en Linux/Mac:
source venv/bin/activate  
# Activar en Windows:
venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno
Crea un archivo `.env` basado en `.env.example` y configura tu cadena de conexión a MongoDB.

### 5. Ejecutar la Aplicación
```bash
python app.py
```

---

## 📈 Próximas Mejoras Planificadas
*   [ ] Inserción de gráficos interactivos dinámicos de curvas de rendimiento (Real vs. Teórico) mediante Chart.js o Plotly.
*   [ ] Panel de control (Dashboard) centralizado para métricas de error del modelo de Machine Learning ($R^2$, MAE).
*   [ ] Exportación automatizada de los datasets generados a formatos estándar (.csv, .json).

---

## 👤 Autora
**Julia Gabriela Chiapello** 
*Estudiante Avanzada de Analista en Computación (FaMAF - UNC) & Técnica Superior en Bromatología.*

Proyecto desarrollado como pieza de portafolio profesional de alto rendimiento con foco en: **Ingeniería de datos, Criterio algorítmico, Buenas prácticas de Machine Learning y Diseño experimental.**
