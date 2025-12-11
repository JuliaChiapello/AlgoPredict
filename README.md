# AlgoPredict  
Plataforma interactiva para analizar, comparar y visualizar el rendimiento real y teórico de algoritmos de ordenamiento y busqueda (iterativos y recursivos).  
Desarrollada con **Flask + MongoDB** y estilizada con **TailwindCSS (dark mode)**.

---

## Descripción
**AlgoPredict** es un sistema completo que:
- Genera datasets masivos con tiempos reales y teóricos de algoritmos de ordenamiento.
- Permite visualizar, filtrar y explorar miles de registros.
- Incluye una UI moderna, oscura, fluida y responsiva.
- Ofrece filtros dinámicos por cada columna de forma independiente.
- Implementa paginación completa con botones *Primero / Anterior / Siguiente / Último*.
- Está listo para incorporar dashboards gráficos interactivos (Chart.js o Plotly).

Es un proyecto ideal para analizar comportamientos de algoritmos, documentar benchmarks o enseñar estructuras de datos.

---

## Tecnologías Utilizadas

### Backend
- Python 3.13  
- Flask  
- MongoDB  
- PyMongo  

### Frontend
- HTML + Jinja2  
- TailwindCSS (dark mode)  
- Animaciones suaves y diseño minimalista  

## Tecnologías de Procesamiento y Ciencia de Datos

Este proyecto incorpora un conjunto de herramientas orientadas al análisis, manipulación de datos y generación de predicciones teóricas para los algoritmos evaluados:

- **Pandas**: Manejo y manipulación de estructuras tabulares (DataFrames), filtrado, ordenamiento y limpieza de datos.
- **NumPy**: Cálculo numérico de alto rendimiento, operaciones vectorizadas y soporte matemático para los modelos teóricos.
- **Scikit-Learn**: Utilizado para entrenar modelos de regresión (Lineal, Polinomial, etc.) aplicados a la predicción teórica del tiempo de ejecución de cada algoritmo.
- **Matplotlib** (próxima integración): Visualización gráfica comparativa entre tiempos reales y teóricos.
- **Python Standard Library**: Módulos nativos como `time`, `math` y `statistics` complementan el procesamiento de datos y cálculos estadísticos internos.

### Base de datos
- Colección única `Dataset`
- Orden descendente por `_id`
- Filtros y paginación totalmente integrados

---

## Funcionalidades Implementadas

### 1. Filtros dinámicos por columna
Cada columna tiene su propio dropdown:
- `algorithm`
- `numElements`
- `dataType`
- `sorted`
- `type` (real / theoretical)

Los filtros:
- Son independientes  
- Persisten entre sí  
- No rompen la paginación  
- Se regeneran dinámicamente desde MongoDB  

---

### 2. Paginación profesional  
Botones incluidos:
- ⏮ Primero  
- ◀️ Anterior  
- ▶️ Siguiente  
- ⏭ Último  

Características:
- Disponible **arriba y abajo** de la tabla  
- Compatible con filtros  
- URLs limpias usando GET  
- Estilo profesional con TailwindCSS  

---

### 3. UI Profesional
- Tema oscuro por defecto  
- Tabla responsive  
- Hover states  
- Dropdowns centrados  
- Transiciones suaves  
- Layout limpio y elegante  

---

### 4. Código ordenado y mantenible
- Rutas simples y claras  
- Paginación manual optimizada  
- Uso correcto de Jinja2  
- Variables de contexto limpias y explícitas  
- Separación lógica del backend y frontend  

---

## Próximas Mejoras
- Gráficos con Chart.js / Plotly  
- Exportación CSV / XLSX / JSON  
- Dashboard interactivo  
- Benchmark multiproceso  
- Panel de análisis avanzado  

---

## Estructura del proyecto

AlgoPredict/
|---app/
|   |---algorithms.py
|   |---model.py
|---templates/
|   |---base.html
|   |---index.html
|   |---predict.html
|   |---train.html
|   |---generate_dataset.html
|   |---dataset.html
|---app.py
|---dualModelTrain.pkl

---

## Cómo ejecutar

### 1. Clonar el repo:
```bash
git clone https://github.com/juliachiapello/AlgoPredict.git
cd AlgoPredict
```
---

### 2. Creacion de entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
---

### 3.Instalar dependencias:
```bash
pip install -r requirements.txt
```

### 4. Ejecucion:
```bash
python app.py
```

## Autora
Julia Gabriela Chiapello

