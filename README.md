# AlgoPredict  
**Predicción y análisis del tiempo de ejecución de algoritmos de ordenamiento y búsqueda**

Plataforma experimental e interactiva para **medir, modelar y predecir** el comportamiento temporal de algoritmos clásicos, combinando **benchmarking empírico**, **teoría de algoritmos** y **Machine Learning**.

Desarrollado con **Python, Flask y MongoDB**, con un enfoque en **ingeniería de datos**, **criterio algorítmico** y **diseño experimental reproducible**.

---

## Motivación del proyecto

En la práctica profesional, la complejidad algorítmica rara vez se comporta exactamente como la teoría indica.  
Factores como:

- tipo de datos  
- estado de ordenamiento  
- implementación concreta  
- overhead del lenguaje  
- hardware y sistema operativo  

hacen que el **tiempo real de ejecución** difiera del **tiempo teórico esperado**.

**AlgoPredict** nace con el objetivo de:

- medir **tiempos reales** de ejecución bajo escenarios controlados  
- modelar **tiempos teóricos** a partir de complejidad algorítmica  
- comparar ambos mundos  
- y **predecir tiempos de ejecución** para entradas no observadas  

Todo esto desde una **plataforma reproducible, extensible y explicable**.

---

## ¿Qué se quiere demostrar?

Este proyecto busca demostrar que:

1. El comportamiento real de los algoritmos **no es lineal ni trivial**
2. La teoría algorítmica sigue siendo válida, pero **requiere contexto**
3. El Machine Learning es útil **solo cuando el fenómeno lo justifica**
4. Un buen diseño experimental es tan importante como el modelo elegido

Además, el proyecto refleja **criterio técnico** al elegir **diferentes enfoques de modelado** según el tipo de problema.

---

## Enfoque general

El sistema trabaja con **dos tipos de datos claramente diferenciados**:

### 🔹 1. Tiempos reales
- Medidos empíricamente ejecutando algoritmos reales
- Incluyen ruido, variabilidad y efectos del entorno
- Se utilizan **modelos no lineales** para capturar su comportamiento

### 🔹 2. Tiempos teóricos
- Generados a partir de la complejidad algorítmica esperada  
  (O(n), O(n log n), O(n²))
- Se modelan mediante **regresión polinómica regularizada**
- Se prioriza **interpretabilidad y coherencia matemática**

Esta separación no es casual:  
    es una **decisión de ingeniería**, no de conveniencia.

---

## Modelado y Machine Learning

### Predicción de tiempos reales
Se utiliza:

- **HistGradientBoostingRegressor**
- Transformación logarítmica del target
- Búsqueda de hiperparámetros con **GridSearchCV**

**Motivo de la elección**:
- Captura no linealidades
- Es robusto al ruido
- Escala bien
- Funciona correctamente con features mixtas

Este modelo se utiliza **únicamente donde la teoría no alcanza**.

---

### Predicción de tiempos teóricos
Se utiliza:

- **Regresión Polinómica + Ridge**
- Features polinómicas sobre el tamaño de entrada
- Regularización para evitar sobreajuste

**Motivo de la elección**:
- El crecimiento algorítmico tiene forma conocida
- Se prioriza interpretabilidad sobre complejidad
- El modelo aprende coeficientes de crecimiento reales

Aquí el ML **acompaña a la teoría**, no la reemplaza.

---

## Generación del dataset

El dataset se construye de forma **determinística y reproducible**:

- Algoritmos iterativos y recursivos
- Ordenamiento y búsqueda
- Diferentes tamaños de entrada
- Diferentes tipos de datos
- Estados ordenados y desordenados

### Características del benchmark:

- Medición con `perf_counter`
- Uso de la **mediana** para reducir ruido
- Paralelización con `multiprocessing`
- Supuestos experimentales explícitos y controlados

El dataset completo se almacena en **MongoDB** y puede ser regenerado en cualquier momento.

---

## Funcionalidades principales

### 🔹 Predicción interactiva

El usuario puede:
- Elegir algoritmo
- Definir tipo de dato
- Indicar si la entrada está ordenada
- Seleccionar tamaño de entrada
- Obtener una predicción automática
- Guardar predicciones en base de datos

El sistema decide internamente si utilizar:
- modelo real
- o modelo teórico  
según el rango de entrada.

---

### 🔹 Exploración del dataset
- Filtros dinámicos por columna
- Paginación completa
- Persistencia de filtros
- Dataset masivo navegable
- Preparado para análisis exploratorio

---

### 🔹 Procesos en background
- Generación de dataset
- Entrenamiento / reentrenamiento de modelos
- Bloqueo de rutas críticas
- Logs claros y control de estado

---

## Tecnologías utilizadas

### Backend
- Python 3.11+
- Flask
- MongoDB
- PyMongo
- Jinja2

### Ciencia de datos / ML
- NumPy
- Pandas
- Scikit-learn

### Frontend
- HTML
- TailwindCSS (dark mode)
- UI minimalista y responsive

---

## Funcionalidades Implementadas

### 1. Filtros dinámicos para columnas
Las siguientes columnas tiene su propio dropdown:
- `algorithm`
- `numElements`
- `dataType`
- `sorted`

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

## Estructura del proyecto
```
AlgoPredict/
    app/
        algorithms.py
        model.py
    templates/
        base.html
        index.html
        predict.html
        train.html
        generate_dataset.html
        dataset.html
    app.py
    .env.example
    dualModelTrain.pkl
    README.md
    requirements.txt
```

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

## Próximas Mejoras
- Visualizaciones comparativas (real vs teórico)
- Dashboards interactivos
- Exportación de datasets

## Autora
Julia Gabriela Chiapello

Proyecto desarrollado como pieza de portfolio profesional,
con foco en:

- Ingeniería de datos
- Criterio algorítmico
- Buenas prácticas de ML
- Diseño experimental
