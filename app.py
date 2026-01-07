"""
Aplicación Flask principal del proyecto.

Responsabilidades:
- Exponer una interfaz web para:
  - Predicción de tiempos de ejecución de algoritmos
  - Entrenamiento del modelo
  - Generación del dataset
  - Visualización del dataset almacenado
- Coordinar la interacción entre:
  - Frontend (Jinja2)
  - Base de datos MongoDB
  - Modelos entrenados (archivo .pkl)
"""

import os
import pickle
import threading
import time
from flask import (
    Flask, 
    render_template, 
    request, 
    redirect, 
    url_for, 
    flash, 
    jsonify
)    
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np
import pandas as pd

# ==========================================================
# CARGA DE VARIABLES DE ENTORNO
# ==========================================================
# Permite leer variables desde un archivo .env
# (ej: MONGO_URI, DATABASE_NAME, FLASK_SECRET)

load_dotenv()

# ==========================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# ==========================================================

# Ruta absoluta del directorio del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuracion de Mongo
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "AlgorithmsDB")

# Ruta del modelo entrenado
MODEL_PKL_PATH = os.path.join(BASE_DIR, "dualModelTrain.pkl")

# ==========================================================
# INICIALIZACIÓN DE FLASK
# ==========================================================

app = Flask(__name__, 
            template_folder="templates", 
            static_folder="static"
)

# Clave secreta usada por Flask para sesiones y mensajes flash
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")

# ==========================================================
# CONTROL GLOBAL DE PROCESOS PESADOS
# ==========================================================

"""
Esta variable se usa para evitar que el usuario:
- Prediga
- Navegue el dataset

mientras se está:
- Generando el dataset
- Entrenando el modelo

Evita inconsistencias y errores de concurrencia.
"""
PROCESS_RUNNING = False

# ==========================================================
# CONEXIÓN A MONGODB
# ==========================================================

def get_db_client():
    """
    Crea y devuelve un cliente de MongoDB.
    """
    return MongoClient(MONGO_URI)

def get_db():
    """
    Devuelve la base de datos configurada.
    """
    client = get_db_client()
    return client[DATABASE_NAME]

# ==========================================================
# CARGA DEL MODELO ENTRENADO
# ==========================================================

def load_models():
    """
    Carga el archivo dualModelTrain.pkl si existe.

    Retorna:
    - Diccionario con modelos, encoder y escaladores
    - None si el archivo no existe
    """
    if not os.path.exists(MODEL_PKL_PATH):
        return None
    with open(MODEL_PKL_PATH, "rb") as f:
        return pickle.load(f)

# ==========================================================
# MIDDLEWARE DE BLOQUEO DE RUTAS CRÍTICAS
# ==========================================================

@app.before_request
def check_process_running():
    """
    Middleware que se ejecuta antes de cada request.

    Si hay un proceso pesado en ejecución, bloquea
    rutas sensibles para evitar conflictos.
    """
    global PROCESS_RUNNING
    blocked_routes = ["predict_route", "dataset_view"]
    if PROCESS_RUNNING and request.endpoint in blocked_routes:
        flash("🚧 Hay un proceso en curso (dataset o entrenamiento). Por favor espera a que termine.", "warning")
        return redirect(url_for("index"))

# ==========================================================
# RUTAS PRINCIPALES (JINJA2)
# ==========================================================

@app.route("/")
def index():
    """
    Página principal del sistema.
    """
    return render_template("index.html")

# ==========================================================
# RUTA DE PREDICCIÓN
# ==========================================================

@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    """
    Permite realizar predicciones de tiempo de ejecución
    usando el modelo entrenado.
    """
    models_data = load_models()
    algorithms = [
        "IterativeQuickSort", "RecursiveQuickSort",
        "IterativeMergeSort", "RecursiveMergeSort",
        "IterativeInsertionSort", "IterativeBubbleSort",
        "IterativeSelectionSort", "IterativeBinarySearch",
        "RecursiveBinarySearch", "IterativeSequentialSearch",
        "RecursiveSequentialSearch"
    ]

    # ======================
    # Lectura del formulario
    # ======================

    if request.method == "POST":
        algorithm = request.form.get("algorithm")
        dataType = request.form.get("dataType")
        sorted_val = request.form.get("sorted") == "true"
        try:
            numElements = int(request.form.get("numElements"))
        except Exception:
            flash("Cantidad de elementos inválida", "danger")
            return redirect(url_for("predict_route"))
        
        # Heurística:
        # datasets chicos → modelo real
        # datasets grandes → modelo teórico
        prediction_type = "real" if numElements <= 2450 else "theoretical"
        save_pred = request.form.get("save_prediction") == "on"

        # ======================
        # Validaciones
        # ======================

        if models_data is None:
            flash("No hay modelo entrenado. Entrená primero desde la sección 'Entrenar'.", "danger")
            return redirect(url_for("predict_route"))

        # ======================
        # Extracción de modelos
        # ======================

        encoder = models_data["encoder"]
        scaler_X = models_data["scaler_X"]
        scaler_y = models_data["scaler_y"]
        model_real = models_data["model_real"]
        model_theoretical = models_data["model_theoretical"]
        cat_columns = models_data.get("cat_columns", ["algorithm","dataType","sorted"])
        num_columns = models_data.get("num_columns", ["numElements"])

        # ======================
        # Preparación del input
        # ======================

        input_df = pd.DataFrame([{
            "algorithm": algorithm,
            "dataType": dataType,
            "sorted": sorted_val,
            "numElements": numElements
        }])

        X_cat = encoder.transform(input_df[cat_columns])
        X_num = input_df[num_columns].values
        X_input = np.hstack((X_cat, X_num))

        # ======================
        # Predicción
        # ======================

        if prediction_type == "real":
            # El modelo real fue entrenado en escala log
            y_log_pred = model_real.predict(X_input)
            y_pred = float(np.expm1(y_log_pred)[0])
        else:
            # El modelo teórico trabaja con variables escaladas
            X_scaled = scaler_X.transform(X_input)
            y_scaled = model_theoretical.predict(X_scaled)
            y_pred = float(scaler_y.inverse_transform(y_scaled.reshape(-1,1)).ravel()[0])

        # Evitar valores negativos
        y_pred_value = max(y_pred, 0.0)

        # ======================
        # Guardado opcional
        # ======================

        if save_pred:
            db = get_db()
            db.UserPredictions.insert_one({
                "timestamp": time.time(),
                "algorithm": algorithm,
                "dataType": dataType,
                "sorted": sorted_val,
                "numElements": numElements,
                "prediction_type": prediction_type,
                "predicted_time": y_pred_value
            })

        return render_template("predict.html", algorithms=algorithms, result=y_pred_value, form=request.form)

    return render_template("predict.html", algorithms=algorithms, result=None)

# ==========================================================
# RUTA DE ENTRENAMIENTO
# ==========================================================

@app.route("/train", methods=["GET", "POST"])
def train_route():
    """
    Lanza el entrenamiento del modelo en un hilo separado
    para no bloquear la aplicación Flask.
    """
    global PROCESS_RUNNING
    if request.method == "POST":
        try:
            from app.model import train as train_model
        except Exception as e:
            flash(f"No se pudo importar train(): {e}", "danger")
            return redirect(url_for("train_route"))

        def worker():
            global PROCESS_RUNNING
            PROCESS_RUNNING = True
            try:
                train_model()
                app.logger.info("Entrenamiento finalizado.")
            except Exception as ex:
                app.logger.exception("Error en entrenamiento: %s", ex)
            finally:
                PROCESS_RUNNING = False

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        flash("Entrenamiento iniciado en background. Revisa la consola para logs.", "info")
        return redirect(url_for("train_route"))

    return render_template("train.html")

# ==========================================================
# RUTA DE GENERACION DE DATASET
# ==========================================================

@app.route("/generate_dataset", methods=["GET", "POST"])
def generate_dataset():
    global PROCESS_RUNNING
    if request.method == "POST":
        try:
            from app.algorithms import generateDeterministicPredictions, generateTheoreticalPredictions
        except Exception as e:
            flash(f"No se pudieron importar las funciones de algorithms.py: {e}", "danger")
            return redirect(url_for("index"))

        db = get_db()

        def worker():
            global PROCESS_RUNNING
            PROCESS_RUNNING = True
            try:
                db.Dataset.delete_many({})
                data1 = generateDeterministicPredictions()
                data2 = generateTheoreticalPredictions(data1)
                data = data1 + data2
                if data:
                    db.Dataset.insert_many(data)
                app.logger.info("Dataset generado e insertado.")
            except Exception as ex:
                app.logger.exception("Error al generar dataset: %s", ex)
            finally:
                PROCESS_RUNNING = False

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        flash("Generación del dataset iniciada en background. Puede tardar varios minutos.", "info")
        return redirect(url_for("index"))

    return render_template("generate_dataset.html")

# ==========================================================
# RUTA DE VISUALIZACIÓN DEL DATASET
# ==========================================================

@app.route("/dataset")
def dataset_view():
    db = get_db()
    if db.Dataset.count_documents({}) == 0:
        flash("No hay dataset disponible todavía. Generalo primero.", "warning")
        return redirect(url_for("index"))
    
    # ======================
    # Filtros desde la URL
    # ======================

    selected_algorithm = request.args.get("algorithm", "")
    selected_numElements = request.args.get("numElements", "")
    selected_dataType = request.args.get("dataType", "")
    selected_sorted = request.args.get("sorted", "")
    selected_type = request.args.get("type", "")

    query = {}
    if selected_algorithm: query["algorithm"] = selected_algorithm
    if selected_numElements: query["numElements"] = int(selected_numElements)
    if selected_dataType: query["dataType"] = selected_dataType
    if selected_sorted: query["sorted"] = selected_sorted.lower() in ["true", "1"]
    if selected_type: query["type"] = selected_type

    docs = list(db.Dataset.find(query).sort("_id", -1))
    count = len(docs)

    # ======================
    # Paginación
    # ======================

    per_page = 20
    page = int(request.args.get("page", 1))
    total_pages = (count + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    docs_paginated = docs[start:end]

    # ===========================
    # Valores únicos para filtros
    # ===========================

    algorithms = sorted(list(db.Dataset.distinct("algorithm")))
    numElements_list = sorted(list(db.Dataset.distinct("numElements")))
    dataTypes = sorted(list(db.Dataset.distinct("dataType")))
    sorteds = sorted(list(db.Dataset.distinct("sorted")))
    types = sorted(list(db.Dataset.distinct("type")))

    return render_template(
        "dataset.html",
        rows=docs_paginated,
        count=count,
        algorithms=algorithms,
        numElements_list=numElements_list,
        dataTypes=dataTypes,
        sorteds=sorteds,
        types=types,
        selected_algorithm=selected_algorithm,
        selected_numElements=selected_numElements,
        selected_dataType=selected_dataType,
        selected_sorted=selected_sorted,
        selected_type=selected_type,
        page=page,
        per_page=per_page,
        total_pages=total_pages
    )

# ==========================================================
# RUTA DE VISUALIZACION DE DATOS (PREDICCION VS ACTUAL)
# ==========================================================

@app.route("/api/pred_vs_actual")
def api_pred_vs_actual():
    """
    Endpoint API simple para devolver datos del dataset
    y permitir visualizaciones (charts).
    """
    db = get_db()
    docs = list(db.Dataset.find().limit(10000))

    for d in docs:
        d["_id"] = str(d["_id"])

    return jsonify(docs)

# ==========================================================
# EJECUCIÓN LOCAL
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)
