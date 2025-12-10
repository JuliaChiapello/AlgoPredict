import os
import pickle
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env file
load_dotenv()

# Project absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "AlgorithmsDB")
MODEL_PKL_PATH = os.path.join(BASE_DIR, "dualModelTrain.pkl")

# Initialize Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")

# Global state to control heavy processes (Dataset Generation and Model Training/Retraining)
PROCESS_RUNNING = False

# MongoDB connection
def get_db_client():
    return MongoClient(MONGO_URI)

def get_db():
    client = get_db_client()
    return client[DATABASE_NAME]

# Load model from .pkl file (if it exists)
def load_models():
    if not os.path.exists(MODEL_PKL_PATH):
        return None
    with open(MODEL_PKL_PATH, "rb") as f:
        return pickle.load(f)

# ----------------------
# Middleware to block critical routes when a process is running
# ----------------------
@app.before_request
def check_process_running():
    global PROCESS_RUNNING
    blocked_routes = ["predict_route", "dataset_view"]
    if PROCESS_RUNNING and request.endpoint in blocked_routes:
        flash("🚧 Hay un proceso en curso (dataset o entrenamiento). Por favor espera a que termine.", "warning")
        return redirect(url_for("index"))

# ----------------------
# Jinja2 routes
# ----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    models_data = load_models()
    algorithms = [
        "IterativeQuickSort", "RecursiveQuickSort",
        "IterativeMergeSort", "RecursiveMergeSort",
        "IterativeInsertionSort", "IterativeBubbleSort",
        "IterativeSelectionSort", "IterativeBinarySearch",
        "RecursiveBinarySearch", "IterativeSequentialSearch",
        "RecursiveSequentialSearch"
    ]

    if request.method == "POST":
        algorithm = request.form.get("algorithm")
        dataType = request.form.get("dataType")
        sorted_val = request.form.get("sorted") == "true"
        try:
            numElements = int(request.form.get("numElements"))
        except Exception:
            flash("Cantidad de elementos inválida", "danger")
            return redirect(url_for("predict_route"))
        prediction_type = request.form.get("prediction_type")
        save_pred = request.form.get("save_prediction") == "on"

        if models_data is None:
            flash("No hay modelo entrenado. Entrená primero desde la sección 'Entrenar'.", "danger")
            return redirect(url_for("predict_route"))

        import numpy as np
        import pandas as pd

        encoder = models_data["encoder"]
        scaler_X = models_data["scaler_X"]
        scaler_y = models_data["scaler_y"]
        model_real = models_data["model_real"]
        model_theoretical = models_data["model_theoretical"]
        cat_columns = models_data.get("cat_columns", ["algorithm","dataType","sorted"])
        num_columns = models_data.get("num_columns", ["numElements"])

        input_df = pd.DataFrame([{
            "algorithm": algorithm,
            "dataType": dataType,
            "sorted": sorted_val,
            "numElements": numElements
        }])

        X_cat = encoder.transform(input_df[cat_columns])
        X_num = input_df[num_columns].values
        X_input = np.hstack((X_cat, X_num))

        if prediction_type == "real":
            y_log_pred = model_real.predict(X_input)
            y_pred = float(np.expm1(y_log_pred)[0])
        else:
            X_scaled = scaler_X.transform(X_input)
            y_scaled = model_theoretical.predict(X_scaled)
            y_pred = float(scaler_y.inverse_transform(y_scaled.reshape(-1,1)).ravel()[0])

        y_pred_value = max(y_pred, 0.0)

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

@app.route("/train", methods=["GET", "POST"])
def train_route():
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

@app.route("/dataset")
def dataset_view():
    db = get_db()
    if db.Dataset.count_documents({}) == 0:
        flash("No hay dataset disponible todavía. Generalo primero.", "warning")
        return redirect(url_for("index"))

    # Filters from query parameters
    selected_algorithm = request.args.get("algorithm", "")
    selected_numElements = request.args.get("numElements", "")
    selected_dataType = request.args.get("dataType", "")
    selected_sorted = request.args.get("sorted", "")
    selected_type = request.args.get("type", "")

    # Dynamic filter
    query = {}
    if selected_algorithm: query["algorithm"] = selected_algorithm
    if selected_numElements: query["numElements"] = int(selected_numElements)
    if selected_dataType: query["dataType"] = selected_dataType
    if selected_sorted: query["sorted"] = selected_sorted.lower() in ["true", "1"]
    if selected_type: query["type"] = selected_type

    # Retrieve all documents matching the filter
    docs = list(db.Dataset.find(query).sort("_id", -1))
    count = len(docs)

    # Pagination
    per_page = 20
    page = int(request.args.get("page", 1))
    total_pages = (count + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    docs_paginated = docs[start:end]

    # Unique values for filters
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


# API ejemplo para gráficas
@app.route("/api/pred_vs_actual")
def api_pred_vs_actual():
    db = get_db()
    docs = list(db.Dataset.find().limit(10000))
    for d in docs:
        d["_id"] = str(d["_id"])
    return jsonify(docs)

if __name__ == "__main__":
    app.run(debug=True)
