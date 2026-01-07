"""
Este módulo contiene la lógica completa de entrenamiento de modelos
de Machine Learning para el proyecto AlgoPredict.

Su objetivo es entrenar un MODELO DUAL que permita:
1) Predecir tiempos de ejecución REALES (medidos empíricamente)
2) Aproximar tiempos de ejecución TEÓRICOS (complejidad algorítmica)

El archivo se encarga de:
- Cargar datos desde MongoDB
- Limpiar y preparar el dataset
- Entrenar modelos separados para datos reales y teóricos
- Evaluar métricas
- Persistir el modelo entrenado para inferencia posterior
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Se silencian warnings para evitar ruido durante el entrenamiento
warnings.filterwarnings("ignore")

# ==================================================
# UTILIDAD: ELIMINACIÓN DE OUTLIERS (IQR)
# ==================================================

def remove_outliers_iqr(df, target_col):
    """
    Elimina outliers utilizando el método del rango intercuartílico (IQR).

    Este paso es clave porque:
    - Los tiempos de ejecución pueden contener picos extremos
    - Dichos valores afectan negativamente el entrenamiento
    - Se mejora la estabilidad y generalización del modelo

    Parámetros:
    - df: DataFrame original
    - target_col: columna objetivo (timePredicted)

    Retorna:
    - DataFrame filtrado sin outliers extremos
    """
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    return df[
        (df[target_col] >= Q1 - 1.5 * IQR) &
        (df[target_col] <= Q3 + 1.5 * IQR)
    ]

# ==================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ==================================================

def train():
    """
    Función principal que ejecuta todo el pipeline de entrenamiento.

    Este método:
    - Carga datos desde MongoDB
    - Preprocesa features y target
    - Entrena dos modelos independientes
    - Evalúa métricas
    - Guarda el modelo entrenado en disco
    """

    # --------------------------------------------------
    # CARGA DE VARIABLES DE ENTORNO
    # --------------------------------------------------

    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DATABASE_NAME")]
    collection = db.Dataset

    # --------------------------------------------------
    # CARGA Y LIMPIEZA DEL DATASET
    # --------------------------------------------------

    df = pd.DataFrame(collection.find())

    # Eliminamos el _id de Mongo y valores nulos
    df = df.drop(columns=["_id"], errors="ignore")
    df = df.dropna()

    # Se descartan tiempos inválidos o negativos
    df = df[df["timePredicted"] > 0]

    # --------------------------------------------------
    # SEPARACIÓN DE DATOS REALES Y TEÓRICOS
    # --------------------------------------------------
    
    df_real = df[df["type"] == "real"].copy()
    df_theoretical = df[df["type"] == "theoretical"].copy()

    # --------------------------------------------------
    # ELIMINACIÓN DE OUTLIERS (POR SEPARADO)
    # --------------------------------------------------

    df_real = remove_outliers_iqr(df_real, "timePredicted")
    df_theoretical = remove_outliers_iqr(df_theoretical, "timePredicted")

    # --------------------------------------------------
    # DEFINICIÓN DE FEATURES
    # --------------------------------------------------

    # Categóricas: describen el algoritmo y el tipo de dato
    cat_cols = ["algorithm", "dataType", "sorted"]
    # Numéricas: tamaño del input
    num_cols = ["numElements"]

    # --------------------------------------------------
    # ENCODER CATEGÓRICO
    # --------------------------------------------------

    # Se entrena UNA SOLA VEZ con todo el dataset
    # para garantizar consistencia entre modelos
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )
    encoder.fit(df[cat_cols])

    # --------------------------------------------------
    # ESCALADORES (usados solo en modelo teórico)
    # --------------------------------------------------

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # --------------------------------------------------
    # TRANSFORMACIÓN LOGARÍTMICA (SOLO DATOS REALES)
    # --------------------------------------------------

    # Reduce la asimetría de los tiempos reales
    # y mejora la estabilidad del modelo
    df_real["timePredicted"] = np.log1p(df_real["timePredicted"])

    # ==================================================
    # FUNCIÓN INTERNA DE PREPARACIÓN DE DATOS
    # ==================================================

    def prepare_data(sub_df, scale=False):
        """
        Prepara datos para entrenamiento y test.

        - Aplica OneHotEncoding a variables categóricas
        - Concatena variables numéricas
        - Aplica escalado opcional
        - Divide en train / test

        Parámetro:
        - scale: indica si se debe escalar X e y

        Retorna:
        - X_train, X_test, y_train, y_test
        """
        X_cat = encoder.transform(sub_df[cat_cols])
        X_num = sub_df[num_cols].values
        X = np.hstack((X_cat, X_num))
        y = sub_df["timePredicted"].values

        if scale:
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        return train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # ==================================================
    # MODELO 1: TIEMPOS REALES
    # ==================================================
    
    print("\nEntrenando modelo REAL (HistGradientBoosting)...")

    X_train_r, X_test_r, y_train_r, y_test_r = prepare_data(df_real)

    # Se utiliza Gradient Boosting por su capacidad de:
    # - modelar relaciones no lineales
    # - manejar ruido
    # - ofrecer alta precisión sin sobreajuste severo
    param_grid_real = {
        "max_iter": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [31, 63],
        "l2_regularization": [0.0, 0.1],
    }

    model_real = GridSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_grid=param_grid_real,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    model_real.fit(X_train_r, y_train_r)

    # Inversa del log-transform para evaluación real
    y_pred_r = model_real.predict(X_test_r)
    y_pred_r = np.expm1(y_pred_r)
    y_test_r = np.expm1(y_test_r)

    print("Mejores parámetros REAL:", model_real.best_params_)
    print("RMSE Real:", round(np.sqrt(mean_squared_error(y_test_r, y_pred_r)), 4))
    print("R² Real:", round(r2_score(y_test_r, y_pred_r), 4))

    # ==================================================
    # MODELO 2: TIEMPOS TEÓRICOS
    # ==================================================

    print("\nEntrenando modelo TEÓRICO (Polynomial Regression + Ridge)...")

    X_train_t, X_test_t, y_train_t, y_test_t = prepare_data(
        df_theoretical,
        scale=True
    )

    # Se utiliza regresión polinómica para:
    # - aproximar funciones O(n), O(n log n), O(n²)
    # Ridge evita sobreajuste y estabiliza coeficientes
    pipeline_theoretical = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("ridge", Ridge())
    ])

    param_grid_theoretical = {
        "poly__degree": [1, 2, 3],
        "ridge__alpha": [0.01, 0.1, 1.0, 10.0]
    }

    model_theoretical = GridSearchCV(
        pipeline_theoretical,
        param_grid=param_grid_theoretical,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    model_theoretical.fit(X_train_t, y_train_t)

    # Inversa del escalado
    y_pred_t = model_theoretical.predict(X_test_t)
    y_pred_t = scaler_y.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()
    y_test_t = scaler_y.inverse_transform(y_test_t.reshape(-1, 1)).ravel()

    print("Mejores parámetros TEÓRICO:", model_theoretical.best_params_)
    print("RMSE Teórico:", round(np.sqrt(mean_squared_error(y_test_t, y_pred_t)), 4))
    print("R² Teórico:", round(r2_score(y_test_t, y_pred_t), 4))

    # ==================================================
    # GUARDADO DEL MODELO DUAL
    # ==================================================

    with open("dualModelTrain.pkl", "wb") as f:
        pickle.dump({
            "model_real": model_real.best_estimator_,
            "model_theoretical": model_theoretical.best_estimator_,
            "encoder": encoder,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "cat_columns": cat_cols,
            "num_columns": num_cols
        }, f)

    print("\nModelo dual guardado correctamente en dualModelTrain.pkl")
