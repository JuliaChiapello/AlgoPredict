import os
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

def train():
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DATABASE_NAME")]
    collection = db.Dataset
    df = pd.DataFrame(collection.find())
    df = df.drop(columns=["_id"], errors="ignore")

    # Dividir en reales y teoricos segun una columna 'type' que debe tener valores "real" o "theoretical"
    df = df.dropna()
    df = df[df["timePredicted"] > 0]

    Q1 = df["timePredicted"].quantile(0.25)
    Q3 = df["timePredicted"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["timePredicted"] >= Q1 - 1.5 * IQR) & (df["timePredicted"] <= Q3 + 1.5 * IQR)]

    df_real = df[df['type'] == 'real'].copy()
    df_theoretical = df[df['type'] == 'theoretical'].copy()

    cat_cols = ["algorithm", "dataType", "sorted"]
    num_cols = ["numElements"]

    # OneHotEncoder en comun
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[cat_cols])

    # Escalador para datos teoricos
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Aplicar log1p al target real
    df_real["timePredicted"] = np.log1p(df_real["timePredicted"])

    def prepare_data(sub_df, use_scaler=False):
        X_cat = encoder.transform(sub_df[cat_cols])
        X_num = sub_df[num_cols].values
        X = np.hstack((X_cat, X_num))
        y = sub_df["timePredicted"].values
        if use_scaler:
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        return train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo para tiempos reales: HistGradientBoosting
    X_train_r, X_test_r, y_train_r, y_test_r = prepare_data(df_real)
    param_grid_hgbr = {
        'max_iter': [100, 300],
        'max_leaf_nodes': [31, 63],
        'learning_rate': [0.1, 0.01],
        'max_depth': [None, 10],
        'l2_regularization': [0.0, 0.1]
    }
    model_real = GridSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_grid=param_grid_hgbr,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    print("Entrenando modelo REAL con HistGradientBoostingRegressor...")
    model_real.fit(X_train_r, y_train_r)
    print("Mejores params reales:", model_real.best_params_)

    y_pred_r = model_real.predict(X_test_r)
    y_pred_r_exp = np.expm1(y_pred_r)
    y_test_r_exp = np.expm1(y_test_r)
    print("RMSE Real:", round(np.sqrt(mean_squared_error(y_test_r_exp, y_pred_r_exp)), 4))
    print("R² Real:", round(r2_score(y_test_r_exp, y_pred_r_exp), 4))

    # Modelo para tiempos teoricos: MLPRegressor
    X_train_t, X_test_t, y_train_t, y_test_t = prepare_data(df_theoretical, use_scaler=True)
    param_grid_mlp = {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'max_iter': [2000],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001]
    }
    model_theoretical = GridSearchCV(
        MLPRegressor(random_state=42),
        param_grid=param_grid_mlp,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    print("Entrenando modelo TEORICO con MLPRegressor...")
    model_theoretical.fit(X_train_t, y_train_t)
    print("Mejores params teoricos:", model_theoretical.best_params_)

    y_pred_t = model_theoretical.predict(X_test_t)
    y_pred_t_inv = scaler_y.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()
    y_test_t_inv = scaler_y.inverse_transform(y_test_t.reshape(-1, 1)).ravel()
    print("RMSE Teorico:", round(np.sqrt(mean_squared_error(y_test_t_inv, y_pred_t_inv)), 4))
    print("R² Teorico:", round(r2_score(y_test_t_inv, y_pred_t_inv), 4))

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


