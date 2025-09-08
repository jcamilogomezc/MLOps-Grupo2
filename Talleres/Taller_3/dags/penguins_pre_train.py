from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator

import palmerpenguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
import joblib

# =========================
# 1) CARGA RAW A MySQL
# =========================
def load_penguins_data_to_raw():
    penguins = palmerpenguins.load_penguins()

    NA_SET = {"", "na", "n/a", "null", "none", "nan"}

    def to_float(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in NA_SET:
            return None
        s = s.replace(",", ".")  # soportar "43,2"
        try:
            return float(s)
        except ValueError:
            return None

    def to_text(x):
        if x is None:
            return None
        s = str(x).strip()
        return s if s and s.lower() not in NA_SET else None

    hook = MySqlHook(mysql_conn_id="mymysql")

    rows = []
    skipped = 0
    total = 0

    for _, row in penguins.iterrows():
        total += 1
        species  = to_text(row.get("species"))
        island   = to_text(row.get("island"))
        bill_len = to_float(row.get("bill_length_mm"))
        bill_dep = to_float(row.get("bill_depth_mm"))
        flip_len = to_float(row.get("flipper_length_mm"))
        body_g   = to_float(row.get("body_mass_g"))
        sex      = to_text(row.get("sex"))

        if all(v is None for v in (bill_len, bill_dep, flip_len, body_g)):
            skipped += 1
            continue

        rows.append((species, island, bill_len, bill_dep, flip_len, body_g, sex))

    if rows:
        hook.insert_rows(
            table="penguins_raw",
            rows=rows,
            target_fields=[
                "species","island","bill_length_mm","bill_depth_mm",
                "flipper_length_mm","body_mass_g","sex"
            ],
        )

    print(f"[LOAD] Total={total} | Insertadas={len(rows)} | Omitidas={skipped}")

# =========================
# 2) PREPROCESAR -> TABLE CLEAN
# =========================
def preprocess_task():
    Path("/opt/airflow/dags/artifacts").mkdir(parents=True, exist_ok=True)

    sql_hook = MySqlHook(mysql_conn_id="mymysql")

    # Leer RAW
    with sql_hook.get_conn() as conn:
        df = pd.read_sql("""
            SELECT species,island,bill_length_mm,bill_depth_mm,
                   flipper_length_mm,body_mass_g,sex
            FROM penguins_raw;
        """, conn)

    # Limpieza básica
    df = df.dropna(subset=['species']).copy()

    # Identificar tipos de columnas
    numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

    # características categóricas = ['sex']
    categorical_features = ['island', 'sex']

    # Preprocesamiento para datos numéricos
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Imputación de valores faltantes con la mediana
        ('scaler', StandardScaler()) # Escalado de características numéricas
    ])

    # Preprocesamiento para datos categóricos
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Imputación de valores faltantes con la moda
        ('encoder', OneHotEncoder(handle_unknown='ignore')) # Codificación one-hot para variables categóricas
    ])

    # Combinar preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Guardar preprocesador en artifacts
    joblib.dump(preprocessor, "/opt/airflow/dags/artifacts/preprocessor.joblib")
    print("[PREPROCESS] Preprocessor saved to /opt/airflow/dags/artifacts/preprocessor.joblib")

    # Guardar en penguins_clean (sobrescribimos)
    rows = list(df[['species','island','bill_length_mm','bill_depth_mm',
                    'flipper_length_mm','body_mass_g','sex']].itertuples(index=False, name=None))

    # Limpiar tabla y escribir
    with sql_hook.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE penguins_clean;")
        conn.commit()

    sql_hook.insert_rows(
        table='penguins_clean',
        rows=rows,
        target_fields=[
            'species','island','bill_length_mm','bill_depth_mm',
            'flipper_length_mm','body_mass_g','sex'
        ]
    )
    print(f"[PREPROCESS] Filas limpias insertadas: {len(rows)}")

# =========================
# 3) CHECK COUNTS
# =========================
def count_tables():
    hook = MySqlHook(mysql_conn_id="mymysql")
    raw  = hook.get_first("SELECT COUNT(*) FROM penguins_raw")[0]
    clean= hook.get_first("SELECT COUNT(*) FROM penguins_clean")[0]
    print(f"[CHECK] RAW={raw} | CLEAN={clean}")

# =========================
# 4) TRAIN 4 MODELOS
# =========================
def train_models_task():
    # Rutas de salida (persisten en el contenedor)
    Path("/opt/airflow/dags/models").mkdir(parents=True, exist_ok=True)

    # Importar preprocesador
    preprocessor = joblib.load("/opt/airflow/dags/artifacts/preprocessor.joblib")
    print("[TRAIN] Preprocessor loaded from /opt/airflow/dags/artifacts/preprocessor.joblib")

    sql_hook = MySqlHook(mysql_conn_id="mymysql")
    with sql_hook.get_conn() as conn:
        df = pd.read_sql("SELECT * FROM penguins_clean;", conn)
    
    # Separar características (X) y variable objetivo (y)
    penguins = df.copy()
    X = penguins.drop(columns='species')
    y = penguins['species']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Definir los modelos a entrenar
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=42),
        "svc": SVC(kernel='rbf', probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "catboost": CatBoostClassifier(random_state=42, verbose=False)
    }

    metrics = []
    best_name, best_f1 = None, -1.0

    for name, model in models.items():
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Entrenar el modelo
        clf.fit(X_train, y_train)
        # Realizar predicciones
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")

        # Guardar modelo
        model_path = f"/opt/airflow/dags/models/{name}.joblib"
        joblib.dump(clf, model_path)

        print(f"[TRAIN] {name} -> acc={acc:.4f} | f1_macro={f1:.4f} | saved={model_path}")
        metrics.append({"model": name, "accuracy": acc, "f1_macro": f1, "path": model_path})

        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    # Guardar métricas
    mdf = pd.DataFrame(metrics).sort_values(by="f1_macro", ascending=False)
    mdf.to_csv("/opt/airflow/dags/artifacts/metrics.csv", index=False)
    print("[METRICS]\n", mdf)

    # Copia del mejor como best_model.joblib
    best_src = f"/opt/airflow/dags/models/{best_name}.joblib"
    best_dst = f"/opt/airflow/dags/models/best_model_{best_name}.joblib"
    joblib.dump(joblib.load(best_src), best_dst)
    print(f"[BEST] {best_name} con f1_macro={best_f1:.4f} -> {best_dst}")

# =========================
# DAG
# =========================
with DAG(
    dag_id="penguins_mysql_ml_train_models",
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["mysql","etl","mlops_puj","training"]
) as dag:

    # Esquema explícito para clean (incluye columnas *_numeric)
    create_tables = MySqlOperator(
        task_id="create_tables",
        mysql_conn_id="mymysql",
        sql="""
        CREATE TABLE IF NOT EXISTS penguins_raw (
          id INT AUTO_INCREMENT PRIMARY KEY,
          species VARCHAR(64), island VARCHAR(64),
          bill_length_mm DECIMAL(10,4), bill_depth_mm DECIMAL(10,4),
          flipper_length_mm DECIMAL(10,4), body_mass_g DECIMAL(10,4),
          sex VARCHAR(16), inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS penguins_clean (
          id INT AUTO_INCREMENT PRIMARY KEY,
          species VARCHAR(64), island VARCHAR(64),
          bill_length_mm DECIMAL(10,4), bill_depth_mm DECIMAL(10,4),
          flipper_length_mm DECIMAL(10,4), body_mass_g DECIMAL(10,4),
          sex VARCHAR(16), inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    truncate_raw = MySqlOperator(
        task_id="truncate_raw",
        mysql_conn_id="mymysql",
        sql="TRUNCATE TABLE penguins_raw;"
    )

    load_raw = PythonOperator(
        task_id="load_penguins_data_to_raw",
        python_callable=load_penguins_data_to_raw
    )

    preprocess = PythonOperator(
        task_id="preprocess_to_clean",
        python_callable=preprocess_task
    )

    check_counts = PythonOperator(
        task_id="check_counts",
        python_callable=count_tables
    )

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=train_models_task
    )

    create_tables >> truncate_raw >> load_raw >> preprocess >> check_counts >> train_models