from datetime import datetime
import csv
from airflow import DAG
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator
from pathlib import Path

DATA_PATH = Path("/opt/airflow/dags/data/penguins.csv")

def load_csv_to_raw():
    """Carga CSV → penguins_raw, normalizando NAs y tipos."""
    NA_SET = {"", "na", "n/a", "null", "none", "nan"}
    def to_float(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in NA_SET:
            return None
        # Soportar coma decimal "43,2"
        s = s.replace(",", ".")
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

    with DATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total += 1
            species  = to_text(r.get("species"))
            island   = to_text(r.get("island"))
            bill_len = to_float(r.get("bill_length_mm"))
            bill_dep = to_float(r.get("bill_depth_mm"))
            flip_len = to_float(r.get("flipper_length_mm"))
            body_g   = to_float(r.get("body_mass_g"))
            sex      = to_text(r.get("sex"))

            # Opcional: descartar filas vacías (sin ninguna métrica)
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

    print(f"[LOAD] Total filas leídas: {total} | Insertadas: {len(rows)} | Omitidas: {skipped}")

def count_tables():
    hook = MySqlHook(mysql_conn_id="mymysql")
    raw  = hook.get_first("SELECT COUNT(*) FROM penguins_raw")[0]
    clean= hook.get_first("SELECT COUNT(*) FROM penguins_clean")[0]
    print(f"[CHECK] RAW={raw} | CLEAN={clean}")

with DAG(
    dag_id="penguins_mysql_etl",
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["mysql","etl","mlops_puj"]
) as dag:

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
        CREATE TABLE IF NOT EXISTS penguins_clean LIKE penguins_raw;
        """
    )

    truncate_raw = MySqlOperator(
        task_id="truncate_raw",
        mysql_conn_id="mymysql",
        sql="TRUNCATE TABLE penguins_raw;"
    )

    load_raw = PythonOperator(
        task_id="load_raw_from_csv",
        python_callable=load_csv_to_raw
    )

    preprocess = MySqlOperator(
        task_id="preprocess_clean",
        mysql_conn_id="mymysql",
        sql="""
        DELETE FROM penguins_clean;
        INSERT INTO penguins_clean (species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
        SELECT species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,
               CASE LOWER(TRIM(COALESCE(sex,'')))
                    WHEN 'male' THEN 'male'
                    WHEN 'female' THEN 'female'
                    ELSE NULL END AS sex_norm
        FROM penguins_raw
        WHERE bill_length_mm IS NOT NULL
          AND bill_depth_mm  IS NOT NULL
          AND flipper_length_mm IS NOT NULL
          AND body_mass_g    IS NOT NULL;
        """
    )

    check_counts = PythonOperator(
        task_id="check_counts",
        python_callable=count_tables
    )

    create_tables >> truncate_raw >> load_raw >> preprocess >> check_counts