from datetime import datetime
from airflow import DAG
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator


# =========================
# REVIEW TABLES FUNCTION
# =========================
def review_tables():
    """
    Check the dimensions (row count) of both covertype tables
    and log the results for monitoring purposes.
    """
    hook = MySqlHook(mysql_conn_id="mysql_trn")
    
    tables_to_check = ["covertype_raw", "covertype_clean"]
    results = {}
    
    with hook.get_conn() as conn:
        cursor = conn.cursor()
        
        for table in tables_to_check:
            try:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) as row_count FROM {table};")
                row_count = cursor.fetchone()[0]
                
                # Get column information
                cursor.execute(f"""
                    SELECT COUNT(*) as column_count 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = '{table}' 
                    AND TABLE_SCHEMA = DATABASE();
                """)
                column_count = cursor.fetchone()[0]
                
                # Store results
                results[table] = {
                    'rows': row_count,
                    'columns': column_count
                }
                
                print(f"[REVIEW] Table {table}: {row_count:,} rows, {column_count} columns")
                
            except Exception as e:
                print(f"[REVIEW] Error checking table {table}: {str(e)}")
                results[table] = {
                    'rows': 'ERROR',
                    'columns': 'ERROR',
                    'error': str(e)
                }
    
    # Summary log
    print(f"[REVIEW] ============ TABLE REVIEW SUMMARY ============")
    for table, stats in results.items():
        if 'error' not in stats:
            print(f"[REVIEW] {table.upper()}: {stats['rows']:,} rows × {stats['columns']} columns")
        else:
            print(f"[REVIEW] {table.upper()}: ERROR - {stats['error']}")
    print(f"[REVIEW] ===============================================")
    
    return results


# =========================
# DAG DEFINITION
# =========================
with DAG(
    dag_id="covertype_truncate_tables",
    start_date=datetime(2024,1,1),
    schedule=None,  # Updated from schedule_interval
    catchup=False,
    tags=["mysql","etl","mlops_puj","maintenance"]
) as dag:

    # Review tables before truncation
    review_tables_before = PythonOperator(
        task_id="review_tables_before",
        python_callable=review_tables
    )

    # Truncate raw table
    truncate_raw = MySqlOperator(
        task_id="truncate_raw",
        mysql_conn_id="mysql_trn",  # Fixed connection ID
        sql="TRUNCATE TABLE covertype_raw;"
    )

    # Truncate clean table  
    truncate_clean = MySqlOperator(
        task_id="truncate_clean",  # Fixed duplicate task_id
        mysql_conn_id="mysql_trn",  # Fixed connection ID
        sql="TRUNCATE TABLE covertype_clean;"
    )

    # Review tables after truncation
    review_tables_after = PythonOperator(
        task_id="review_tables_after", 
        python_callable=review_tables
    )

    # Task dependencies
    review_tables_before >> truncate_raw >> truncate_clean >> review_tables_after