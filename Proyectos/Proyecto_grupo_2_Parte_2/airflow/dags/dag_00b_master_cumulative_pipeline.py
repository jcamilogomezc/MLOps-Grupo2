from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import functions from modular DAGs
from dag_01_download_data import download_diabetes_data
from dag_02_split_data import split_dataset
from dag_03b_store_batched_raw_data import (
    create_batch_tables,
    create_cumulative_views,
    store_validation_test_data
)
from dag_05b_train_cumulative_batches import train_cumulative_batch
from dag_06b_select_best_cumulative_model import (
    aggregate_cumulative_results,
    select_best_model,
    evaluate_on_full_validation
)
from dag_07_publish_to_production import publish_model_to_production


# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Create the master DAG
with DAG(
    'dag_00b_master_cumulative_pipeline',
    default_args=default_args,
    description='Master pipeline with dynamic parallel cumulative batch training',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['diabetes', 'ml-pipeline', 'master', 'cumulative-training', 'dynamic'],
) as dag:

    # ===================================================================
    # PHASE 1: Data Acquisition and Preparation
    # ===================================================================

    download_task = PythonOperator(
        task_id='download_diabetes_dataset',
        python_callable=download_diabetes_data,
        doc_md="""
        Downloads the diabetes readmission dataset from Google Drive.
        Expected: 101,767 records with 50+ features.
        """
    )

    split_task = PythonOperator(
        task_id='split_train_val_test',
        python_callable=split_dataset,
        doc_md="""
        Splits dataset into:
        - Training: 70% (~71,237 records)
        - Validation: 15% (~15,265 records)
        - Test: 15% (~15,265 records)
        """
    )

    # ===================================================================
    # PHASE 2: Batch Storage in PostgreSQL
    # ===================================================================

    create_batches = PythonOperator(
        task_id='create_batch_tables',
        python_callable=create_batch_tables,
        doc_md="""
        Stores training data in PostgreSQL batches:
        - 7 batch tables (15k records each, last batch ~11.7k)
        - Shuffled data for random distribution
        """
    )

    create_views = PythonOperator(
        task_id='create_cumulative_views',
        python_callable=create_cumulative_views,
        doc_md="""
        Creates cumulative views for progressive training:
        - train_cumulative_0: 15k records
        - train_cumulative_1: 30k records
        - train_cumulative_2: 45k records
        - train_cumulative_3: 60k records
        - train_cumulative_4: 75k records
        - train_cumulative_5: 90k records
        - train_cumulative_6: ~102k records (all data)
        """
    )

    store_val_test = PythonOperator(
        task_id='store_validation_test',
        python_callable=store_validation_test_data,
        doc_md="""
        Stores validation and test datasets in PostgreSQL:
        - validation_raw table
        - test_raw table
        """
    )

    # ===================================================================
    # PHASE 3: Parallel Cumulative Training
    # ===================================================================

    def generate_training_tasks(**context):
        """
        Dynamically generate training task configurations based on cumulative views.
        This function pulls the cumulative view information from XCom and creates
        the appropriate task configurations.
        """
        ti = context['ti']

        # Pull cumulative view information from the create_cumulative_views task
        cumulative_views = ti.xcom_pull(
            task_ids='create_cumulative_views',
            key='cumulative_views'
        )

        if not cumulative_views:
            raise ValueError("No cumulative views information found in XCom!")

        print(f"\n{'='*60}")
        print(f"Generating Training Configurations")
        print(f"{'='*60}")
        print(f"Number of cumulative datasets: {len(cumulative_views)}")

        configs = []
        for view_info in cumulative_views:
            cumulative_idx = view_info['num_batches'] - 1  # 0-indexed
            total_records = view_info['total_records']

            configs.append({
                'cumulative_idx': cumulative_idx,
                'expected_records': total_records,
                'view_name': view_info['view_name']
            })

            print(f"  Config {cumulative_idx}: {total_records} records ({view_info['view_name']})")

        print(f"{'='*60}\n")

        # Push configurations to XCom for downstream tasks
        ti.xcom_push(key='training_configs', value=configs)

        return configs

    # Task to generate dynamic configurations
    generate_configs_task = PythonOperator(
        task_id='generate_training_configs',
        python_callable=generate_training_tasks,
        doc_md="""
        Dynamically generates training task configurations based on actual data size.
        Pulls cumulative view information from XCom and creates training configs.
        """
    )

    def should_train_cumulative(**context):
        """
        Determines if this cumulative index should be trained based on actual data.
        Returns True if this index exists in the configurations.
        """
        ti = context['ti']
        cumulative_idx = context['params']['cumulative_idx']

        # Pull the training configurations
        configs = ti.xcom_pull(
            task_ids='generate_training_configs',
            key='training_configs'
        )

        # Check if this cumulative index exists
        config = next((c for c in configs if c['cumulative_idx'] == cumulative_idx), None)

        if config:
            print(f"✓ Cumulative {cumulative_idx} will be trained ({config['expected_records']} records)")
            return f'train_cumulative_{cumulative_idx}'
        else:
            print(f"✗ Cumulative {cumulative_idx} skipped (no data)")
            return f'skip_cumulative_{cumulative_idx}'

    def create_training_task_wrapper(cumulative_idx):
        """
        Wrapper function to validate expected records before training.
        """
        def training_with_validation(**context):
            ti = context['ti']

            # Pull the training configurations
            configs = ti.xcom_pull(
                task_ids='generate_training_configs',
                key='training_configs'
            )

            # Find the config for this cumulative index
            config = next((c for c in configs if c['cumulative_idx'] == cumulative_idx), None)

            if not config:
                # This should not happen if branching works correctly
                print(f"⚠ Warning: No configuration found for cumulative_idx={cumulative_idx}, skipping...")
                return None

            print(f"\n{'='*60}")
            print(f"Training Configuration for Cumulative {cumulative_idx}")
            print(f"{'='*60}")
            print(f"  View: {config['view_name']}")
            print(f"  Expected Records: {config['expected_records']}")
            print(f"{'='*60}\n")

            # Call the actual training function
            return train_cumulative_batch(cumulative_idx=cumulative_idx, **context)

        return training_with_validation

    # Create training tasks dynamically
    # We'll create tasks for indices 0-9 (assuming max 10 batches)
    # Only tasks with actual data will execute
    train_tasks = []

    # Maximum expected batches (conservative estimate based on 15k batch size)
    # 101,767 / 15,000 ≈ 7 batches, but we'll support up to 10 for flexibility
    MAX_BATCHES = 10

    for cumulative_idx in range(MAX_BATCHES):
        task = PythonOperator(
            task_id=f'train_cumulative_{cumulative_idx}',
            python_callable=create_training_task_wrapper(cumulative_idx),
            trigger_rule='none_failed',  # Run even if some upstream tasks are skipped
            doc_md=f"""
            Train all 4 model types on cumulative dataset {cumulative_idx}.
            Actual record count determined dynamically at runtime.
            Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
            """
        )
        train_tasks.append(task)

    # ===================================================================
    # PHASE 4: Model Selection and Evaluation
    # ===================================================================

    aggregate_results_task = PythonOperator(
        task_id='aggregate_results',
        python_callable=aggregate_cumulative_results,
        doc_md="""
        Aggregates results from all 28 trained models:
        - Pulls results from XCom
        - Creates comparison DataFrame
        - Sorts by validation F1 score
        """
    )

    select_best_task = PythonOperator(
        task_id='select_best_model',
        python_callable=select_best_model,
        doc_md="""
        Selects best model across all 28 trained models:
        - Primary criterion: Highest F1 score
        - Secondary criterion: ROC-AUC as tiebreaker
        - Provides performance analysis across dataset sizes and model types
        """
    )

    evaluate_full_val_task = PythonOperator(
        task_id='evaluate_on_full_validation',
        python_callable=evaluate_on_full_validation,
        doc_md="""
        Evaluates best model on full validation set:
        - Loads model from MLflow
        - Evaluates on complete validation dataset from PostgreSQL
        - Logs final metrics to MLflow
        """
    )

    # ===================================================================
    # PHASE 5: Production Deployment
    # ===================================================================

    publish_task = PythonOperator(
        task_id='publish_best_model_production',
        python_callable=publish_model_to_production,
        doc_md="""
        Publishes best model to production:
        - Registers model in MLflow Model Registry
        - Transitions to Production stage
        - Archives previous production models
        - Completes deployment workflow
        """
    )

    # ===================================================================
    # Pipeline Dependencies
    # ===================================================================

    # Phase 1: Data acquisition and preparation
    download_task >> split_task

    # Phase 2: Batch storage in PostgreSQL (parallel storage for validation/test)
    split_task >> create_batches
    create_batches >> create_views
    create_batches >> store_val_test

    # Phase 3: Generate training configurations and parallel cumulative training
    # First generate configs based on actual data, then train
    store_val_test >> generate_configs_task
    create_views >> generate_configs_task

    # All training tasks depend on configuration generation
    for train_task in train_tasks:
        generate_configs_task >> train_task

    # Phase 4: Model selection and evaluation
    # Aggregate results depends on all training tasks completing
    for train_task in train_tasks:
        train_task >> aggregate_results_task

    aggregate_results_task >> select_best_task >> evaluate_full_val_task

    # Phase 5: Production deployment
    evaluate_full_val_task >> publish_task