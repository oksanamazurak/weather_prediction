from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator

# Configuration
F1_THRESHOLD = 0.70

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='ML Training Pipeline with DVC, MLflow, and Model Registry',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['ml', 'training', 'dvc', 'mlflow'],
)
def ml_training_pipeline():
    
    start_pipeline = DummyOperator(
        task_id='start_pipeline'
    )

    install_deps_task = BashOperator(
        task_id='install_deps',
        bash_command=(
            'pip install --no-cache-dir '
            'pandas scikit-learn matplotlib seaborn mlflow joblib'
        ),
    )

    def check_data_availability():
        """Check if data is available and updated in DVC"""
        project_path = "/opt/airflow/project"
        os.chdir(project_path)
        
        # Check if DVC data exists
        if not os.path.exists("data/raw/WeatherAUS.csv"):
            raise FileNotFoundError("Source data not found")
            
        # Check if prepared data exists or needs update
        if not os.path.exists("data/prepared/train.csv"):
            return "prepare_data"
            
        # Check if data is recent (within last 24 hours)
        data_mtime = os.path.getmtime("data/prepared/train.csv")
        if (datetime.now().timestamp() - data_mtime) > 86400:  # 24 hours
            return "prepare_data"
            
        return "skip_preparation"

    check_data_task = BranchPythonOperator(
        task_id='check_data_availability',
        python_callable=check_data_availability,
    )

    skip_preparation = DummyOperator(
        task_id='skip_preparation'
    )

    prepare_data_task = BashOperator(
        task_id='prepare_data',
        bash_command='cd /opt/airflow/project && python src/prepare.py data/raw/WeatherAUS.csv data/prepared',
    )

    def train_model():
        """Train model using DVC repro"""
        project_path = "/opt/airflow/project"
        os.chdir(project_path)
        
        # Run training script
        os.system("python src/train.py --input_dir data/prepared --output_dir models")
        
        return "models/model.pkl"

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        trigger_rule='none_failed_min_one_success',
    )

    def evaluate_model(ti):
        """Evaluate model and decide if it should be registered"""
        project_path = "/opt/airflow/project"
        os.chdir(project_path)
        
        # Load metrics
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
            
        f1_score = metrics["f1"]
        accuracy = metrics["accuracy"]
        
        # Decision logic
        if f1_score >= F1_THRESHOLD:
            return "register_model"
        else:
            return "model_rejected"

    evaluate_model_task = BranchPythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    model_rejected = DummyOperator(
        task_id='model_rejected',
    )

    def register_model():
        """Register the model (simplified version)"""
        project_path = "/opt/airflow/project"
        os.chdir(project_path)
        
        # For now, just log that model was registered
        # In real implementation, this would register to MLflow
        print("Model successfully registered (simplified version)")
        
        # Load metrics to log
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
            
        print(f"Model metrics - Accuracy: {metrics['accuracy']}, F1: {metrics['f1']}")

    register_model_task = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
    )

    end_pipeline = DummyOperator(
        task_id='end_pipeline',
        trigger_rule='none_failed_or_skipped',
    )

    # Define the pipeline
    start_pipeline >> install_deps_task >> check_data_task
    check_data_task >> [prepare_data_task, skip_preparation]
    prepare_data_task >> train_model_task
    skip_preparation >> train_model_task
    train_model_task >> evaluate_model_task
    evaluate_model_task >> [register_model_task, model_rejected]
    register_model_task >> end_pipeline
    model_rejected >> end_pipeline


ml_training_pipeline_dag = ml_training_pipeline()
