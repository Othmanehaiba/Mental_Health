
from airflow import DAG
from airflow.operators import BashOperator
from datetime import datetime

with DAG(
    dag_id="mental_health_project",
    start_date=datetime(2026, 11, 17),
    schedule=None,
    catchup=False
) as dag:

    preprocessing = BashOperator(
        task_id="preprocessing",
        bash_command="cd /mnt/c/Projects/Mental_Health && python src/data_preprocessing.py"
    )

    training = BashOperator(
        task_id="training",
        bash_command="cd /mnt/c/Projects/Mental_Health && python src/train.py"
    )

    evaluation = BashOperator(
        task_id="evaluation",
        bash_command="cd /mnt/c/Projects/Mental_Health && python src/evaluate.py"
    )

    preprocessing >> training >> evaluation
