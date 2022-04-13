from airflow import DAG
from airflow.models import BaseOperator
from airflow.operators.bash import BbashOperator
from airflow.operators.dummy_operator import DummyOperator
from  airflow.operators.postgres_operator import PostgresOperator
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.S3_hook import S3Hook
from airflow.models import Variable
from plugins/hooks.posgres_hook import PosgresHook

from datetime import datetime
from datetime import timedelts
import logging

log = logging.gettogger(__name__)

# Step 1: Set up the main configuration of the DAG

