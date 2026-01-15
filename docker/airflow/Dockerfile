FROM apache/airflow:2.8.3-python3.10

USER root
RUN apt-get update && apt-get install -y build-essential && apt-get clean
USER airflow

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    mlflow==2.14.3 \
    pycaret==3.3.2 \
    pandas==1.5.3 \
    scikit-learn==1.4.2 \
    scipy==1.11.4
