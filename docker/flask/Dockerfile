FROM python:3.10-slim

WORKDIR /app

# системные зависимости (pycaret/склейка, cffi, иногда графы)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libffi-dev \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Важно: ставим pycaret и версии, которые просит модель (по твоим логам)
RUN pip install --no-cache-dir \
    flask==3.0.3 \
    mlflow==2.14.3 \
    pycaret==3.3.2 \
    pandas==1.5.3 \
    scipy==1.11.4 \
    pyarrow==14.0.2 \
    packaging==23.2 \
    jinja2==3.1.3 \
    importlib-metadata==6.11.0 \
    more-itertools==10.2.0 \
    pyyaml==6.0.1 \
    matplotlib==3.7.5 \
    cffi==1.16.0 \
    graphviz==0.20.1 \
    scikit-learn==1.4.2

CMD ["python", "app.py"]
