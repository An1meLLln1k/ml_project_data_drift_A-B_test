# ML-процесс от мониторинга drift до A/B теста

Проект показывает полный цикл MLOps на простом примере (Iris):  
проверка data drift, автоматическое переобучение, регистрация модели в MLflow и A/B роутинг запросов через Flask.

## Что реализовано
- Airflow DAG: генерация данных, расчёт drift, ветвление по порогу, запуск переобучения при необходимости.
- Drift detection: проверка с порогом (если drift высокий — запускаем retraining).
- Retrain: обучение модели через PyCaret AutoML.
- MLflow: логирование экспериментов и Model Registry (версии + стадии Production/Staging).
- A/B Router (Flask): раздаёт трафик между Production и Staging и логирует все запросы в CSV.

## Архитектура (сервисы)
- Airflow (UI: http://localhost:8080)
- MLflow (UI: http://localhost:5000)
- A/B Router API (Flask: http://localhost:8000)

## Структура репозитория (пример)
```text
ml_project/
  dags/                     # Airflow DAG
  docker/
    airflow/                # Dockerfile/настройки Airflow
    flask/                  # Dockerfile + app.py (A/B router)
    mlflow.Dockerfile       # образ MLflow (если используется)
  src/                      # код retrain/utility (если вынесено сюда)
  data/                     # данные (если нужно)
  reports/                  # презентация и скриншоты для защиты
  docker-compose.yml
  .env.example
  README.md
