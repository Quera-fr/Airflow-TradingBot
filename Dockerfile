FROM apache/airflow:3.0.1

WORKDIR /opt/airflow

COPY requirements.txt .

COPY memory.json .

RUN pip install -r requirements.txt

COPY plugins/ plugins/

COPY dags/ dags/

CMD ["airflow", "standalone"]