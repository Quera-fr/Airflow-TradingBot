docker build -t airflow-server .

docker run -p 8060:8080 \
 -v "C:\Users\Quera\Desktop\Airflow:/opt/airflow" \
 -it airflow-server