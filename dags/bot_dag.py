from utiles import *
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from openai import OpenAI
from binance.client import Client


with DAG(
    dag_id="bot_dag",
    start_date=datetime(2025, 1, 1),
    schedule=timedelta(hours=1),
    end_date=datetime(2025, 10, 10),
    catchup=False,
    max_active_tasks=1,
    max_active_runs=1,
    ) as dag:

    def connexion_task(ti):
        print("Start BotDag")

        ti.xcom_push(key="api_key", value=api_key)
        ti.xcom_push(key="api_secret", value=api_secret)
        ti.xcom_push(key="openai_key", value=openai_key)

        print("API Key, API Secret, OpenAI Key pushed to XCom")


    def get_memory(ti):
        api_key = ti.xcom_pull(key="api_key", task_ids="connexion_task")
        api_secret = ti.xcom_pull(key="api_secret", task_ids="connexion_task")
        openai_key = ti.xcom_pull(key="openai_key", task_ids="connexion_task")

        client_openai = OpenAI(api_key=openai_key)
        client = Client(api_key, api_secret)
        capital_initial = 38.35

        with open("memory.json", "r") as f:
            memory = json.load(f)

        capital, balances = get_capital_and_balances(client, INITIAL_CAPITAL_USDC=memory["capital"]["initial"])

        memory["capital"] = capital
        memory["balances"] = balances

        recent_decisions = memory.get("recent_decisions", [])

        # 5) Performance (historique + métriques actuelles)
        performance = compute_performance(recent_decisions, capital["current"])

        memory["performance"] = performance

        # 2) Market data
        market_data = {
            "ETHUSDC": get_market_data(client, "ETHUSDC"),
            "BTCUSDC": get_market_data(client, "BTCUSDC")
        }
        print("===================================")
        print("Market data fetched.")

        payload = {"market_data": market_data, "memory": memory}
        ti.xcom_push(key="memory", value=memory)
        ti.xcom_push(key="payload", value=payload)


    def decision_task(ti):
        api_key = ti.xcom_pull(key="api_key", task_ids="connexion_task")
        api_secret = ti.xcom_pull(key="api_secret", task_ids="connexion_task")
        openai_key = ti.xcom_pull(key="openai_key", task_ids="connexion_task")

        client_openai = OpenAI(api_key=openai_key)
        payload = ti.xcom_pull(key="payload", task_ids="get_memory_task")

        memory = ti.xcom_pull(key="memory", task_ids="get_memory_task")
        decision = get_decision(client_openai, payload)
        print("===================================")
        print("Décision:", decision)
        ti.xcom_push(key="decision", value=decision)

    def action_task(ti):
        api_key = ti.xcom_pull(key="api_key", task_ids="connexion_task")
        api_secret = ti.xcom_pull(key="api_secret", task_ids="connexion_task")
        memory = ti.xcom_pull(key="memory", task_ids="get_memory_task")
        decision = ti.xcom_pull(key="decision", task_ids="decision_task")
        print("===================================")

        client = Client(api_key=api_key, api_secret=api_secret)

        memory = execute_trade(client, decision, memory)

        with open("memory.json", "w") as f:
            json.dump(memory, f, indent=2)

    connexion_task = PythonOperator(
        task_id="connexion_task",
        python_callable=connexion_task
    )

    get_memory_task = PythonOperator(
        task_id="get_memory_task",
        python_callable=get_memory
    )

    decision_task = PythonOperator(
        task_id="decision_task",
        python_callable=decision_task
    )

    action_task = PythonOperator(
        task_id="action_task",
        python_callable=action_task
    )

    connexion_task >> get_memory_task >> decision_task >> action_task