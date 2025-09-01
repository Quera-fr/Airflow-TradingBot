import json
from utiles import *
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import time
from openai import OpenAI
from binance.client import Client
from airflow.hooks.base import BaseHook



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
        with open("/opt/airflow/dags/data/memory.json", "r") as f:
            memory = json.load(f)
    
        dag_id = ti.dag_id
        run_id = ti.run_id

        with open(f"/opt/airflow/dags/data/memory_{dag_id}_{run_id}.json", "w") as f:
            json.dump(memory, f, indent=2)

        print(f"Dag id: {dag_id}, Run id: {run_id}")

    def get_memory(ti):
        dag_id = ti.dag_id
        run_id = ti.run_id
        b = BaseHook.get_connection("binance_api")
        api_key = b.login
        api_secret = b.password
        
  
        client = Client(api_key, api_secret)
        binance_time_offset_ms(client)
        capital_initial = 38.35

        with open(f"/opt/airflow/dags/data/memory_{dag_id}_{run_id}.json", "r") as f:
            memory = json.load(f)
    
        capital, balances = get_capital_and_balances(client, INITIAL_CAPITAL_USDC=memory["capital"]["initial"])
        recent_decisions = memory.get("recent_decisions", [])
        performance = compute_performance(recent_decisions, capital["current"])
        
        memory["capital_history"].append({"timestamp": int(time.time()), "capital": capital["current"]})
        memory["capital"] = capital
        memory["capital"]["max_dd_7d"] = compute_max_dd_7d(memory.get("capital_history", []), capital["current"])
        memory["balances"] = balances
        memory["performance"] = performance

        usdc_free = float(balances.get("USDC", {}).get("amount", 0.0))
        max_reb = float(memory.get("constraints", {}).get("max_rebalance_pct", 0.30))
        risk_cap = float(memory["capital"].get("risk_per_trade_cap", capital["current"]*0.10))
        cap_max = min(capital["current"]*max_reb, risk_cap, usdc_free)

        memory["sizing"] = {
            "usdc_free": round(usdc_free, 4),
            "cap_max": round(cap_max, 4),
            "risk_cap": round(risk_cap, 4),
            "max_rebalance_pct": max_reb
        }

        snapshot_capital(memory, capital["current"])

        market_data = {
            "ETHUSDC": get_market_data(client, "ETHUSDC", balances=balances, memory=memory),
            "BTCUSDC": get_market_data(client, "BTCUSDC", balances=balances, memory=memory),
        }
        print("===================================")
        print("Market data fetched.")

        payload = {"market_data": market_data, "memory": memory}

        with open(f"/opt/airflow/dags/data/memory_{dag_id}_{run_id}.json", "w") as f:
            json.dump(memory, f, indent=2)
        ti.xcom_push(key="payload", value=payload)

    def decision_task(ti):
        openai_key = BaseHook.get_connection("openai_default").password

        client_openai = OpenAI(api_key=openai_key)
        
        payload = ti.xcom_pull(key="payload", task_ids="get_memory_task")

        decision = get_decision(client_openai, payload)
        print("===================================")
        print("Décision:", decision)
        ti.xcom_push(key="decision", value=decision)

    def action_task(ti):
        dag_id = ti.dag_id
        run_id = ti.run_id
        api_key = BaseHook.get_connection("binance_api").login
        api_secret = BaseHook.get_connection("binance_api").password

        with open(f"/opt/airflow/dags/data/memory_{dag_id}_{run_id}.json", "r") as f:
            memory = json.load(f)

        decision = ti.xcom_pull(key="decision", task_ids="decision_task")
        print("===================================")

        client = Client(api_key=api_key, api_secret=api_secret)

        memory = execute_trade(client, decision, memory)

        with open("/opt/airflow/dags/data/memory.json", "w") as f:
            json.dump(memory, f, indent=2)

        sleep_s = int(decision.get("time_sleep_s", 0))
        if sleep_s > 0:
            print(f"Time sleep (Décideur) : {sleep_s}s")
            time.sleep(sleep_s)

    connexion_task = PythonOperator(task_id="connexion_task", python_callable=connexion_task)

    get_memory_task = PythonOperator(task_id="get_memory_task", python_callable=get_memory)

    decision_task = PythonOperator(task_id="decision_task", python_callable=decision_task)

    action_task = PythonOperator(task_id="action_task", python_callable=action_task)

    connexion_task >> get_memory_task >> decision_task >> action_task