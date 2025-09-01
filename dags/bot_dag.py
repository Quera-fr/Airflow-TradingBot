
from utiles import (
    binance_time_offset_ms,
    get_capital_and_balances,
    snapshot_capital,
    compute_max_dd_7d,
    compute_performance,
    get_market_data,
    get_decision,
    execute_trade,
    update_memory_with_decision
)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook

from datetime import datetime, timedelta
import json, time
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
        with open("/opt/airflow/dags/data/memory.json", "r") as f:
            memory = json.load(f)
    
        dag_id,run_id  = ti.dag_id, ti.run_id

        with open(f"/opt/airflow/dags/data/memory_{dag_id}_{run_id}.json", "w") as f:
            json.dump(memory, f, indent=2)

        print(f"Dag id: {dag_id}, Run id: {run_id}")

    def get_memory(ti):
        # --- 0) Contexte & client Binance ---
        dag_id, run_id = ti.dag_id, ti.run_id
        snap_path = f"/opt/airflow/dags/data/memory_{dag_id}_{run_id}.json"
        base_path = "/opt/airflow/dags/data/memory.json"

        b = BaseHook.get_connection("binance_api")
        client = Client(b.login, b.password)

        try:binance_time_offset_ms(client)  # diagnostic, n'influence pas la logique
        except Exception:pass

        # --- 1) Lecture mémoire robuste (snapshot > base > vide) ---
        with open(base_path, "r") as f: memory = json.load(f)

        memory.setdefault("capital", {})
        memory.setdefault("recent_decisions", [])
        memory.setdefault("capital_history", [])

        # --- 2) Capital & balances (initial depuis mémoire si présent) ---
        initial_cap = float(memory["capital"].get("initial", 5000.0))
        capital, balances = get_capital_and_balances(client, INITIAL_CAPITAL_USDC=initial_cap)

        # --- 3) MAJ état capital : snapshot -> DD 7j -> halt ---
        snapshot_capital(memory, capital["current"])  # ajoute le point du run (anti-doublon horaire)
        capital["max_dd_7d"] = compute_max_dd_7d(memory.get("capital_history", []), capital["current"])
        capital["halt_triggered"] = bool(capital["current"] < 0.5 * capital["initial"])

        # --- 4) Performance (à partir des décisions conservées) ---
        recent_decisions = memory.get("recent_decisions", [])
        performance = compute_performance(recent_decisions, capital["current"])

        # --- 5) État mémoire consolidé (sert au sizing paliers) ---
        memory["capital"] = capital
        memory["balances"] = balances
        memory["recent_decisions"] = recent_decisions
        memory["performance"] = performance

        # --- 6) Market data avec bloc sizing "paliers" (BUY borné par headroom & rebalance, SELL non capé) ---
        md_eth = get_market_data(client, "ETHUSDC", balances=balances, memory=memory)
        md_btc = get_market_data(client, "BTCUSDC", balances=balances, memory=memory)
        market_data = {"ETHUSDC": md_eth, "BTCUSDC": md_btc}

        # --- 7) Payload, persistance snapshot & XCom ---
        payload = {"market_data": market_data, "memory": memory}
        
        with open(snap_path, "w") as f:json.dump(memory, f, indent=2)
        ti.xcom_push(key="payload", value=payload)


    def decision_task(ti):
        """
        Appelle le LLM pour obtenir une décision, avec :
        - Court-circuit HALT -> HOLD canonique 12h
        - Try/except robuste + fallback HOLD en cas d'erreur LLM
        - Validation/normalisation stricte du JSON (asset/pair/size_pct/risk_check/confidence/time_sleep_s)
        - Troncature des champs verbeux pour éviter d'enfler XCom
        """

        # -------- 1) clés & payload amont --------
        openai_key = BaseHook.get_connection("openai_default").password
        client_openai = OpenAI(api_key=openai_key)
        payload = ti.xcom_pull(key="payload", task_ids="get_memory_task") or {}

        decision = get_decision(client_openai, payload)

        print("===================================")
        print(f"Décision : {decision}")
        ti.xcom_push(key="decision", value=decision)


    def action_task(ti):
        """
        Exécute la décision normalisée :
        - Récupère le snapshot mémoire du run (fallback vers memory.json si besoin)
        - Court-circuite en HOLD si HALT actif ou décision invalide
        - Ouvre un client Binance, aligne l'horloge, puis exécute via execute_trade (logique à paliers)
        - Persiste la mémoire mise à jour (snapshot du run + mémoire globale)
        - Ne DORS PAS dans un task Airflow : on log le time_sleep_s et on le laisse au scheduler
        """

        # ------- chemins & IO -------
        dag_id, run_id = ti.dag_id, ti.run_id
        snap_path = f"/opt/airflow/dags/data/memory_{dag_id}_{run_id}.json"

        with open(snap_path, "r") as f: memory = json.load(f)

        decision = ti.xcom_pull(key="decision", task_ids="decision_task") or {}

        print("===================================")
        print("Décision reçue (brève) →",
            f"side={decision.get('decision')} asset={decision.get('asset')} size_pct={decision.get('size_pct')}")

        # ------- client Binance -------
        b = BaseHook.get_connection("binance_api")
        client = Client(api_key=b.login, api_secret=b.password)

        try:binance_time_offset_ms(client)  # best effort
        except Exception:pass

        # ------- exécution (paliers, contrôles internes dans execute_trade) -------
        try:memory = execute_trade(client, decision, memory)
            
        except Exception as e:
            print(f"[action_task] Erreur exécution : {type(e).__name__} → skip.")
            memory = update_memory_with_decision(
                memory=memory,
                symbol=decision.get("pair", "ETHUSDC"),
                tf=decision.get("tf", "1h"),
                decision_dict={**decision, "decision": "HOLD", "reason": f"action_error:{type(e).__name__}"},
                price_usdc=0.0,
                qty_quote=0.0,
                note="action_task fallback",
                keep_last=20
            )
        
        with open("/opt/airflow/dags/data/memory.json", "w") as f:
            json.dump(memory, f, indent=2)

        # ------- cadence : on ne bloque pas un worker Airflow avec sleep -------
        sleep_s = int(decision.get("time_sleep_s", 0) or 0)
        if sleep_s > 0:
            print(f"[Décideur] time_sleep_s suggéré : {sleep_s}s (aucun sleep ici ; la cadence est gérée par Airflow).")
            time.sleep(sleep_s)


    connexion_task = PythonOperator(task_id="connexion_task", python_callable=connexion_task)

    get_memory_task = PythonOperator(task_id="get_memory_task", python_callable=get_memory)

    decision_task = PythonOperator(task_id="decision_task", python_callable=decision_task)

    action_task = PythonOperator(task_id="action_task", python_callable=action_task)


    connexion_task >> get_memory_task >> decision_task >> action_task
