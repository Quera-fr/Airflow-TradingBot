import streamlit as st
import json, time, os
from functions import sidebar
from binance.client import Client
import pandas as pd

with open("dags/data/memory.json") as f:
    memory = json.load(f)

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret)

sidebar(client)

st.title("Historique des décisions")
if st.checkbox("Voir le contenu complet du fichier mémoire"):
    decisions = {}
    for memo in memory.get("recent_decisions", [])[::-1]:
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(memo["ts"]))
        except:
            ts = None

        try:
            action = memo["action"]
        except:
            action = None

        try:
            time_sleep = int(memo["time_sleep"])
        except:
            time_sleep = None

        try:
            reason = memo["reason"].replace("\n", " ")
        except:
            reason = None

        try:
            next_steps = memo['next_steps'].replace("\n", " ")
        except:
            next_steps = None

        decisions[ts] = {"reason": reason, "next_steps": next_steps, "time_sleep": time_sleep, "action": action}

    st.write(decisions)



import altair as alt

with open("memory.json") as f:
    memory = json.load(f)

st.title("Historique du capital")

cap_hist = memory.get("capital_history", [])
if cap_hist:
    data = [{"t": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p["timestamp"])),
             "c": float(p["capital"])} for p in cap_hist]
    df = pd.DataFrame(data)

    ymin, ymax = df["c"].min(), df["c"].max()

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("t:T", title="Temps"),
            y=alt.Y("c:Q", title="Capital", scale=alt.Scale(domain=[ymin-5, ymax+5])),
            tooltip=[alt.Tooltip("t:T", title="Date/Heure"),
                     alt.Tooltip("c:Q", title="Capital", format=".2f")]
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)