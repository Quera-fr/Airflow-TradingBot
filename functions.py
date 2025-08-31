import streamlit as st


def sidebar(client):
    rows = []
    total_usdc = 0.0
    for b in client.get_account()["balances"]:
        free = float(b["free"]); locked = float(b["locked"])
        total = free + locked
        if total > 0:
            rows.append((b["asset"], total, free, locked))
            if b["asset"] == "USDC":
                total_usdc += total  # additionne la part en USDC

    rows.sort(key=lambda x: x[1], reverse=True)

    with st.sidebar:
        for asset, total, free, locked in rows:
            st.metric(
                label=asset,
                value=f"{total:,.6f}",
                delta=f"Libre {free:,.6f}"
            )

        acc = client.get_account(recvWindow=5000)
        bal = {b["asset"]: float(b["free"]) + float(b["locked"]) for b in acc["balances"]}
        p = {"ETH": float(client.get_symbol_ticker(symbol="ETHUSDC")["price"]),
            "BTC": float(client.get_symbol_ticker(symbol="BTCUSDC")["price"])}
        total_usdc = bal.get("USDC", 0) + sum(bal.get(a, 0) * p[a] for a in ("ETH", "BTC"))
        with st.sidebar:
            st.link_button("Se connecter à Binance", "https://www.binance.com/fr/login",
                        type="primary", use_container_width=True)
            st.metric("Total Portefeuille (USDC)", f"{total_usdc:,.2f}")

    return total_usdc
