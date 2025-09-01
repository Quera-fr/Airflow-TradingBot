import streamlit as st

def sidebar(client):
    rows = []
    acc = client.get_account(recvWindow=5000)
    bal = {b["asset"]: float(b["free"]) + float(b["locked"]) for b in acc["balances"]}
    px = {
        "ETH": float(client.get_symbol_ticker(symbol="ETHUSDC")["price"]),
        "BTC": float(client.get_symbol_ticker(symbol="BTCUSDC")["price"]),
    }

    for b in acc["balances"]:
        total = float(b["free"]) + float(b["locked"])
        if total > 0:
            rows.append((b["asset"], total, float(b["free"])))

    rows.sort(key=lambda x: x[1], reverse=True)

    with st.sidebar:
        for asset, total, free in rows:
            usdc_val = f"≈ {total*px[asset]:.2f} USDC" if asset in px else ""
            st.metric(label=asset, value=f"{total:,.6f}", delta=usdc_val)

        total_usdc = bal.get("USDC", 0) + sum(bal[a] * px[a] for a in ("ETH", "BTC"))
        st.link_button("Se connecter à Binance", "https://www.binance.com/fr/login",
                       type="primary", use_container_width=True)
        st.metric("Total Portefeuille (USDC)", f"{total_usdc:,.2f} USDC")
    return total_usdc
