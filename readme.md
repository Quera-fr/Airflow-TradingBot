# Crypto Trading Bot – Airflow + OpenAI + Binance

## 📌 Description
Ce projet est un **bot de trading spot automatisé** pour les cryptomonnaies (ETH, BTC, USDC).  
Il combine :
- **Airflow** : orchestration des tâches (cadence par défaut 1h, adaptative selon conditions de marché).
- **Binance API** : exécution des ordres et récupération des données de marché.
- **OpenAI (GPT)** : moteur de décision basé sur un préprompt rigoureux et des données enrichies (OHLCV, indicateurs techniques, sizing, régime de marché, mémoire).

L’objectif est d’assurer une **croissance durable du capital** tout en respectant des contraintes strictes de gestion du risque et d’exposition.

---

## 🚀 Fonctionnalités principales
- **Collecte des données marché** (`get_market_data`)
  - OHLCV (1h, 1d, 1M).
  - Indicateurs techniques : EMA20/50, RSI14, MACD, ATR.
  - Snapshots horaires et statistiques consolidées.
  - Sizing dynamique avec **paliers d’achat/vente** (5, 10, 25, 50, 75 %).
- **Suivi du capital et balances** (`get_capital_and_balances`)
  - Valorisation multi-actifs (USDC, ETH, BTC).
  - Historique du capital et calcul du drawdown 7j.
- **Mémoire persistante**
  - Sauvegarde des décisions passées, performance, contraintes, capital.
  - Suivi des résultats trades fermés/ouverts.
- **Analyse de régime** (`compute_regime`)
  - Tendances daily et monthly (slopes EMA, retours 3/6/12 mois).
  - Volatilité relative (ATR rank).
  - Force relative ETH vs BTC (H1 et D1).
- **Décideur IA** (`get_decision`)
  - Prend en entrée un payload JSON enrichi.
  - Respecte un préprompt de trading avec contraintes strictes.
  - Génère une décision unique : **BUY / SELL / HOLD**.
  - Cadence adaptative (`time_sleep_s`).
- **Exécution des trades** (`execute_trade`)
  - Vérification des contraintes (`lot_size`, `min_notional`, `max_rebalance_pct`).
  - Passage d’ordres sur Binance (spot).
  - Mise à jour mémoire et suivi des exécutions.

---

## 📂 Structure du projet
```

dags/
│── dag\_bot.py              # DAG Airflow principal (cadence 1h par défaut)
│── utiles.py               # Fonctions utilitaires (market data, regime, capital, mémoire…)
│── data/
│    └── memory\_<dag>\_<run>.json   # Mémoire persistante par exécution
README.md
requirements.txt

````

---

## ⚙️ Installation
### 1. Cloner le repo
```bash
git clone https://github.com/<votre_repo>.git
cd trading-bot
````

### 2. Créer et activer un environnement Python

```bash
python3 -m venv venv
source venv/bin/activate   # sous Linux/Mac
venv\Scripts\activate      # sous Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les connexions Airflow

Dans Airflow :

* `binance_api` : `login=<API_KEY>` / `password=<API_SECRET>`.
* `openai_default` : `password=<OPENAI_API_KEY>`.

### 5. Lancer Airflow

```bash
airflow standalone
```

---

## 📊 Paramètres et contraintes de trading

* **Drawdown max toléré (7j)** : −5 % → HOLD forcé, capital 100 % USDC.
* **Halt** : si capital.current < 0.5 × capital.initial → 100 % USDC.
* **Risque max par trade** : 10 % du capital.
* **Max rebalance par tour (BUY)** : 30 % du capital courant.
* **Exposition max par actif** : 25 % (modifiable dans `memory.constraints`).
* **Respect strict** : `LOT_SIZE` et `MIN_NOTIONAL`.

---

## 🔍 Exemple de payload transmis au décideur

```json
{
  "market_data": {
    "ETHUSDC": {
      "symbol": "ETHUSDC",
      "tf": "1h",
      "features": {
        "ohlcv_stats": {"change_1h_pct": 0.31, "change_3h_pct": 0.15, "change_12h_pct": 0.25},
        "stats": {"ema20": {"last": 4419.48}, "rsi14": {"last": 45.73}, "macd": {"last": -9.05}}
      },
      "sizing": {
        "usdc_free": 49.63,
        "paliers_buy": [{"pct": 25, "qty_base": 0.0028, "notional_usdc": 12.32, "feasible": true}],
        "paliers_sell": [{"pct": 75, "qty_base": 0.0012, "notional_usdc": 5.28, "feasible": true}]
      },
      "position_state": {"status": "long_spot", "size_quote": 7.15}
    }
  },
  "memory": {
    "capital": {"initial": 41.0, "current": 61.3, "max_dd_7d": -0.08, "halt_triggered": false},
    "constraints": {"min_notional_usdc": 5.0, "lot_size_eth": 0.0001, "lot_size_btc": 0.00001}
  }
}
```

---

## 📤 Exemple de sortie du décideur

```json
{
  "decision": "BUY",
  "asset": "ETH",
  "pair": "ETHUSDC",
  "confidence": 0.78,
  "entry": "market",
  "sl": 0.0,
  "tp": 0.0,
  "qty_base": 0.0028,
  "reason": "Momentum H1 positif (MACD_hist >0, RSI>52, EMA20/50 slope>0), palier 25% faisable (12.3 USDC, ≥min_notional, lot_size respecté).",
  "risk_check": "ok",
  "next_steps": "Surveiller consolidation > EMA50 H1, réduire si RSI<50.",
  "time_sleep_s": 1800
}
```

---

## 🛡️ Notes importantes

* Le bot est conçu pour du **spot trading uniquement** (jamais de short).
* Le sizing est basé sur des **paliers progressifs**, validés après contrôle `lot_size` et `min_notional`.
* Le décideur peut forcer des **HOLD** en cas de contraintes non respectées, drawdown trop élevé, ou absence de convergence technique.
* Ce projet est fourni à titre éducatif — **utilisation en réel sous votre entière responsabilité**.

---

## 📜 Licence

Projet sous licence MIT. Vous êtes libre de l’utiliser, le modifier et le redistribuer.
