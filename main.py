from flask import Flask, render_template
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os

app = Flask(__name__)

@app.route('/')
def crypto_analyzer():
    try:
        coin_id = "bitcoin"
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
        response = requests.get(url, timeout=10).json()
        if not response.get("prices") or len(response["prices"]) == 0:
            return "Error: No price data from API. Please wait 5-10 minutes and retry (API rate limit)."

        price_df = pd.DataFrame(response["prices"], columns=["time", "price"])
        price_df["time"] = pd.to_datetime(price_df["time"], unit="ms")

        volume_df = pd.DataFrame(response["total_volumes"], columns=["time", "volume"])
        volume_df["time"] = pd.to_datetime(volume_df["time"], unit="ms")

        market_cap_df = pd.DataFrame(response["market_caps"], columns=["time", "market_cap"])
        market_cap_df["time"] = pd.to_datetime(market_cap_df["time"], unit="ms")

        df = price_df.merge(volume_df, on="time", how="outer").merge(market_cap_df, on="time", how="outer")
        df = df.sort_values("time").fillna(method="ffill").fillna(0)

        df["price_change"] = df["price"].pct_change().fillna(0)
        df["inflow"] = np.where(df["price_change"] > 0, df["price"] * df["price_change"], 0)
        df["outflow"] = np.where(df["price_change"] < 0, df["price"] * abs(df["price_change"]), 0)
        df["volume_percent_mc"] = (df["volume"] / df["market_cap"]).fillna(0) * 100

        delta = df["price"].diff().fillna(0)
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)
        df["RSI"] = 100 - (100 / (1 + rs)).fillna(50)

        exp1 = df["price"].ewm(span=12, min_periods=1).mean()
        exp2 = df["price"].ewm(span=26, min_periods=1).mean()
        macd = exp1 - exp2
        df["MACD"] = macd.fillna(0)
        df["MACD_signal"] = macd.ewm(span=9, min_periods=1).mean().fillna(0)
        df["signal"] = np.where((df["RSI"] < 30) & (df["MACD"] > df["MACD_signal"]), "Buy",
                               np.where((df["RSI"] > 70) & (df["MACD"] < df["MACD_signal"]), "Sell", "Hold"))

        vol_7d = df["volume"].tail(7).mean() if len(df) >= 7 else df["volume"].mean()
        vol_30d = df["volume"].mean()
        vol_ratio = vol_7d / vol_30d if vol_30d > 0 else 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price", line=dict(color="#00ff00")))
        fig.add_trace(go.Scatter(x=df.index, y=df["market_cap"], name="Market Cap", line=dict(color="#ffa500")))
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#ff00ff")))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#0000ff")))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="MACD Signal", line=dict(color="#ff0000")))
        fig.update_layout(title=f"{coin_id.capitalize()} Analysis", xaxis_title="Days", yaxis_title="Value",
                          height=600, template="plotly_dark")

        plot_div = fig.to_html(full_html=False)
        latest_data = df.iloc[-1].fillna(0)

        return render_template('index.html',
                              price=f"${latest_data['price']:,.2f}",
                              market_cap=f"${latest_data['market_cap']:,.0f}",
                              vol_percent=f"{latest_data['volume_percent_mc']:.2f}%",
                              inflow=f"${df['inflow'].sum():,.0f}",
                              outflow=f"${df['outflow'].sum():,.0f}",
                              vol_ratio=f"{vol_ratio:.2f}",
                              signal=latest_data["signal"],
                              plot=plot_div,
                              time=datetime.now().strftime("%Y-%m-%d %H:%M"))
    except Exception as e:
        return f"Server Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
