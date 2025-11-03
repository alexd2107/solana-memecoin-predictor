import pickle
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DEXSCREENER_API = "https://api.dexscreener.com/latest/dex/search"

# List of Solana tokens you want to train on (addresses for your featured coins)
tokens = [
    "5UUH9RTDiSpq6HKS6bp4NdU9PNJpXRXuiw6ShBTBhgH2",  # $TROLL
    "GaPbGp23pPuY9QBLPUjUEBn2MKEroTe9Q3M3f2Xpump",    # $SHITCOIN
    "GtDZKAqvMZMnti46ZewMiXCa4oXF4bZxwQPoKzXPFxZn",  # $NUB
    "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"   # $WIF
    # you can add more symbols if desired, like "BONK","PEPE"... to make the model more general
]

X = []
y = []

for symbol in tokens:
    try:
        resp = requests.get(DEXSCREENER_API, params={"q": symbol})
        data = resp.json()
        for pair in data.get("pairs", []):
            if pair.get("chainId") != "solana":
                continue

            price = float(pair["priceUsd"])
            volume = float(pair["volume"]["h24"])
            liquidity = float(pair["liquidity"]["usd"])

            # Feature vector
            X.append([price, volume, liquidity])

            # Simple label logic based on liquidity and volume
            if volume > 20000 and liquidity > 50000:
                label = 2  # up
            elif volume < 5000 or liquidity < 20000:
                label = 0  # down
            else:
                label = 1  # sideways
            y.append(label)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

X = np.array(X)
y = np.array(y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open("solana_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained on live data for your custom tokens and saved as 'solana_model.pkl'")
