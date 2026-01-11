from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import requests
import pickle
import numpy as np
from datetime import datetime, timedelta
import random
import base64
from openai import OpenAI

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# API Keys
BITQUERY_API_KEY = "ory_at_f1B3dQRfIiJSDEKQOkxr4OXXQ1tMwcMN6CQuIWjevc4.4ySJCw0ZUx-zS5nXnJUXRY59X9NXR6uWf_RnEaNvlqc"
MORALIS_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6ImU0ZGQzYzQyLWIyYjgtNDNkZC1iZmE4LTgzMmU3NTgzNzM3YiIsIm9yZ0lkIjoiNDA5MjA3IiwidXNlcklkIjoiNDIwNTY5IiwidHlwZUlkIjoiNjljNzBmMzYtNzBjMS00OTVlLThkNzAtYjM2NzRlMzFjYzExIiwidHlwZSI6IlBST0pFQ1QiLCJpYXQiOjE3MzAwNzQ2MDUsImV4cCI6NDg4NTgzNDYwNX0.ZHXgLyqMR9ijN-vKFxzxgwf0WPKJXcmdsFQCZsDIzOI"
OPENAI_API_KEY = "sk-proj-mz9TE9TCZnsq66V3O-C1M1JjD80Q92tsEEu4WJutZcjkqSKCf_yN8Cy3FdH-4DafD56-YxBvzfT3BlbkFJwNc0wDdGkEKpD6wvRcO8K-CqmIY4Kz1DVPJHNy-oi5z_zNgjw4P4zMuOSk-cC9XQ19fqisA"

# Solscan Pro API key
SOLSCAN_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjcmVhdGVkQXQiOjE3NjgxMzcwODYwMTYsImVtYWlsIjoic29jY2VyYWxleGRva29AZ21haWwuY29tIiwiYWN0aW9uIjoidG9rZW4tYXBpIiwiYXBpVmVyc2lvbiI6InYyIiwiaWF0IjoxNzY4MTM3MDg2fQ.df2kEcUDB_Ti_UKv6gaiJ8CERFlsBpiQ8XIuLEdb4XE"

# Helius DAS API key (for creator history)
HELIUS_API_KEY = "aa25304b-753b-466b-ad17-598a69c0cb7c"
HELIUS_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

# Discord Webhook
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1437292750960594975/2EHZkITnwOC3PwG-h1es1hokmehqlcvUpP6QJPMsIdMjI54YZtP0NdNyEzuE-CCwbRF5"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load the trained model
try:
    with open('solana_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception:
    model = None
    print("Warning: Model file not found")


# ===== On-chain + creator history helpers =====

def get_token_onchain_info(mint_address: str) -> dict:
    """
    Use Solscan Pro API to fetch top holders for this token.
    Powers dev_hold_pct / top5_pct risk checks.
    """
    url = "https://pro-api.solscan.io/v2.0/token/holdersv2"
    params = {
        "address": mint_address,
        "page": 1,
        "page_size": 50
    }
    headers = {
        "accept": "application/json",
        "token": SOLSCAN_API_KEY
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            print("Solscan holders error:", resp.status_code, resp.text)
            return {
                "creator": None,
                "top_holders": [],
                "lp_locked": True,
                "total_supply": 0
            }

        data = resp.json()
        holders_raw = data.get("data", []) or []

        top_holders = []
        for h in holders_raw[:10]:
            pct_raw = h.get("percentage", 0.0)
            try:
                pct = float(pct_raw) / 100.0
            except Exception:
                pct = 0.0
            top_holders.append({
                "address": h.get("owner"),
                "pct": pct
            })

        total_supply = data.get("total", 0)

        # Creator + LP lock would need extra endpoints; set placeholders for now.
        return {
            "creator": None,
            "top_holders": top_holders,
            "lp_locked": True,
            "total_supply": total_supply
        }

    except Exception as e:
        print("Solscan holders exception:", e)
        return {
            "creator": None,
            "top_holders": [],
            "lp_locked": True,
            "total_supply": 0
        }


def get_holder_metrics(onchain_info: dict) -> dict:
    top_holders = onchain_info.get("top_holders", []) or []
    dev_hold_pct = top_holders[0]["pct"] if top_holders else 0.0
    top5_pct = sum(h.get("pct", 0.0) for h in top_holders[:5])
    top10_pct = sum(h.get("pct", 0.0) for h in top_holders[:10])
    lp_locked = onchain_info.get("lp_locked", True)

    return {
        "dev_hold_pct": dev_hold_pct,
        "top5_pct": top5_pct,
        "top10_pct": top10_pct,
        "lp_locked": lp_locked
    }


def get_creator_history(creator_address: str | None) -> dict | None:
    """
    v1 creator history using Helius DAS getAssetsByCreator.
    Currently counts how many assets the creator has; rug classification can be added later.
    """
    if not creator_address:
        return None

    try:
        resp = requests.post(
            HELIUS_URL,
            headers={"Content-Type": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": "creator-history",
                "method": "getAssetsByCreator",
                "params": {
                    "creatorAddress": creator_address,
                    "onlyVerified": True,
                    "page": 1,
                    "limit": 1000,
                },
            },
            timeout=10,
        )

        if resp.status_code != 200:
            print("Helius getAssetsByCreator error:", resp.status_code, resp.text)
            return None

        result = resp.json().get("result", {})
        items = result.get("items", []) or []

        total_tokens = len(items)
        rugged_tokens = 0  # TODO: inspect each asset and mark rugs
        rug_rate = (rugged_tokens / total_tokens) if total_tokens > 0 else 0.0

        return {
            "total_tokens": total_tokens,
            "rugged_tokens": rugged_tokens,
            "rug_rate": rug_rate,
            "last_rug_days_ago": None,
        }

    except Exception as e:
        print("Helius creator history exception:", e)
        return None


# ===== Risk gate =====

def risk_gate(price: float,
              volume24h: float,
              liquidity: float,
              holder_metrics: dict | None = None,
              creator_history: dict | None = None):
    reasons = []
    high_risk = False

    vol_liq_ratio = volume24h / liquidity if liquidity > 0 else 0

    # Pump & dump / structural checks
    if vol_liq_ratio > 5 and liquidity < 50000:
        high_risk = True
        reasons.append("🚨 EXTREME volume/liquidity ratio with low liquidity — likely pump scheme")
    elif liquidity < 30000 and volume24h > 100000:
        high_risk = True
        reasons.append("🚨 Very low liquidity with high volume — potential rug pull risk")
    elif liquidity < 10000 and volume24h > 50000:
        high_risk = True
        reasons.append("🚨 Critically low liquidity — high rug pull risk")

    # Holder / dev concentration checks
    if holder_metrics:
        dev = holder_metrics.get("dev_hold_pct", 0)
        top5 = holder_metrics.get("top5_pct", 0)
        lp_locked = holder_metrics.get("lp_locked", True)

        if dev >= 0.09:
            high_risk = True
            reasons.append(f"🚨 Developer holds ~{dev*100:.1f}% of supply — strong market control risk")
        elif dev >= 0.05:
            reasons.append(f"⚠️ Developer holds ~{dev*100:.1f}% of supply — elevated control risk")

        if top5 >= 0.50:
            high_risk = True
            reasons.append(f"🚨 Top 5 wallets hold {top5*100:.1f}% of supply — whale concentration")
        elif top5 >= 0.40:
            reasons.append(f"⚠️ Top 5 wallets hold {top5*100:.1f}% of supply — watch whale activity")

        if not lp_locked:
            high_risk = True
            reasons.append("🚨 Liquidity is not locked — common rug‑pull pattern")

    # Creator history (will matter once rug_rate > 0)
    if creator_history:
        rug_rate = creator_history.get("rug_rate", 0)
        rugged_tokens = creator_history.get("rugged_tokens", 0)
        total_tokens = creator_history.get("total_tokens", 0)

        if total_tokens >= 2 and rug_rate >= 0.5:
            high_risk = True
            reasons.append(
                f"🚨 Creator has rugged {rugged_tokens}/{total_tokens} previous tokens "
                f"({rug_rate*100:.0f}% rug rate)."
            )
        elif rugged_tokens >= 1 and rug_rate >= 0.25:
            reasons.append(
                f"⚠️ Creator has prior rug history: {rugged_tokens}/{total_tokens} tokens."
            )

    return high_risk, reasons, vol_liq_ratio


def send_discord_notification(symbol: str, token_name: str = None, price: float = None,
                              prediction: str = None, volume24h: float = None,
                              liquidity: float = None):
    try:
        color = 5814783
        if prediction and "10x+ GAIN" in prediction:
            color = 5763719
        elif prediction and "5x GAIN" in prediction:
            color = 5763719
        elif prediction and "2x GAIN" in prediction:
            color = 16776960
        elif prediction and ("LIMITED UPSIDE" in prediction or "AVOID" in prediction):
            color = 15548997

        embed = {
            "title": "🔍 New Crypto Search",
            "color": color,
            "fields": [
                {"name": "🪙 Symbol", "value": symbol, "inline": True},
                {"name": "🕒 Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S EST"), "inline": True}
            ],
            "footer": {"text": "Solana Memecoin Predictor"},
            "timestamp": datetime.utcnow().isoformat()
        }

        if token_name:
            embed["fields"].insert(1, {"name": "📛 Token Name", "value": token_name, "inline": False})
        if price:
            embed["fields"].append({"name": "💰 Price", "value": f"${price:.8f}", "inline": True})
        if volume24h:
            embed["fields"].append({"name": "📊 Volume 24h", "value": f"${volume24h:,.0f}", "inline": True})
        if liquidity:
            embed["fields"].append({"name": "💧 Liquidity", "value": f"${liquidity:,.0f}", "inline": True})
        if prediction:
            embed["fields"].append({"name": "🎯 Prediction", "value": prediction, "inline": False})

        payload = {"embeds": [embed]}
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"Discord notification failed: {e}")


@app.get("/")
async def read_root():
    return FileResponse('static/index.html')


@app.get("/api")
async def get_api():
    return {"message": "Solana Memecoin Predictor API", "status": "running"}


def get_dexscreener_chart_url(mint_address: str) -> str:
    return f"https://dexscreener.com/solana/{mint_address}"


def analyze_chart_image(chart_url: str) -> str:
    try:
        response = requests.get(chart_url, timeout=10)
        if response.status_code != 200:
            return "❌ Unable to fetch chart image for analysis."

        image_base64 = base64.b64encode(response.content).decode('utf-8')

        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this cryptocurrency chart and provide:
1. Pattern identification (pump/dump, accumulation, breakout, consolidation, etc.)
2. Trend direction (bullish/bearish/neutral)
3. Key support and resistance levels
4. Volume trend analysis
5. Risk level (1-10 scale)
6. Whether this shows multi‑X opportunity potential (YES/NO)

Keep analysis concise and actionable."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        analysis = vision_response.choices[0].message.content
        return f"📊 VISUAL CHART ANALYSIS:\n{analysis}"

    except Exception as e:
        return f"❌ Chart analysis unavailable: {str(e)}"


def predict_trend(price: float,
                  volume24h: float,
                  liquidity: float,
                  mint_address: str = None,
                  holder_metrics: dict | None = None,
                  creator_history: dict | None = None) -> dict:
    """Predict if token can achieve multi‑X gains with detailed reasoning"""

    # Risk gate first
    high_risk, reasons, vol_liq_ratio = risk_gate(
        price,
        volume24h,
        liquidity,
        holder_metrics=holder_metrics,
        creator_history=creator_history
    )

    if high_risk:
        reasoning = f"""🔴 GAIN POTENTIAL SCORE: 0/17


🚨 PREDICTION: AVOID THIS TOKEN
⚠️ CONFIDENCE: DO NOT TRADE


{chr(10).join(reasons)}


⚠️ Volume/Liquidity Ratio: {vol_liq_ratio:.2f}
❌ This token shows strong rug‑pull / manipulation characteristics


🛑 RECOMMENDATION: Do NOT enter this trade."""
        return {
            'prediction': '🚨 AVOID - HIGH RUG/PUMP RISK',
            'confidence': 0,
            'reasoning': reasoning,
            'highest_price': price * 1.05,
            'lowest_price': price * 0.70,
            'chart_analysis': ""
        }

    # Upside scoring
    vol_liq_ratio = volume24h / liquidity if liquidity > 0 else 0
    gain_score = 0
    reasoning_parts = []

    # 1. Price Analysis
    if price < 0.00001:
        gain_score += 4
        reasoning_parts.append(f"✅ Ultra-low price (${price:.8f}) — micro-cap potential (+4)")
    elif price < 0.0001:
        gain_score += 3
        reasoning_parts.append(f"✅ Very low price (${price:.6f}) — good growth room (+3)")
    elif price < 0.001:
        gain_score += 2
        reasoning_parts.append(f"✅ Low price (${price:.6f}) — moderate growth potential (+2)")
    elif price < 0.01:
        gain_score += 1
        reasoning_parts.append(f"⚡ Low-mid price (${price:.6f}) — some room to grow (+1)")
    else:
        reasoning_parts.append(f"⚠️ Higher price (${price:.4f}) — less explosive potential (0)")

    # 2. Volume/Liquidity Ratio
    if 1 <= vol_liq_ratio <= 3:
        gain_score += 4
        reasoning_parts.append(f"✅ Optimal volume/liquidity ratio ({vol_liq_ratio:.2f}) — healthy trading (+4)")
    elif 0.5 <= vol_liq_ratio < 1:
        gain_score += 2
        reasoning_parts.append(f"⚡ Moderate ratio ({vol_liq_ratio:.2f}) — building momentum (+2)")
    elif 3 < vol_liq_ratio <= 5:
        gain_score += 1
        reasoning_parts.append(f"⚠️ High ratio ({vol_liq_ratio:.2f}) — watch for volatility (+1)")
    else:
        reasoning_parts.append(f"⚠️ Volume/Liquidity Ratio: {vol_liq_ratio:.2f} — TOO HIGH, possible pump scheme.")

    # 3. ML Model Prediction
    ml_prediction = "Unknown"
    ml_confidence = 0
    if model:
        try:
            features = np.array([[price, volume24h, liquidity]])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            ml_confidence = max(probabilities) * 100

            if prediction == 2:
                ml_prediction = "up"
                if ml_confidence > 70:
                    gain_score += 3
                    reasoning_parts.append(f"✅ ML Model: Strong 'UP' signal ({ml_confidence:.0f}% confidence) (+3)")
                elif ml_confidence > 50:
                    gain_score += 2
                    reasoning_parts.append(f"⚡ ML Model: 'UP' signal ({ml_confidence:.0f}% confidence) (+2)")
                else:
                    gain_score += 1
                    reasoning_parts.append(f"⚡ ML Model: Weak 'UP' signal ({ml_confidence:.0f}% confidence) (+1)")
            elif prediction == 1:
                ml_prediction = "sideways"
                reasoning_parts.append("⚠️ ML Model: 'SIDEWAYS' — neutral momentum (0)")
            else:
                ml_prediction = "down"
                reasoning_parts.append("❌ ML Model: 'DOWN' signal — bearish (0)")
        except Exception:
            reasoning_parts.append("⚠️ ML Model: unavailable")

    # 4. Volume Intensity
    if volume24h > 2000000:
        gain_score += 3
        reasoning_parts.append(f"✅ Exceptionally high trading volume (${volume24h:,.0f}/24h) — strong momentum (+3)")
    elif volume24h > 1000000:
        gain_score += 2
        reasoning_parts.append(f"✅ High trading volume (${volume24h:,.0f}/24h) — good momentum (+2)")
    elif volume24h > 500000:
        gain_score += 1
        reasoning_parts.append(f"⚡ Moderate volume (${volume24h:,.0f}/24h) — building interest (+1)")
    else:
        reasoning_parts.append(f"⚠️ Low volume (${volume24h:,.0f}/24h) — limited momentum (0)")

    # 5. Liquidity Sweet Spot
    if 30000 <= liquidity <= 300000:
        gain_score += 2
        reasoning_parts.append(f"✅ Good liquidity (${liquidity:,.0f}) — optimal for big moves (+2)")
    elif 10000 <= liquidity < 30000 or 300000 < liquidity <= 500000:
        gain_score += 1
        reasoning_parts.append(f"⚡ Acceptable liquidity (${liquidity:,.0f}) (+1)")
    else:
        reasoning_parts.append(f"⚠️ Liquidity (${liquidity:,.0f}) — outside optimal range (0)")

    # 6. Red Flags
    if liquidity < 20000:
        gain_score -= 3
        reasoning_parts.append(f"❌ Very low liquidity (${liquidity:,.0f}) — high risk (-3)")
    if volume24h < 50000:
        gain_score -= 2
        reasoning_parts.append(f"❌ Dead volume (${volume24h:,.0f}/24h) — no momentum (-2)")

    gain_score = max(0, gain_score)

    # Final tiers
    if gain_score >= 15:
        prediction_text = "🔥 10x+ GAIN POTENTIAL"
        confidence_level = "VERY HIGH CONFIDENCE"
        target_mult = 10.0
    elif gain_score >= 12:
        prediction_text = "🚀 5x GAIN POTENTIAL"
        confidence_level = "HIGH CONFIDENCE"
        target_mult = 5.0
    elif gain_score >= 9:
        prediction_text = "⚡ 2x GAIN POTENTIAL"
        confidence_level = "MODERATE CONFIDENCE"
        target_mult = 2.0
    elif gain_score >= 6:
        prediction_text = "📈 30%+ GAIN POTENTIAL"
        confidence_level = "LOW–MODERATE CONFIDENCE"
        target_mult = 1.3
    else:
        prediction_text = "⚠️ LIMITED UPSIDE (<30%)"
        confidence_level = "LOW CONFIDENCE"
        target_mult = 1.1

    if target_mult >= 10:
        recommendation = "✅ RECOMMENDATION: High‑conviction degen play; size small and manage risk aggressively."
    elif target_mult >= 5:
        recommendation = "✅ RECOMMENDATION: Strong upside; consider staged entries and profit‑taking levels."
    elif target_mult >= 2:
        recommendation = "⚠️ RECOMMENDATION: Good 2x potential; enter with clear stop loss and TP targets."
    elif target_mult >= 1.3:
        recommendation = "⚠️ RECOMMENDATION: Solid 30%+ setup; suitable for shorter swing trades."
    else:
        recommendation = "❌ RECOMMENDATION: Upside is limited; better opportunities likely elsewhere."

    reasoning_output = f"""📊 GAIN POTENTIAL SCORE: {gain_score}/17


🎯 PREDICTION: {prediction_text}
💪 CONFIDENCE: {confidence_level}


{chr(10).join(reasoning_parts)}


🤖 ML Model says: '{ml_prediction}' with {ml_confidence:.0f}% confidence


{recommendation}"""

    chart_analysis = ""
    if mint_address:
        try:
            chart_url = get_dexscreener_chart_url(mint_address)
            chart_analysis = analyze_chart_image(chart_url)
        except Exception as e:
            chart_analysis = f"❌ Chart analysis failed: {str(e)}"

    if target_mult >= 5:
        max_drop_mult = 0.6
    elif target_mult >= 2:
        max_drop_mult = 0.75
    else:
        max_drop_mult = 0.85

    return {
        'prediction': prediction_text,
        'confidence': ml_confidence if model else 50,
        'reasoning': reasoning_output,
        'highest_price': price * target_mult,
        'lowest_price': price * max_drop_mult,
        'chart_analysis': chart_analysis
    }


@app.get("/api/predict")
async def predict(symbol: str):
    try:
        # Dexscreener
        search_url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}"
        response = requests.get(search_url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get('pairs'):
                pair = data['pairs'][0]
                token_symbol = pair.get('baseToken', {}).get('symbol', symbol)
                token_name = pair.get('baseToken', {}).get('name', 'Unknown')
                token_address = pair.get('baseToken', {}).get('address', symbol)
                price = float(pair.get('priceUsd', 0))
                volume24h = float(pair.get('volume', {}).get('h24', 0))
                liquidity = float(pair.get('liquidity', {}).get('usd', 0))

                onchain_info = get_token_onchain_info(token_address)
                holder_metrics = get_holder_metrics(onchain_info)
                creator_addr = onchain_info.get("creator")
                creator_history = get_creator_history(creator_addr)

                result = predict_trend(
                    price,
                    volume24h,
                    liquidity,
                    token_address,
                    holder_metrics=holder_metrics,
                    creator_history=creator_history
                )

                send_discord_notification(
                    symbol=token_symbol,
                    token_name=token_name,
                    price=price,
                    prediction=result['prediction'],
                    volume24h=volume24h,
                    liquidity=liquidity
                )

                return {
                    'symbol': token_symbol,
                    'name': token_name,
                    'price': price,
                    'volume24h': volume24h,
                    'liquidity': liquidity,
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'reasoning': result['reasoning'],
                    'highest_price': result['highest_price'],
                    'lowest_price': result['lowest_price'],
                    'chart_analysis': result['chart_analysis']
                }

        # Moralis fallback
        moralis_url = f"https://solana-gateway.moralis.io/token/mainnet/{symbol}/price"
        headers = {"X-API-Key": MORALIS_API_KEY}
        response = requests.get(moralis_url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            price = float(data.get('usdPrice', 0))
            token_name = data.get('name', 'Unknown')

            volume24h = 100000
            liquidity = 50000

            onchain_info = get_token_onchain_info(symbol)
            holder_metrics = get_holder_metrics(onchain_info)
            creator_addr = onchain_info.get("creator")
            creator_history = get_creator_history(creator_addr)

            result = predict_trend(
                price,
                volume24h,
                liquidity,
                symbol,
                holder_metrics=holder_metrics,
                creator_history=creator_history
            )

            send_discord_notification(
                symbol=symbol,
                token_name=token_name,
                price=price,
                prediction=result['prediction'],
                volume24h=volume24h,
                liquidity=liquidity
            )

            return {
                'symbol': symbol,
                'name': token_name,
                'price': price,
                'volume24h': volume24h,
                'liquidity': liquidity,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'reasoning': result['reasoning'],
                'highest_price': result['highest_price'],
                'lowest_price': result['lowest_price'],
                'chart_analysis': result['chart_analysis']
            }

        # Bitquery fallback
        bitquery_url = "https://streaming.bitquery.io/graphql"
        headers = {
            "Authorization": f"Bearer {BITQUERY_API_KEY}",
            "Content-Type": "application/json"
        }

        query = """
        query ($token: String!) {
            Solana {
                DEXTradeByTokens(
                    where: {Trade: {Currency: {MintAddress: {is: $token}}}}
                    limit: {count: 1}
                ) {
                    Trade {
                        Currency {
                            Symbol
                            Name
                            MintAddress
                        }
                        PriceInUSD
                    }
                }
            }
        }
        """

        bitquery_response = requests.post(
            bitquery_url,
            json={'query': query, 'variables': {'token': symbol}},
            headers=headers,
            timeout=10
        )

        if bitquery_response.status_code == 200:
            bitquery_data = bitquery_response.json()
            trades = bitquery_data.get('data', {}).get('Solana', {}).get('DEXTradeByTokens', [])

            if trades:
                trade = trades[0]['Trade']
                token_symbol = trade['Currency']['Symbol']
                token_name = trade['Currency']['Name']
                price = float(trade['PriceInUSD'])

                volume24h = 100000
                liquidity = 50000

                onchain_info = get_token_onchain_info(symbol)
                holder_metrics = get_holder_metrics(onchain_info)
                creator_addr = onchain_info.get("creator")
                creator_history = get_creator_history(creator_addr)

                result = predict_trend(
                    price,
                    volume24h,
                    liquidity,
                    symbol,
                    holder_metrics=holder_metrics,
                    creator_history=creator_history
                )

                send_discord_notification(
                    symbol=token_symbol,
                    token_name=token_name,
                    price=price,
                    prediction=result['prediction'],
                    volume24h=volume24h,
                    liquidity=liquidity
                )

                return {
                    'symbol': token_symbol,
                    'name': token_name,
                    'price': price,
                    'volume24h': volume24h,
                    'liquidity': liquidity,
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'reasoning': result['reasoning'],
                    'highest_price': result['highest_price'],
                    'lowest_price': result['lowest_price'],
                    'chart_analysis': result['chart_analysis']
                }

        raise HTTPException(status_code=404, detail="Token not found on any exchange")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/latest-tokens")
async def get_latest_tokens():
    try:
        trending = [
            {"symbol": "$TROLL", "price": "$0.000010"},
            {"symbol": "$SHITCOIN", "price": "$0.000020"},
            {"symbol": "$NUB", "price": "$0.000030"},
            {"symbol": "$WIF", "price": "$0.000040"}
        ]
        return {"tokens": trending}
    except Exception as e:
        return {"tokens": [], "error": str(e)}


@app.get("/api/history")
async def get_history(symbol: str):
    try:
        search_url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}"
        response = requests.get(search_url, timeout=10)

        current_price = 0.0001
        if response.status_code == 200:
            data = response.json()
            if data.get('pairs'):
                pair = data['pairs'][0]
                current_price = float(pair.get('priceUsd', 0.0001))

        history = []
        base_time = datetime.now()

        for i in range(100, 0, -1):
            timestamp = (base_time - timedelta(minutes=i * 5)).isoformat()
            variation = random.uniform(-0.05, 0.05)
            price = current_price * (1 + variation)
            history.append({'time': timestamp, 'price': price})

        history.append({'time': base_time.isoformat(), 'price': current_price})

        future = []
        for i in range(1, 13):
            future_time = (base_time + timedelta(minutes=i * 5)).isoformat()
            trend = random.uniform(0.001, 0.015)
            future_price = current_price * (1 + trend * i * 0.3)
            future.append({'time': future_time, 'price': future_price})

        all_prices = [p['price'] for p in history] + [p['price'] for p in future]

        return {
            'history': history,
            'future': future,
            'high_prediction': max(all_prices),
            'low_prediction': min(all_prices)
        }

    except Exception as e:
        print(f"Chart generation error: {str(e)}")
        current_time = datetime.now()
        fallback_price = 0.0001

        return {
            'history': [{'time': (current_time - timedelta(hours=i)).isoformat(),
                        'price': fallback_price * random.uniform(0.95, 1.05)}
                       for i in range(10, 0, -1)],
            'future': [{'time': (current_time + timedelta(hours=i)).isoformat(),
                       'price': fallback_price * random.uniform(1.0, 1.1)}
                      for i in range(1, 5)],
            'high_prediction': fallback_price * 1.1,
            'low_prediction': fallback_price * 0.95
        }


@app.get("/api/token-info")
async def get_token_info(symbol: str):
    try:
        search_url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}"
        response = requests.get(search_url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get('pairs'):
                pair = data['pairs'][0]
                return {
                    'symbol': pair.get('baseToken', {}).get('symbol', symbol),
                    'name': pair.get('baseToken', {}).get('name', 'Unknown'),
                    'address': pair.get('baseToken', {}).get('address', symbol),
                    'price': float(pair.get('priceUsd', 0)),
                    'volume24h': float(pair.get('volume', {}).get('h24', 0)),
                    'liquidity': float(pair.get('liquidity', {}).get('usd', 0))
                }

        raise HTTPException(status_code=404, detail="Token not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/solana-price")
async def get_solana_price():
    try:
        response = requests.get(
            'https://api.coingecko.com/api/v3/simple/price',
            params={'ids': 'solana', 'vs_currencies': 'usd', 'include_24hr_change': 'true'},
            timeout=10,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        if response.status_code == 200:
            data = response.json()
            if 'solana' in data:
                return {
                    'price': data['solana']['usd'],
                    'change_24h': data['solana'].get('usd_24h_change', 0)
                }

        dex_response = requests.get(
            'https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112',
            timeout=10
        )
        if dex_response.status_code == 200:
            dex_data = dex_response.json()
            if dex_data.get('pairs'):
                pair = dex_data['pairs'][0]
                price = float(pair.get('priceUsd', 0))
                change = float(pair.get('priceChange', {}).get('h24', 0))
                if price > 0:
                    return {'price': price, 'change_24h': change}

        binance_response = requests.get(
            'https://api.binance.com/api/v3/ticker/24hr?symbol=SOLUSDT',
            timeout=10
        )
        if binance_response.status_code == 200:
            binance_data = binance_response.json()
            return {
                'price': float(binance_data['lastPrice']),
                'change_24h': float(binance_data['priceChangePercent'])
            }

        return {'error': 'Failed to fetch price from all sources'}

    except Exception as e:
        print(f"Solana price error: {str(e)}")
        return {'error': str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
