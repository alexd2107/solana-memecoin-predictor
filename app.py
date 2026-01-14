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
import yfinance as yf

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# API Keys
BITQUERY_API_KEY = "ory_at_f1B3dQRfIiJSDEKQOkxr4OXXQ1tMwcMN6CQuIWjevc4.4ySJCw0ZUx-zS5nXnJUXRY59X9NXR6uWf_RnEaNvlqc"
MORALIS_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6ImU0ZGQzYzQyLWIyYjgtNDNkZC1iZmE4LTgzMmU3NTgzNzM3YiIsIm9yZ0lkIjoiNDA5MjA3IiwidXNlcklkIjoiNDIwNTY5IiwidHlwZUlkIjoiNjljNzBmMzYtNzBjMS00OTVlLThkNzAtYjM2NzRlMzFjYzExIiwidHlwZSI6IlBST0pFQ1QiLCJpYXQiOjE3MzAwNzQ2MDUsImV4cCI6NDg4NTgzNDYwNX0.ZHXgLyqMR9ijN-vKFxzxgwf0WPKJXcmdsFQCZsDIzOI"
OPENAI_API_KEY = "sk-proj-mz9TE9TCZnsq66V3O-C1M1JjD80Q92tsEEu4WJutZcjkqSKCf_yN8Cy3FdH-4DafD56-YxBvzfT3BlbkFJwNc0wDdGkEKpD6wvRcO8K-CqmIY4Kz1DVPJHNy-oi5z_zNgjw4P4zMuOSk-cC9XQ19fqisA"
SOLSCAN_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjcmVhdGVkQXQiOjE3NjgxMzcwODYwMTYsImVtYWlsIjoic29jY2VyYWxleGRva29AZ21haWwuY29tIiwiYWN0aW9uIjoidG9rZW4tYXBpIiwiYXBpVmVyc2lvbiI6InYyIiwiaWF0IjoxNzY4MTM3MDg2fQ.df2kEcUDB_Ti_UKv6gaiJ8CERFlsBpiQ8XIuLEdb4XE"
HELIUS_API_KEY = "aa25304b-753b-466b-ad17-598a69c0cb7c"
HELIUS_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

# Discord Webhooks (separate for crypto and stocks)
DISCORD_WEBHOOK_CRYPTO = "https://discord.com/api/webhooks/1437292750960594975/2EHZkITnwOC3PwG-h1es1hokmehqlcvUpP6QJPMsIdMjI54YZtP0NdNyEzuE-CCwbRF5"
DISCORD_WEBHOOK_STOCK = "https://discord.com/api/webhooks/1460815556130246729/7yfC-1AAJ51T9aVrtcU0cNQBxfZXLl177kNMiSVJfd6bamVHG-4u4VRJAPh8d94wlK1s"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load the trained model
try:
    with open('solana_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception:
    model = None
    print("Warning: Model file not found")


# ===== CRYPTO: On-chain + creator history helpers =====

def get_token_onchain_info(mint_address: str) -> dict:
    url = "https://pro-api.solscan.io/v2.0/token/holdersv2"
    params = {"address": mint_address, "page": 1, "page_size": 50}
    headers = {"accept": "application/json", "token": SOLSCAN_API_KEY}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            print("Solscan holders error:", resp.status_code, resp.text)
            return {"creator": None, "top_holders": [], "lp_locked": True, "total_supply": 0}
        
        data = resp.json()
        holders_raw = data.get("data", []) or []
        
        top_holders = []
        for h in holders_raw[:10]:
            pct_raw = h.get("percentage", 0.0)
            try:
                pct = float(pct_raw) / 100.0
            except Exception:
                pct = 0.0
            top_holders.append({"address": h.get("owner"), "pct": pct})
        
        return {
            "creator": None,
            "top_holders": top_holders,
            "lp_locked": True,
            "total_supply": data.get("total", 0)
        }
    except Exception as e:
        print("Solscan holders exception:", e)
        return {"creator": None, "top_holders": [], "lp_locked": True, "total_supply": 0}


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


def get_dexscreener_chart_url(mint_address: str) -> str:
    return f"https://dexscreener.com/solana/{mint_address}"


def get_creator_history(creator_address: str | None) -> dict | None:
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
            return None
        
        result = resp.json().get("result", {})
        items = result.get("items", []) or []
        total_tokens = len(items)
        rugged_tokens = 0
        
        for asset in items[:10]:
            mint = asset.get("id")
            if not mint:
                continue
            
            chart_url = get_dexscreener_chart_url(mint)
            
            try:
                chart_response = requests.get(chart_url, timeout=10)
                if chart_response.status_code != 200:
                    continue
                
                image_base64 = base64.b64encode(chart_response.content).decode('utf-8')
                vision_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": """Analyze this cryptocurrency chart and classify it:

Does this show a PUMP-AND-DUMP / RUG PULL pattern?

Rug indicators:
- Parabolic spike followed by 80%+ drop
- Sudden liquidity drain
- Volume spike then dead volume
- No recovery after crash

Answer with: RUG: YES or RUG: NO

Then briefly explain why in 1-2 sentences."""
                        }, {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        }]
                    }],
                    max_tokens=200
                )
                
                analysis = vision_response.choices[0].message.content
                if "RUG: YES" in analysis or ("pump" in analysis.lower() and "dump" in analysis.lower()):
                    rugged_tokens += 1
            except Exception:
                continue
        
        rug_rate = (rugged_tokens / min(total_tokens, 10)) if total_tokens > 0 else 0.0
        return {
            "total_tokens": total_tokens,
            "rugged_tokens": rugged_tokens,
            "rug_rate": rug_rate,
            "last_rug_days_ago": None,
        }
    except Exception:
        return None


def risk_gate(price: float, volume24h: float, liquidity: float,
              holder_metrics: dict | None = None,
              creator_history: dict | None = None):
    reasons = []
    high_risk = False
    vol_liq_ratio = volume24h / liquidity if liquidity > 0 else 0
    
    if vol_liq_ratio > 5 and liquidity < 50000:
        high_risk = True
        reasons.append("🚨 EXTREME volume/liquidity ratio with low liquidity — likely pump scheme")
    elif liquidity < 30000 and volume24h > 100000:
        high_risk = True
        reasons.append("🚨 Very low liquidity with high volume — potential rug pull risk")
    elif liquidity < 10000 and volume24h > 50000:
        high_risk = True
        reasons.append("🚨 Critically low liquidity — high rug pull risk")
    
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
    
    if creator_history:
        rug_rate = creator_history.get("rug_rate", 0)
        rugged_tokens = creator_history.get("rugged_tokens", 0)
        total_tokens = creator_history.get("total_tokens", 0)
        
        if total_tokens >= 2 and rug_rate >= 0.5:
            high_risk = True
            reasons.append(f"🚨 Creator has rugged {rugged_tokens}/{total_tokens} previous tokens ({rug_rate*100:.0f}% rug rate based on chart analysis).")
        elif rugged_tokens >= 1 and rug_rate >= 0.25:
            reasons.append(f"⚠️ Creator has prior rug history: {rugged_tokens}/{total_tokens} tokens showed rug patterns.")
    
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
        elif prediction and ("LIMITED UPSIDE" in prediction or "AVOID" in prediction or "RUG PULL" in prediction):
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
        requests.post(DISCORD_WEBHOOK_CRYPTO, json=payload, timeout=5)
    except Exception as e:
        print(f"Discord crypto notification failed: {e}")


def send_stock_discord_notification(ticker: str, company_name: str = None, price: float = None,
                                    prediction: str = None, sector: str = None, position_type: str = None):
    """Send stock search notification to Discord"""
    try:
        color = 5814783
        if prediction and "STRONG BUY" in prediction:
            color = 5763719
        elif prediction and "BUY" in prediction:
            color = 5763719
        elif prediction and "HOLD" in prediction:
            color = 16776960
        elif prediction and "AVOID" in prediction:
            color = 15548997
        
        embed = {
            "title": "📊 New Stock Search",
            "color": color,
            "fields": [
                {"name": "📈 Ticker", "value": ticker, "inline": True},
                {"name": "🕒 Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S EST"), "inline": True}
            ],
            "footer": {"text": "Stock Market Analyst"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if company_name:
            embed["fields"].insert(1, {"name": "🏢 Company", "value": company_name, "inline": False})
        if price:
            embed["fields"].append({"name": "💰 Price", "value": f"${price:.2f}", "inline": True})
        if sector:
            embed["fields"].append({"name": "🏭 Sector", "value": sector, "inline": True})
        if prediction:
            embed["fields"].append({"name": "🎯 Prediction", "value": prediction, "inline": False})
        if position_type:
            embed["fields"].append({"name": "📍 Position", "value": position_type, "inline": False})
        
        payload = {"embeds": [embed]}
        requests.post(DISCORD_WEBHOOK_STOCK, json=payload, timeout=5)
    except Exception as e:
        print(f"Discord stock notification failed: {e}")


@app.get("/")
async def read_root():
    return FileResponse('static/index.html')


@app.get("/api")
async def get_api():
    return {"message": "Market Analyst API - Crypto & Stocks", "status": "running"}


def analyze_chart_image(chart_url: str) -> str:
    try:
        response = requests.get(chart_url, timeout=10)
        if response.status_code != 200:
            return "❌ Unable to fetch chart image for analysis."
        
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": """Analyze this cryptocurrency chart and provide:
1. Pattern identification (pump/dump, accumulation, breakout, consolidation, etc.)
2. Trend direction (bullish/bearish/neutral)
3. Key support and resistance levels
4. Volume trend analysis
5. Risk level (1-10 scale)
6. Whether this shows multi‑X opportunity potential (YES/NO)

Keep analysis concise and actionable."""
                }, {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                }]
            }],
            max_tokens=500
        )
        
        analysis = vision_response.choices[0].message.content
        return f"📊 VISUAL CHART ANALYSIS:\n{analysis}"
    except Exception as e:
        return f"❌ Chart analysis unavailable: {str(e)}"
def predict_trend(price: float, volume24h: float, liquidity: float,
                  mint_address: str = None,
                  holder_metrics: dict | None = None,
                  creator_history: dict | None = None) -> dict:
    
    high_risk, reasons, vol_liq_ratio = risk_gate(price, volume24h, liquidity, holder_metrics, creator_history)
    
    if high_risk:
        reasoning = f"""🔴 GAIN POTENTIAL SCORE: 0/17

🚨 PREDICTION: AVOID THIS TOKEN
⚠️ CONFIDENCE: HIGH CHANCE OF RUG PULL

{chr(10).join(reasons)}

⚠️ Volume/Liquidity Ratio: {vol_liq_ratio:.2f}
❌ This token shows strong rug‑pull / manipulation characteristics

🛑 RECOMMENDATION: Do NOT enter this trade."""
        return {
            'prediction': '🚨 AVOID - HIGH CHANCE OF RUG PULL',
            'confidence': 0,
            'reasoning': reasoning,
            'highest_price': price * 1.05,
            'lowest_price': price * 0.70,
            'chart_analysis': ""
        }
    
    vol_liq_ratio = volume24h / liquidity if liquidity > 0 else 0
    gain_score = 0
    reasoning_parts = []
    
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
    
    if 30000 <= liquidity <= 300000:
        gain_score += 2
        reasoning_parts.append(f"✅ Good liquidity (${liquidity:,.0f}) — optimal for big moves (+2)")
    elif 10000 <= liquidity < 30000 or 300000 < liquidity <= 500000:
        gain_score += 1
        reasoning_parts.append(f"⚡ Acceptable liquidity (${liquidity:,.0f}) (+1)")
    else:
        reasoning_parts.append(f"⚠️ Liquidity (${liquidity:,.0f}) — outside optimal range (0)")
    
    if liquidity < 20000:
        gain_score -= 3
        reasoning_parts.append(f"❌ Very low liquidity (${liquidity:,.0f}) — high risk (-3)")
    if volume24h < 50000:
        gain_score -= 2
        reasoning_parts.append(f"❌ Dead volume (${volume24h:,.0f}/24h) — no momentum (-2)")
    
    gain_score = max(0, gain_score)
    
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
                
                result = predict_trend(price, volume24h, liquidity, token_address, holder_metrics, creator_history)
                send_discord_notification(token_symbol, token_name, price, result['prediction'], volume24h, liquidity)
                
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
        token_address = symbol
        volume24h = 100000
        liquidity = 50000
        
        if response.status_code == 200:
            data = response.json()
            if data.get('pairs'):
                pair = data['pairs'][0]
                current_price = float(pair.get('priceUsd', 0.0001))
                token_address = pair.get('baseToken', {}).get('address', symbol)
                volume24h = float(pair.get('volume', {}).get('h24', 0))
                liquidity = float(pair.get('liquidity', {}).get('usd', 0))
        
        onchain_info = get_token_onchain_info(token_address)
        holder_metrics = get_holder_metrics(onchain_info)
        creator_addr = onchain_info.get("creator")
        creator_history = get_creator_history(creator_addr)
        result = predict_trend(current_price, volume24h, liquidity, token_address, holder_metrics, creator_history)
        
        history = []
        base_time = datetime.now()
        
        for i in range(100, 0, -1):
            timestamp = (base_time - timedelta(minutes=i * 5)).isoformat()
            variation = random.uniform(-0.05, 0.05)
            price = current_price * (1 + variation)
            history.append({'time': timestamp, 'price': price})
        
        history.append({'time': base_time.isoformat(), 'price': current_price})
        
        future = []
        prediction = result['prediction']
        
        if 'AVOID' in prediction or 'RUG' in prediction:
            base_trend = -0.015
            volatility = 0.03
        elif '10x+' in prediction:
            base_trend = 0.012
            volatility = 0.025
        elif '5x' in prediction:
            base_trend = 0.008
            volatility = 0.02
        elif '2x' in prediction:
            base_trend = 0.005
            volatility = 0.018
        elif '30%+' in prediction:
            base_trend = 0.003
            volatility = 0.015
        else:
            base_trend = 0.0
            volatility = 0.02
        
        last_price = current_price
        for i in range(1, 13):
            future_time = (base_time + timedelta(minutes=i * 5)).isoformat()
            trend_component = base_trend * i * 0.4
            noise = random.uniform(-volatility * 1.5, volatility * 1.5)
            
            if i > 1:
                price_change = (future[-1]['price'] - last_price) / last_price
                momentum = price_change * 0.5
                oscillation = 0.01 * random.choice([-1, 1]) * (i % 3)
            else:
                momentum = 0
                oscillation = 0
            
            future_price = current_price * (1 + trend_component + noise + momentum + oscillation)
            future_price = max(future_price, current_price * 0.5)
            
            future.append({'time': future_time, 'price': future_price})
            last_price = future_price
        
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
                       'price': fallback_price * (1 + random.uniform(-0.02, 0.02) * i)}
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
                return {'price': data['solana']['usd'], 'change_24h': data['solana'].get('usd_24h_change', 0)}
        
        dex_response = requests.get('https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112', timeout=10)
        if dex_response.status_code == 200:
            dex_data = dex_response.json()
            if dex_data.get('pairs'):
                pair = dex_data['pairs'][0]
                price = float(pair.get('priceUsd', 0))
                change = float(pair.get('priceChange', {}).get('h24', 0))
                if price > 0:
                    return {'price': price, 'change_24h': change}
        
        binance_response = requests.get('https://api.binance.com/api/v3/ticker/24hr?symbol=SOLUSDT', timeout=10)
        if binance_response.status_code == 200:
            binance_data = binance_response.json()
            return {'price': float(binance_data['lastPrice']), 'change_24h': float(binance_data['priceChangePercent'])}
        
        return {'error': 'Failed to fetch price from all sources'}
    
    except Exception as e:
        print(f"Solana price error: {str(e)}")
        return {'error': str(e)}
# ===== STOCK ANALYST FUNCTIONS =====

def get_stock_data(ticker: str) -> dict:
    """Fetch stock price, volume, and market data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1d")
        
        if hist.empty:
            return None
        
        current_price = hist['Close'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        
        return {
            "price": float(current_price),
            "volume": float(volume),
            "market_cap": info.get("marketCap", 0),
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown")
        }
    except Exception as e:
        print(f"Stock data error for {ticker}: {e}")
        return None


def get_stock_fundamentals(ticker: str) -> dict:
    """Fetch stock fundamentals (P/E, EPS, revenue, debt)"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "pe_ratio": info.get("trailingPE", 0),
            "forward_pe": info.get("forwardPE", 0),
            "eps": info.get("trailingEps", 0),
            "revenue_growth": info.get("revenueGrowth", 0),
            "profit_margin": info.get("profitMargins", 0),
            "debt_to_equity": info.get("debtToEquity", 0),
            "return_on_equity": info.get("returnOnEquity", 0),
            "analyst_rating": info.get("recommendationKey", "none"),
            "target_price": info.get("targetMeanPrice", 0)
        }
    except Exception as e:
        print(f"Fundamentals error for {ticker}: {e}")
        return {}


def get_stock_technicals(ticker: str) -> dict:
    """Calculate technical indicators (RSI, moving averages)"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        if len(hist) < 14:
            return {}
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        ma_200 = hist['Close'].rolling(window=min(200, len(hist))).mean().iloc[-1]
        current_price = hist['Close'].iloc[-1]
        
        return {
            "rsi": float(current_rsi) if not np.isnan(current_rsi) else 50,
            "ma_50": float(ma_50) if not np.isnan(ma_50) else current_price,
            "ma_200": float(ma_200) if not np.isnan(ma_200) else current_price,
            "price_vs_ma50": ((current_price - ma_50) / ma_50 * 100) if not np.isnan(ma_50) else 0,
            "price_vs_ma200": ((current_price - ma_200) / ma_200 * 100) if not np.isnan(ma_200) else 0
        }
    except Exception as e:
        print(f"Technicals error for {ticker}: {e}")
        return {}


def stock_risk_gate(fundamentals: dict, technicals: dict, stock_data: dict) -> tuple:
    """Evaluate stock risk based on fundamentals and technicals (UPDATED: more lenient for growth stocks)"""
    reasons = []
    high_risk = False
    
    pe_ratio = fundamentals.get("pe_ratio", 0)
    revenue_growth = fundamentals.get("revenue_growth", 0)
    debt_to_equity = fundamentals.get("debt_to_equity", 0)
    profit_margin = fundamentals.get("profit_margin", 0)
    rsi = technicals.get("rsi", 50)
    
    # P/E ratio checks (more lenient for growth stocks)
    if pe_ratio < 0:
        high_risk = True
        reasons.append("🚨 Negative earnings — company is losing money")
    elif pe_ratio > 200:
        high_risk = True
        reasons.append(f"🚨 Extremely high P/E ratio ({pe_ratio:.1f}) — highly overvalued")
    elif pe_ratio > 100:
        reasons.append(f"⚠️ Very high P/E ratio ({pe_ratio:.1f}) — growth stock or overvalued")
    elif pe_ratio > 50:
        reasons.append(f"⚠️ High P/E ratio ({pe_ratio:.1f}) — premium valuation")
    
    # Revenue growth checks (more important than P/E for growth)
    if revenue_growth < -0.3:
        high_risk = True
        reasons.append(f"🚨 Revenue declining by {abs(revenue_growth)*100:.1f}% — serious trouble")
    elif revenue_growth < -0.1:
        reasons.append(f"⚠️ Revenue declining by {abs(revenue_growth)*100:.1f}% — watch carefully")
    
    # Debt checks (more lenient)
    if debt_to_equity > 3:
        high_risk = True
        reasons.append(f"🚨 Excessive debt-to-equity ratio ({debt_to_equity:.2f}) — overleveraged")
    elif debt_to_equity > 2:
        reasons.append(f"⚠️ High debt-to-equity ratio ({debt_to_equity:.2f})")
    
    # Profitability checks (only flag if deeply negative)
    if profit_margin < -0.2:
        high_risk = True
        reasons.append(f"🚨 Large negative profit margin ({profit_margin*100:.1f}%) — burning cash rapidly")
    elif profit_margin < -0.05:
        reasons.append(f"⚠️ Negative profit margin ({profit_margin*100:.1f}%) — not yet profitable")
    
    # Technical checks
    if rsi > 85:
        reasons.append(f"⚠️ RSI extremely overbought ({rsi:.1f}) — possible pullback")
    elif rsi < 15:
        reasons.append(f"✅ RSI extremely oversold ({rsi:.1f}) — possible bounce opportunity")
    
    return high_risk, reasons


def predict_stock_trend(ticker: str, stock_data: dict, fundamentals: dict, technicals: dict) -> dict:
    """Predict stock trend based on fundamentals and technicals"""
    price = stock_data["price"]
    high_risk, reasons = stock_risk_gate(fundamentals, technicals, stock_data)
    
    if high_risk:
        reasoning = f"""🔴 INVESTMENT SCORE: 0/17

🚨 PREDICTION: AVOID THIS STOCK
⚠️ CONFIDENCE: HIGH RISK

{chr(10).join(reasons)}

🛑 RECOMMENDATION: Do NOT invest in this stock."""
        return {
            'prediction': '🚨 AVOID - HIGH RISK STOCK',
            'confidence': 0,
            'reasoning': reasoning,
            'target_price': price * 0.9,
            'stop_loss': price * 0.85,
            'position_type': '🔻 SHORT CANDIDATE',
            'position_reasoning': 'High risk fundamentals suggest shorting opportunity or avoid entirely.'
        }
    
    score = 0
    reasoning_parts = []
    
    pe_ratio = fundamentals.get("pe_ratio", 0)
    if 0 < pe_ratio < 15:
        score += 4
        reasoning_parts.append(f"✅ Low P/E ratio ({pe_ratio:.1f}) — undervalued (+4)")
    elif 15 <= pe_ratio < 25:
        score += 3
        reasoning_parts.append(f"✅ Reasonable P/E ratio ({pe_ratio:.1f}) — fair value (+3)")
    elif 25 <= pe_ratio < 35:
        score += 1
        reasoning_parts.append(f"⚡ Moderate P/E ratio ({pe_ratio:.1f}) (+1)")
    else:
        reasoning_parts.append(f"⚠️ P/E ratio ({pe_ratio:.1f}) — expensive (0)")
    
    revenue_growth = fundamentals.get("revenue_growth", 0) * 100
    if revenue_growth > 30:
        score += 4
        reasoning_parts.append(f"✅ Strong revenue growth ({revenue_growth:.1f}%) — excellent momentum (+4)")
    elif revenue_growth > 15:
        score += 3
        reasoning_parts.append(f"✅ Good revenue growth ({revenue_growth:.1f}%) (+3)")
    elif revenue_growth > 5:
        score += 2
        reasoning_parts.append(f"⚡ Positive revenue growth ({revenue_growth:.1f}%) (+2)")
    else:
        reasoning_parts.append(f"⚠️ Low/negative revenue growth ({revenue_growth:.1f}%) (0)")
    
    profit_margin = fundamentals.get("profit_margin", 0) * 100
    if profit_margin > 20:
        score += 3
        reasoning_parts.append(f"✅ High profit margin ({profit_margin:.1f}%) — very profitable (+3)")
    elif profit_margin > 10:
        score += 2
        reasoning_parts.append(f"✅ Good profit margin ({profit_margin:.1f}%) (+2)")
    elif profit_margin > 5:
        score += 1
        reasoning_parts.append(f"⚡ Acceptable profit margin ({profit_margin:.1f}%) (+1)")
    else:
        reasoning_parts.append(f"⚠️ Low profit margin ({profit_margin:.1f}%) (0)")
    
    rsi = technicals.get("rsi", 50)
    if 30 <= rsi <= 50:
        score += 3
        reasoning_parts.append(f"✅ RSI in buy zone ({rsi:.1f}) — good entry point (+3)")
    elif 50 < rsi <= 60:
        score += 2
        reasoning_parts.append(f"⚡ RSI neutral ({rsi:.1f}) (+2)")
    elif 60 < rsi <= 70:
        score += 1
        reasoning_parts.append(f"⚡ RSI elevated ({rsi:.1f}) — caution (+1)")
    else:
        reasoning_parts.append(f"⚠️ RSI extreme ({rsi:.1f}) — overbought/oversold (0)")
    
    debt_to_equity = fundamentals.get("debt_to_equity", 0)
    if debt_to_equity < 0.5:
        score += 2
        reasoning_parts.append(f"✅ Low debt-to-equity ({debt_to_equity:.2f}) — strong balance sheet (+2)")
    elif debt_to_equity < 1:
        score += 1
        reasoning_parts.append(f"⚡ Moderate debt-to-equity ({debt_to_equity:.2f}) (+1)")
    else:
        reasoning_parts.append(f"⚠️ High debt-to-equity ({debt_to_equity:.2f}) (0)")
    
    rating = fundamentals.get("analyst_rating", "none")
    if rating in ["strong_buy", "buy"]:
        score += 1
        reasoning_parts.append(f"✅ Analyst rating: {rating.replace('_', ' ').upper()} (+1)")
    else:
        reasoning_parts.append(f"⚠️ Analyst rating: {rating.replace('_', ' ').upper()} (0)")
    
    score = max(0, min(17, score))
    
    if score >= 15:
        prediction_text = "🔥 STRONG BUY"
        confidence_level = "VERY HIGH CONFIDENCE"
        target_mult = 1.5
    elif score >= 12:
        prediction_text = "🚀 BUY"
        confidence_level = "HIGH CONFIDENCE"
        target_mult = 1.3
    elif score >= 9:
        prediction_text = "⚡ MODERATE BUY"
        confidence_level = "MODERATE CONFIDENCE"
        target_mult = 1.15
    elif score >= 6:
        prediction_text = "📊 HOLD"
        confidence_level = "LOW CONFIDENCE"
        target_mult = 1.05
    else:
        prediction_text = "⚠️ WEAK / HOLD"
        confidence_level = "LOW CONFIDENCE"
        target_mult = 1.0
    
    # Determine position type (LONG vs SHORT)
    if score < 4:
        position_type = "🔻 SHORT CANDIDATE"
        position_reasoning = "Weak fundamentals and declining metrics suggest shorting opportunity."
    elif score >= 12:
        position_type = "🔼 LONG (BUY & HOLD)"
        position_reasoning = "Strong fundamentals and technicals support long position with conviction."
    elif score >= 9:
        position_type = "🔼 LONG (MODERATE)"
        position_reasoning = "Good fundamentals support long position; consider staged entry."
    elif score >= 6:
        position_type = "➡️ LONG (CAUTIOUS)"
        position_reasoning = "Moderate fundamentals; suitable for long but with tight stops."
    else:
        position_type = "⏸️ NEUTRAL / WAIT"
        position_reasoning = "Mixed signals; wait for clearer setup before entering position."
    
    if target_mult >= 1.5:
        recommendation = "✅ RECOMMENDATION: Strong buy with 50%+ upside potential. Consider position sizing."
    elif target_mult >= 1.3:
        recommendation = "✅ RECOMMENDATION: Good buy opportunity with 30%+ upside."
    elif target_mult >= 1.15:
        recommendation = "⚡ RECOMMENDATION: Moderate buy for diversification."
    elif target_mult >= 1.05:
        recommendation = "⚠️ RECOMMENDATION: Hold current position or wait for better entry."
    else:
        recommendation = "❌ RECOMMENDATION: Weak fundamentals, consider alternatives."
    
    reasoning_output = f"""📊 INVESTMENT SCORE: {score}/17

🎯 PREDICTION: {prediction_text}
💪 CONFIDENCE: {confidence_level}

{chr(10).join(reasoning_parts)}

{recommendation}"""
    
    target_price = fundamentals.get("target_price", price * target_mult)
    if target_price == 0 or target_price is None:
        target_price = price * target_mult
    
    return {
        'prediction': prediction_text,
        'confidence': int((score / 17) * 100),
        'reasoning': reasoning_output,
        'target_price': target_price,
        'stop_loss': price * 0.92,
        'position_type': position_type,
        'position_reasoning': position_reasoning
    }


@app.get("/api/predict-stock")
async def predict_stock(ticker: str):
    """Stock prediction endpoint"""
    try:
        ticker = ticker.upper().strip()
        
        stock_data = get_stock_data(ticker)
        if not stock_data:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        fundamentals = get_stock_fundamentals(ticker)
        technicals = get_stock_technicals(ticker)
        result = predict_stock_trend(ticker, stock_data, fundamentals, technicals)
        
        # Send Discord notification for stock
        send_stock_discord_notification(
            ticker=ticker,
            company_name=stock_data['name'],
            price=stock_data['price'],
            prediction=result['prediction'],
            sector=stock_data['sector'],
            position_type=result['position_type']
        )
        
        return {
            'ticker': ticker,
            'name': stock_data['name'],
            'price': stock_data['price'],
            'volume': stock_data['volume'],
            'market_cap': stock_data['market_cap'],
            'sector': stock_data['sector'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'reasoning': result['reasoning'],
            'target_price': result['target_price'],
            'stop_loss': result['stop_loss'],
            'position_type': result['position_type'],
            'position_reasoning': result['position_reasoning'],
            'fundamentals': fundamentals,
            'technicals': technicals
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing stock: {str(e)}")


@app.get("/api/stock-history")
async def get_stock_history(ticker: str):
    """Get stock price history and predictions"""
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        current_price = hist['Close'].iloc[-1]
        
        stock_data = get_stock_data(ticker)
        fundamentals = get_stock_fundamentals(ticker)
        technicals = get_stock_technicals(ticker)
        result = predict_stock_trend(ticker, stock_data, fundamentals, technicals)
        
        history = []
        for index, row in hist.tail(60).iterrows():
            history.append({'time': index.isoformat(), 'price': float(row['Close'])})
        
        future = []
        prediction = result['prediction']
        base_time = datetime.now()
        
        if 'AVOID' in prediction or 'WEAK' in prediction:
            base_trend = -0.002
            volatility = 0.015
        elif 'STRONG BUY' in prediction:
            base_trend = 0.008
            volatility = 0.012
        elif 'BUY' in prediction:
            base_trend = 0.005
            volatility = 0.01
        elif 'MODERATE' in prediction:
            base_trend = 0.003
            volatility = 0.01
        else:
            base_trend = 0.001
            volatility = 0.008
        
        last_price = current_price
        for i in range(1, 13):
            future_time = (base_time + timedelta(days=i * 2)).isoformat()
            trend_component = base_trend * i * 0.5
            noise = random.uniform(-volatility * 1.2, volatility * 1.2)
            
            if i > 1:
                price_change = (future[-1]['price'] - last_price) / last_price
                momentum = price_change * 0.4
                oscillation = 0.005 * random.choice([-1, 1]) * (i % 3)
            else:
                momentum = 0
                oscillation = 0
            
            future_price = current_price * (1 + trend_component + noise + momentum + oscillation)
            future_price = max(future_price, current_price * 0.7)
            
            future.append({'time': future_time, 'price': future_price})
            last_price = future_price
        
        all_prices = [p['price'] for p in history] + [p['price'] for p in future]
        
        return {
            'history': history,
            'future': future,
            'high_prediction': max(all_prices),
            'low_prediction': min(all_prices)
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Stock history error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/market-indices")
async def get_market_indices():
    """Get major market indices (S&P 500, NASDAQ, Dow Jones)"""
    try:
        indices = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones"}
        results = []
        
        for symbol, name in indices.items():
            try:
                index = yf.Ticker(symbol)
                hist = index.history(period="5d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change = ((current_price - prev_price) / prev_price) * 100
                    results.append({'name': name, 'price': float(current_price), 'change': float(change)})
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                continue
        
        return {'indices': results}
    
    except Exception as e:
        print(f"Market indices error: {str(e)}")
        return {'indices': []}


@app.get("/api/trending-stocks")
async def get_trending_stocks():
    """Get trending/popular stocks"""
    try:
        trending = [
            {"ticker": "AAPL", "name": "Apple"},
            {"ticker": "TSLA", "name": "Tesla"},
            {"ticker": "NVDA", "name": "NVIDIA"},
            {"ticker": "MSFT", "name": "Microsoft"},
            {"ticker": "GOOGL", "name": "Google"},
            {"ticker": "AMZN", "name": "Amazon"}
        ]
        
        results = []
        for stock in trending:
            try:
                ticker_obj = yf.Ticker(stock['ticker'])
                hist = ticker_obj.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    results.append({'ticker': stock['ticker'], 'name': stock['name'], 'price': f"${price:.2f}"})
            except:
                continue
        
        return {'stocks': results}
    
    except Exception as e:
        print(f"Trending stocks error: {str(e)}")
        return {'stocks': []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
