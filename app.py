from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import requests
import pickle
import numpy as np
from datetime import datetime, timedelta
import random
import base64
from openai import OpenAI
import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from typing import Optional
import secrets
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# API Keys
BITQUERY_API_KEY = "ory_at_f1B3dQRfIiJSDEKQOkxr4OXXQ1tMwcMN6CQuIWjevc4.4ySJCw0ZUx-zS5nXnJUXRY59X9NXR6uWf_RnEaNvlqc"
MORALIS_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6ImU0ZGQzYzQyLWIyYjgtNDNkZC1iZmE4LTgzMmU3NTgzNzM3YiIsIm9yZ0lkIjoiNDA5MjA3IiwidXNlcklkIjoiNDIwNTY5IiwidHlwZUlkIjoiNjljNzBmMzYtNzBjMS00OTVlLThkNzAtYjM2NzRlMzFjYzExIiwidHlwZSI6IlBST0pFQ1QiLCJpYXQiOjE3MzAwNzQ2MDUsImV4cCI6NDg4NTgzNDYwNX0.ZHXgLyqMR9ijN-vKFxzxgwf0WPKJXcmdsFQCZsDIzOI"
OPENAI_API_KEY = "sk-proj-mz9TE9TCZnsq66V3O-C1M1JjD80Q92tsEEu4WJutZcjkqSKCf_yN8Cy3FdH-4DafD56-YxBvzfT3BlbkFJwNc0wDdGkEKpD6wvRcO8K-CqmIY4Kz1DVPJHNy-oi5z_zNgjw4P4zMuOSk-cC9XQ19fqisA"
SOLSCAN_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjcmVhdGVkQXQiOjE3NjgxMzcwODYwMTYsImVtYWlsIjoic29jY2VyYWxleGRva29AZ21haWwuY29tIiwiYWN0aW9uIjoidG9rZW4tYXBpIiwiYXBpVmVyc2lvbiI6InYyIiwiaWF0IjoxNzY4MTM3MDg2fQ.df2kEcUDB_Ti_UKv6gaiJ8CERFlsBpiQ8XIuLEdb4XE"
HELIUS_API_KEY = "aa25304b-753b-466b-ad17-598a69c0cb7c"
HELIUS_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
FINNHUB_API_KEY = "d5jqh61r01qjaedr7460"

DISCORD_WEBHOOK_CRYPTO = "https://discord.com/api/webhooks/1437292750960594975/2EHZkITnwOC3PwG-h1es1hokmehqlcvUpP6QJPMsIdMjI54YZtP0NdNyEzuE-CCwbRF5"
DISCORD_WEBHOOK_STOCK = "https://discord.com/api/webhooks/1460815556130246729/7yfC-1AAJ51T9aVrtcU0cNQBxfZXLl177kNMiSVJfd6bamVHG-4u4VRJAPh8d94wlK1s"

SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

client = OpenAI(api_key=OPENAI_API_KEY)

try:
    with open('solana_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception:
    model = None
    print("Warning: Model file not found")

# ======================= DATABASE SETUP =======================

SQLALCHEMY_DATABASE_URL = "sqlite:///./market_analyst.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    wallet_address = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    analyses = relationship("AnalysisHistory", back_populates="user")


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    analysis_type = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    name = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    prediction = Column(String, nullable=False)
    confidence = Column(Integer, nullable=True)
    position_type = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="analyses")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security),
                     db: Session = Depends(get_db)) -> User:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if not credentials:
        return None
    try:
        return get_current_user(credentials, db)
    except HTTPException:
        return None


def assert_is_dev(current_user: User):
    # Simple dev check; adjust to whatever rule you want
    if current_user.email != "dev@memecoinmetrics.com":
        raise HTTPException(status_code=403, detail="Admin access only")


# ======================= AUTH SCHEMAS =======================

class SignupRequest(BaseModel):
    email: str
    username: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


# ======================= AUTH ENDPOINTS =======================

@app.post("/api/auth/signup")
async def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    email = payload.email
    username = payload.username
    password = payload.password
    try:
        existing_user = db.query(User).filter(
            (User.email == email) | (User.username == username)
        ).first()

        if existing_user:
            if existing_user.email == email:
                raise HTTPException(status_code=400, detail="Email already registered")
            else:
                raise HTTPException(status_code=400, detail="Username already taken")

        hashed_password = get_password_hash(password)
        new_user = User(email=email, username=username, hashed_password=hashed_password)

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        access_token = create_access_token(data={"sub": email})

        return {
            "message": "Account created successfully",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "email": new_user.email,
                "username": new_user.username,
                "wallet_address": new_user.wallet_address,
            },
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Signup error: {str(e)}")


@app.post("/api/auth/login")
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email
    password = payload.password
    try:
        user = db.query(User).filter(User.email == email).first()

        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Incorrect email or password")

        access_token = create_access_token(data={"sub": email})

        return {
            "message": "Login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "email": user.email,
                "username": user.username,
                "wallet_address": user.wallet_address,
            },
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")


@app.post("/api/auth/connect-wallet")
async def connect_wallet(
    wallet_address: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        existing_wallet = db.query(User).filter(User.wallet_address == wallet_address).first()
        if existing_wallet and existing_wallet.id != current_user.id:
            raise HTTPException(status_code=400, detail="Wallet already connected to another account")

        current_user.wallet_address = wallet_address
        db.commit()

        return {
            "message": "Wallet connected successfully",
            "wallet_address": wallet_address,
            "user": {
                "email": current_user.email,
                "username": current_user.username,
                "wallet_address": current_user.wallet_address,
            },
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Wallet connection error: {str(e)}")


@app.get("/api/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "username": current_user.username,
        "wallet_address": current_user.wallet_address,
        "created_at": current_user.created_at.isoformat(),
    }


@app.get("/api/user/history")
async def get_user_history(
    limit: int = 50,
    analysis_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        query = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == current_user.id)

        if analysis_type:
            query = query.filter(AnalysisHistory.analysis_type == analysis_type)

        analyses = query.order_by(AnalysisHistory.timestamp.desc()).limit(limit).all()

        return {
            "total": len(analyses),
            "analyses": [
                {
                    "id": a.id,
                    "type": a.analysis_type,
                    "symbol": a.symbol,
                    "name": a.name,
                    "price": a.price,
                    "prediction": a.prediction,
                    "confidence": a.confidence,
                    "position_type": a.position_type,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in analyses
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@app.get("/api/user/stats")
async def get_user_stats(current_user: User = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    try:
        total_analyses = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == current_user.id
        ).count()
        crypto_analyses = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == current_user.id,
            AnalysisHistory.analysis_type == "crypto",
        ).count()
        stock_analyses = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == current_user.id,
            AnalysisHistory.analysis_type == "stock",
        ).count()

        recent_analyses = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == current_user.id
        ).order_by(AnalysisHistory.timestamp.desc()).limit(100).all()

        symbol_counts = {}
        for analysis in recent_analyses:
            symbol_counts[analysis.symbol] = symbol_counts.get(analysis.symbol, 0) + 1

        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_searches": total_analyses,
            "crypto_searches": crypto_analyses,
            "stock_searches": stock_analyses,
            "top_symbols": [{"symbol": s[0], "count": s[1]} for s in top_symbols],
            "member_since": current_user.created_at.isoformat(),
            "wallet_connected": current_user.wallet_address is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@app.get("/api/admin/users")
async def get_all_users(current_user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    """Admin: Get all registered users (dev only)"""
    assert_is_dev(current_user)
    try:
        users = db.query(User).order_by(User.created_at.desc()).all()

        return {
            "total_users": len(users),
            "users": [
                {
                    "id": u.id,
                    "username": u.username,
                    "email": u.email,
                    "wallet_address": u.wallet_address,
                    "created_at": u.created_at.isoformat(),
                    "total_analyses": len(u.analyses),
                }
                for u in users
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")


@app.get("/api/admin/analyses")
async def get_all_analyses(
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Admin: Get all user analyses (dev only)"""
    assert_is_dev(current_user)
    try:
        analyses = db.query(AnalysisHistory).order_by(
            AnalysisHistory.timestamp.desc()
        ).limit(limit).all()

        return {
            "total": len(analyses),
            "analyses": [
                {
                    "id": a.id,
                    "user_id": a.user_id,
                    "username": a.user.username if a.user else "Unknown",
                    "type": a.analysis_type,
                    "symbol": a.symbol,
                    "name": a.name,
                    "price": a.price,
                    "prediction": a.prediction,
                    "confidence": a.confidence,
                    "position_type": a.position_type,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in analyses
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analyses: {str(e)}")


def save_analysis_to_history(
    user: Optional[User],
    analysis_type: str,
    symbol: str,
    name: str,
    price: float,
    prediction: str,
    confidence: int,
    position_type: str = None,
    db: Session = None,
):
    if not user or not db:
        return

    try:
        analysis = AnalysisHistory(
            user_id=user.id,
            analysis_type=analysis_type,
            symbol=symbol,
            name=name,
            price=price,
            prediction=prediction,
            confidence=confidence,
            position_type=position_type,
        )
        db.add(analysis)
        db.commit()
    except Exception as e:
        print(f"Failed to save analysis to history: {e}")
        db.rollback()

# ======================= CRYPTO FUNCTIONS =======================

def get_token_onchain_info(mint_address: str) -> dict:
    url = "https://pro-api.solscan.io/v2.0/token/holdersv2"
    params = {"address": mint_address, "page": 1, "page_size": 50}
    headers = {"accept": "application/json", "token": SOLSCAN_API_KEY}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
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
            "total_supply": data.get("total", 0),
        }
    except Exception:
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
        "lp_locked": lp_locked,
    }


def get_dexscreener_chart_url(mint_address: str) -> str:
    return f"https://dexscreener.com/solana/{mint_address}"


def get_creator_history(creator_address: Optional[str]) -> Optional[dict]:
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

            try:
                chart_url = get_dexscreener_chart_url(mint)
                chart_response = requests.get(chart_url, timeout=10)
                if chart_response.status_code != 200:
                    continue

                image_base64 = base64.b64encode(chart_response.content).decode("utf-8")
                vision_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Analyze this cryptocurrency chart and classify it:\n\n"
                                        "Does this show a PUMP-AND-DUMP / RUG PULL pattern?\n\n"
                                        "Rug indicators:\n"
                                        "- Parabolic spike followed by 80%+ drop\n"
                                        "- Sudden liquidity drain\n"
                                        "- Volume spike then dead volume\n"
                                        "- No recovery after crash\n\n"
                                        "Answer with: RUG: YES or RUG: NO\n\n"
                                        "Then briefly explain why in 1-2 sentences."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=200,
                )

                analysis = vision_response.choices[0].message.content
                if "RUG: YES" in analysis or (
                    "pump" in analysis.lower() and "dump" in analysis.lower()
                ):
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


def risk_gate(
    price: float,
    volume24h: float,
    liquidity: float,
    holder_metrics: Optional[dict] = None,
    creator_history: Optional[dict] = None,
):
    reasons = []
    high_risk = False
    vol_liq_ratio = volume24h / liquidity if liquidity > 0 else 0

    if vol_liq_ratio > 5 and liquidity < 50000:
        high_risk = True
        reasons.append(
            "🚨 EXTREME volume/liquidity ratio with low liquidity — likely pump scheme"
        )
    elif liquidity < 30000 and volume24h > 100000:
        high_risk = True
        reasons.append(
            "🚨 Very low liquidity with high volume — potential rug pull risk"
        )
    elif liquidity < 10000 and volume24h > 50000:
        high_risk = True
        reasons.append("🚨 Critically low liquidity — high rug pull risk")

    if holder_metrics:
        dev = holder_metrics.get("dev_hold_pct", 0)
        top5 = holder_metrics.get("top5_pct", 0)
        lp_locked = holder_metrics.get("lp_locked", True)

        if dev >= 0.09:
            high_risk = True
            reasons.append(
                f"🚨 Developer holds ~{dev*100:.1f}% of supply — strong market control risk"
            )
        elif dev >= 0.05:
            reasons.append(
                f"⚠️ Developer holds ~{dev*100:.1f}% of supply — elevated control risk"
            )

        if top5 >= 0.50:
            high_risk = True
            reasons.append(
                f"🚨 Top 5 wallets hold {top5*100:.1f}% of supply — whale concentration"
            )
        elif top5 >= 0.40:
            reasons.append(
                f"⚠️ Top 5 wallets hold {top5*100:.1f}% of supply — watch whale activity"
            )

        if not lp_locked:
            high_risk = True
            reasons.append("🚨 Liquidity is not locked — common rug‑pull pattern")

    if creator_history:
        rug_rate = creator_history.get("rug_rate", 0)
        rugged_tokens = creator_history.get("rugged_tokens", 0)
        total_tokens = creator_history.get("total_tokens", 0)

        if total_tokens >= 2 and rug_rate >= 0.5:
            high_risk = True
            reasons.append(
                f"🚨 Creator has rugged {rugged_tokens}/{total_tokens} previous tokens ({rug_rate*100:.0f}% rug rate based on chart analysis)."
            )
        elif rugged_tokens >= 1 and rug_rate >= 0.25:
            reasons.append(
                f"⚠️ Creator has prior rug history: {rugged_tokens}/{total_tokens} tokens showed rug patterns."
            )

    return high_risk, reasons, vol_liq_ratio


def send_discord_notification(
    symbol: str,
    token_name: Optional[str] = None,
    price: Optional[float] = None,
    prediction: Optional[str] = None,
    volume24h: Optional[float] = None,
    liquidity: Optional[float] = None,
):
    try:
        color = 5814783
        if prediction and "10x+ GAIN" in prediction:
            color = 5763719
        elif prediction and "5x GAIN" in prediction:
            color = 5763719
        elif prediction and "2x GAIN" in prediction:
            color = 16776960
        elif prediction and (
            "LIMITED UPSIDE" in prediction
            or "AVOID" in prediction
            or "RUG PULL" in prediction
        ):
            color = 15548997

        embed = {
            "title": "🔍 New Crypto Search",
            "color": color,
            "fields": [
                {"name": "🪙 Symbol", "value": symbol, "inline": True},
                {
                    "name": "🕒 Time",
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S EST"),
                    "inline": True,
                },
            ],
            "footer": {"text": "Solana Memecoin Predictor"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        if token_name:
            embed["fields"].insert(
                1, {"name": "📛 Token Name", "value": token_name, "inline": False}
            )
        if price:
            embed["fields"].append(
                {"name": "💰 Price", "value": f"${price:.8f}", "inline": True}
            )
        if volume24h:
            embed["fields"].append(
                {"name": "📊 Volume 24h", "value": f"${volume24h:,.0f}", "inline": True}
            )
        if liquidity:
            embed["fields"].append(
                {"name": "💧 Liquidity", "value": f"${liquidity:,.0f}", "inline": True}
            )
        if prediction:
            embed["fields"].append(
                {"name": "🎯 Prediction", "value": prediction, "inline": False}
            )

        payload = {"embeds": [embed]}
        requests.post(DISCORD_WEBHOOK_CRYPTO, json=payload, timeout=5)
    except Exception as e:
        print(f"Discord crypto notification failed: {e}")


def analyze_chart_image(chart_url: str) -> str:
    try:
        response = requests.get(chart_url, timeout=10)
        if response.status_code != 200:
            return "❌ Unable to fetch chart image for analysis."

        image_base64 = base64.b64encode(response.content).decode("utf-8")
        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this cryptocurrency chart and provide:\n"
                                "1. Pattern identification (pump/dump, accumulation, breakout, consolidation, etc.)\n"
                                "2. Trend direction (bullish/bearish/neutral)\n"
                                "3. Key support and resistance levels\n"
                                "4. Volume trend analysis\n"
                                "5. Risk level (1-10 scale)\n"
                                "6. Whether this shows multi‑X opportunity potential (YES/NO)\n\n"
                                "Keep analysis concise and actionable."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        analysis = vision_response.choices[0].message.content
        return f"📊 VISUAL CHART ANALYSIS:\n{analysis}"
    except Exception as e:
        return f"❌ Chart analysis unavailable: {str(e)}"


def predict_trend(
    price: float,
    volume24h: float,
    liquidity: float,
    mint_address: Optional[str] = None,
    holder_metrics: Optional[dict] = None,
    creator_history: Optional[dict] = None,
) -> dict:
    high_risk, reasons, vol_liq_ratio = risk_gate(
        price, volume24h, liquidity, holder_metrics, creator_history
    )

    if high_risk:
        reasoning = f"""🔴 GAIN POTENTIAL SCORE: 0/17

🚨 PREDICTION: AVOID THIS TOKEN
⚠️ CONFIDENCE: HIGH CHANCE OF RUG PULL

{chr(10).join(reasons)}

⚠️ Volume/Liquidity Ratio: {vol_liq_ratio:.2f}
❌ This token shows strong rug‑pull / manipulation characteristics

🛑 RECOMMENDATION: Do NOT enter this trade."""
        return {
            "prediction": "🚨 AVOID - HIGH CHANCE OF RUG PULL",
            "confidence": 0,
            "reasoning": reasoning,
            "highest_price": price * 1.05,
            "lowest_price": price * 0.70,
            "chart_analysis": "",
        }

    vol_liq_ratio = volume24h / liquidity if liquidity > 0 else 0
    gain_score = 0
    reasoning_parts = []

    if price < 0.00001:
        gain_score += 4
        reasoning_parts.append(
            f"✅ Ultra-low price (${price:.8f}) — micro-cap potential (+4)"
        )
    elif price < 0.0001:
        gain_score += 3
        reasoning_parts.append(
            f"✅ Very low price (${price:.6f}) — good growth room (+3)"
        )
    elif price < 0.001:
        gain_score += 2
        reasoning_parts.append(
            f"✅ Low price (${price:.6f}) — moderate growth potential (+2)"
        )
    elif price < 0.01:
        gain_score += 1
        reasoning_parts.append(
            f"⚡ Low-mid price (${price:.6f}) — some room to grow (+1)"
        )
    else:
        reasoning_parts.append(
            f"⚠️ Higher price (${price:.4f}) — less explosive potential (0)"
        )

    if 1 <= vol_liq_ratio <= 3:
        gain_score += 4
        reasoning_parts.append(
            f"✅ Optimal volume/liquidity ratio ({vol_liq_ratio:.2f}) — healthy trading (+4)"
        )
    elif 0.5 <= vol_liq_ratio < 1:
        gain_score += 2
        reasoning_parts.append(
            f"⚡ Moderate ratio ({vol_liq_ratio:.2f}) — building momentum (+2)"
        )
    elif 3 < vol_liq_ratio <= 5:
        gain_score += 1
        reasoning_parts.append(
            f"⚠️ High ratio ({vol_liq_ratio:.2f}) — watch for volatility (+1)"
        )
    else:
        reasoning_parts.append(
            f"⚠️ Volume/Liquidity Ratio: {vol_liq_ratio:.2f} — TOO HIGH, possible pump scheme."
        )

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
                    reasoning_parts.append(
                        f"✅ ML Model: Strong 'UP' signal ({ml_confidence:.0f}% confidence) (+3)"
                    )
                elif ml_confidence > 50:
                    gain_score += 2
                    reasoning_parts.append(
                        f"⚡ ML Model: 'UP' signal ({ml_confidence:.0f}% confidence) (+2)"
                    )
                else:
                    gain_score += 1
                    reasoning_parts.append(
                        f"⚡ ML Model: Weak 'UP' signal ({ml_confidence:.0f}% confidence) (+1)"
                    )
            elif prediction == 1:
                ml_prediction = "sideways"
                reasoning_parts.append(
                    "⚠️ ML Model: 'SIDEWAYS' — neutral momentum (0)"
                )
            else:
                ml_prediction = "down"
                reasoning_parts.append("❌ ML Model: 'DOWN' signal — bearish (0)")
        except Exception:
            reasoning_parts.append("⚠️ ML Model: unavailable")

    if volume24h > 2_000_000:
        gain_score += 3
        reasoning_parts.append(
            f"✅ Exceptionally high trading volume (${volume24h:,.0f}/24h) — strong momentum (+3)"
        )
    elif volume24h > 1_000_000:
        gain_score += 2
        reasoning_parts.append(
            f"✅ High trading volume (${volume24h:,.0f}/24h) — good momentum (+2)"
        )
    elif volume24h > 500_000:
        gain_score += 1
        reasoning_parts.append(
            f"⚡ Moderate volume (${volume24h:,.0f}/24h) — building interest (+1)"
        )
    else:
        reasoning_parts.append(
            f"⚠️ Low volume (${volume24h:,.0f}/24h) — limited momentum (0)"
        )

    if 30_000 <= liquidity <= 300_000:
        gain_score += 2
        reasoning_parts.append(
            f"✅ Good liquidity (${liquidity:,.0f}) — optimal for big moves (+2)"
        )
    elif 10_000 <= liquidity < 30_000 or 300_000 < liquidity <= 500_000:
        gain_score += 1
        reasoning_parts.append(
            f"⚡ Acceptable liquidity (${liquidity:,.0f}) (+1)"
        )
    else:
        reasoning_parts.append(
            f"⚠️ Liquidity (${liquidity:,.0f}) — outside optimal range (0)"
        )

    if liquidity < 20_000:
        gain_score -= 3
        reasoning_parts.append(
            f"❌ Very low liquidity (${liquidity:,.0f}) — high risk (-3)"
        )
    if volume24h < 50_000:
        gain_score -= 2
        reasoning_parts.append(
            f"❌ Dead volume (${volume24h:,.0f}/24h) — no momentum (-2)"
        )

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
        recommendation = (
            "✅ RECOMMENDATION: High‑conviction degen play; size small and manage risk aggressively."
        )
    elif target_mult >= 5:
        recommendation = (
            "✅ RECOMMENDATION: Strong upside; consider staged entries and profit‑taking levels."
        )
    elif target_mult >= 2:
        recommendation = (
            "⚠️ RECOMMENDATION: Good 2x potential; enter with clear stop loss and TP targets."
        )
    elif target_mult >= 1.3:
        recommendation = (
            "⚠️ RECOMMENDATION: Solid 30%+ setup; suitable for shorter swing trades."
        )
    else:
        recommendation = (
            "❌ RECOMMENDATION: Upside is limited; better opportunities likely elsewhere."
        )

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
        "prediction": prediction_text,
        "confidence": ml_confidence if model else 50,
        "reasoning": reasoning_output,
        "highest_price": price * target_mult,
        "lowest_price": price * max_drop_mult,
        "chart_analysis": chart_analysis,
    }

# ======================= STOCK NEWS & SENTIMENT =======================

def get_stock_news(ticker: str) -> list:
    try:
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news?"
            f"symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        )

        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Finnhub news error: {response.status_code}")
            return []

        news_data = response.json()
        return news_data[:10] if news_data else []
    except Exception as e:
        print(f"News fetch error for {ticker}: {e}")
        return []


def analyze_news_sentiment(ticker: str, news_articles: list) -> dict:
    if not news_articles:
        return {
            "sentiment": "neutral",
            "sentiment_score": 0,
            "key_headlines": [],
            "analysis": "No recent news available.",
        }

    try:
        headlines_text = "\n".join(
            [
                f"{i+1}. {article.get('headline', 'N/A')}"
                for i, article in enumerate(news_articles[:10])
            ]
        )

        gpt_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Analyze these recent news headlines for {ticker} and provide:\n\n"
                        f"{headlines_text}\n\n"
                        "Provide:\n"
                        "1. Overall sentiment (BULLISH / BEARISH / NEUTRAL)\n"
                        "2. Sentiment score (-10 to +10, where -10 is extremely bearish, +10 is extremely bullish)\n"
                        "3. Top 3 most important headlines (list them)\n"
                        "4. Brief analysis (2-3 sentences) of what's happening with this company\n\n"
                        "Format your response as:\n"
                        "SENTIMENT: [BULLISH/BEARISH/NEUTRAL]\n"
                        "SCORE: [number]\n"
                        "KEY HEADLINES:\n"
                        "- [headline 1]\n"
                        "- [headline 2]\n"
                        "- [headline 3]\n"
                        "ANALYSIS: [your analysis]"
                    ),
                }
            ],
            max_tokens=400,
        )

        analysis_text = gpt_response.choices[0].message.content

        sentiment = "neutral"
        if "SENTIMENT: BULLISH" in analysis_text:
            sentiment = "bullish"
        elif "SENTIMENT: BEARISH" in analysis_text:
            sentiment = "bearish"

        sentiment_score = 0
        try:
            score_line = [
                line for line in analysis_text.split("\n") if "SCORE:" in line
            ][0]
            sentiment_score = int(
                score_line.split("SCORE:")[1].strip().split()[0]
            )
        except Exception:
            pass

        key_headlines = []
        try:
            headlines_section = analysis_text.split("KEY HEADLINES:")[1].split(
                "ANALYSIS:"
            )[0]
            key_headlines = [
                line.strip("- ").strip()
                for line in headlines_section.split("\n")
                if line.strip().startswith("-")
            ]
        except Exception:
            key_headlines = [
                article.get("headline", "N/A") for article in news_articles[:3]
            ]

        analysis_summary = "Recent news analyzed."
        try:
            analysis_summary = analysis_text.split("ANALYSIS:")[1].strip()
        except Exception:
            pass

        return {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "key_headlines": key_headlines[:3],
            "analysis": analysis_summary,
        }
    except Exception as e:
        print(f"News sentiment analysis error: {e}")
        return {
            "sentiment": "neutral",
            "sentiment_score": 0,
            "key_headlines": [],
            "analysis": "Unable to analyze news sentiment.",
        }

# ======================= STOCK FUNCTIONS =======================

def get_stock_data(ticker: str) -> Optional[dict]:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        if hist.empty or len(hist) == 0:
            print(f"No history data for {ticker}")
            return None

        current_price = hist["Close"].iloc[-1]
        volume = hist["Volume"].iloc[-1]

        try:
            info = stock.info
            name = info.get("longName") or info.get("shortName") or ticker
            market_cap = info.get("marketCap", 0) or 0
            sector = info.get("sector", "Unknown") or "Unknown"
            industry = info.get("industry", "Unknown") or "Unknown"
        except Exception:
            name = ticker
            market_cap = 0
            sector = "Technology"
            industry = "Unknown"

        return {
            "price": float(current_price),
            "volume": float(volume),
            "market_cap": market_cap,
            "name": name,
            "sector": sector,
            "industry": industry,
        }
    except Exception as e:
        print(f"Stock data error for {ticker}: {e}")
        return None


def get_stock_fundamentals(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE") or 0,
            "forward_pe": info.get("forwardPE", 0) or 0,
            "eps": info.get("trailingEps", 0) or 0,
            "revenue_growth": info.get("revenueGrowth", 0) or 0,
            "profit_margin": info.get("profitMargins", 0) or 0,
            "debt_to_equity": info.get("debtToEquity", 0) or 0,
            "return_on_equity": info.get("returnOnEquity", 0) or 0,
            "analyst_rating": info.get("recommendationKey", "none") or "none",
            "target_price": info.get("targetMeanPrice", 0) or 0,
        }
    except Exception as e:
        print(f"Fundamentals error for {ticker}: {e}")
        return {
            "pe_ratio": 0,
            "forward_pe": 0,
            "eps": 0,
            "revenue_growth": 0,
            "profit_margin": 0,
            "debt_to_equity": 0,
            "return_on_equity": 0,
            "analyst_rating": "none",
            "target_price": 0,
        }


def get_stock_technicals(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")

        if len(hist) < 14:
            return {"rsi": 50, "ma50": 0, "ma200": 0, "trend": "neutral"}

        close_prices = hist["Close"]
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if loss.iloc[-1] != 0 else 50

        ma50 = (
            close_prices.rolling(window=50).mean().iloc[-1]
            if len(close_prices) >= 50
            else close_prices.iloc[-1]
        )
        ma200 = (
            close_prices.rolling(window=200).mean().iloc[-1]
            if len(close_prices) >= 200
            else close_prices.iloc[-1]
        )

        if close_prices.iloc[-1] > ma50 and ma50 > ma200:
            trend = "bullish"
        elif close_prices.iloc[-1] < ma50 and ma50 < ma200:
            trend = "bearish"
        else:
            trend = "neutral"

        return {"rsi": float(rsi), "ma50": float(ma50), "ma200": float(ma200), "trend": trend}
    except Exception as e:
        print(f"Technicals error for {ticker}: {e}")
        return {"rsi": 50, "ma50": 0, "ma200": 0, "trend": "neutral"}


def predict_stock_trend_with_levels(
    ticker: str,
    price: float,
    fundamentals: dict,
    technicals: dict,
    news_sentiment: dict,
) -> dict:
    score = 0
    reasons = []

    pe = fundamentals.get("pe_ratio", 0)
    fwd_pe = fundamentals.get("forward_pe", 0)
    growth = fundamentals.get("revenue_growth", 0)
    margin = fundamentals.get("profit_margin", 0)
    debt_to_equity = fundamentals.get("debt_to_equity", 0)
    roe = fundamentals.get("return_on_equity", 0)
    analyst_rating = fundamentals.get("analyst_rating", "none")
    target_price = fundamentals.get("target_price", 0)

    rsi = technicals.get("rsi", 50)
    trend = technicals.get("trend", "neutral")

    sentiment = news_sentiment.get("sentiment", "neutral")
    sentiment_score = news_sentiment.get("sentiment_score", 0)

    # valuation
    if 10 <= pe <= 30 or 10 <= fwd_pe <= 30:
        score += 2
        reasons.append("✅ Reasonable P/E valuation (+2)")
    elif pe < 0 and fwd_pe > 0:
        score += 1
        reasons.append("⚠️ Negative earnings but improving forward P/E (+1)")
    else:
        reasons.append("⚠️ P/E outside ideal range (0)")

    # growth
    if growth > 0.15:
        score += 3
        reasons.append(f"✅ Strong revenue growth ({growth*100:.1f}%) (+3)")
    elif growth > 0.05:
        score += 2
        reasons.append(f"✅ Moderate revenue growth ({growth*100:.1f}%) (+2)")
    elif growth > 0:
        score += 1
        reasons.append(f"⚡ Low positive revenue growth ({growth*100:.1f}%) (+1)")
    else:
        reasons.append(f"❌ Flat/negative revenue growth ({growth*100:.1f}%) (0)")

    # margins
    if margin > 0.15:
        score += 2
        reasons.append(f"✅ Healthy profit margin ({margin*100:.1f}%) (+2)")
    elif margin > 0.05:
        score += 1
        reasons.append(f"⚡ Thin profit margin ({margin*100:.1f}%) (+1)")
    else:
        reasons.append(f"⚠️ Weak or negative profit margin ({margin*100:.1f}%) (0)")

    # balance sheet
    if 0 < debt_to_equity < 100:
        score += 2
        reasons.append(f"✅ Reasonable leverage (Debt/Equity {debt_to_equity:.0f}) (+2)")
    elif debt_to_equity >= 200:
        score -= 1
        reasons.append(f"❌ High leverage (Debt/Equity {debt_to_equity:.0f}) (-1)")
    else:
        reasons.append(f"⚠️ Leverage profile mixed (Debt/Equity {debt_to_equity:.0f}) (0)")

    # ROE
    if roe > 0.15:
        score += 2
        reasons.append(f"✅ Strong return on equity ({roe*100:.1f}%) (+2)")
    elif roe > 0.05:
        score += 1
        reasons.append(f"⚡ Moderate ROE ({roe*100:.1f}%) (+1)")
    else:
        reasons.append(f"⚠️ Weak ROE ({roe*100:.1f}%) (0)")

    # RSI
    if rsi < 30:
        score += 2
        reasons.append(f"✅ RSI {rsi:.1f} — oversold zone (+2)")
    elif 30 <= rsi <= 70:
        score += 1
        reasons.append(f"⚡ RSI {rsi:.1f} — neutral/healthy (+1)")
    else:
        score -= 1
        reasons.append(f"❌ RSI {rsi:.1f} — overbought zone (-1)")

    # trend
    if trend == "bullish":
        score += 2
        reasons.append("✅ Price above key moving averages — bullish trend (+2)")
    elif trend == "bearish":
        score -= 1
        reasons.append("❌ Price below key moving averages — bearish trend (-1)")
    else:
        reasons.append("⚠️ Mixed/sideways technical trend (0)")

    # news
    if sentiment == "bullish":
        if sentiment_score >= 5:
            score += 3
            reasons.append(
                f"✅ News sentiment: strong bullish (score {sentiment_score}) (+3)"
            )
        else:
            score += 2
            reasons.append(
                f"✅ News sentiment: bullish (score {sentiment_score}) (+2)"
            )
    elif sentiment == "bearish":
        if sentiment_score <= -5:
            score -= 3
            reasons.append(
                f"❌ News sentiment: strongly bearish (score {sentiment_score}) (-3)"
            )
        else:
            score -= 2
            reasons.append(
                f"❌ News sentiment: bearish (score {sentiment_score}) (-2)"
            )
    else:
        reasons.append("⚠️ News sentiment: neutral/unclear (0)")

    # analyst rating
    if analyst_rating in ["strong_buy", "buy"]:
        score += 2
        reasons.append(
            f"✅ Analyst consensus: {analyst_rating.replace('_', ' ').title()} (+2)"
        )
    elif analyst_rating in ["hold"]:
        score += 1
        reasons.append("⚡ Analyst consensus: Hold (+1)")
    elif analyst_rating in ["sell", "strong_sell"]:
        score -= 2
        reasons.append(
            f"❌ Analyst consensus: {analyst_rating.replace('_', ' ').title()} (-2)"
        )
    else:
        reasons.append("⚠️ Analyst rating: not available (0)")

    # analyst target vs current
    if target_price and target_price > 0:
        upside = (target_price - price) / price
        if upside > 0.3:
            score += 2
            reasons.append(
                f"✅ Analyst target implies {upside*100:.0f}% upside (+2)"
            )
        elif upside > 0.1:
            score += 1
            reasons.append(
                f"⚡ Analyst target implies {upside*100:.0f}% upside (+1)"
            )
        elif upside < -0.1:
            score -= 1
            reasons.append(
                f"❌ Analyst target below current price ({upside*100:.0f}% downside) (-1)"
            )
        else:
            reasons.append(
                f"⚠️ Analyst target near current price ({upside*100:.0f}% move) (0)"
            )
    else:
        reasons.append("⚠️ No clear analyst target price (0)")

    max_score = 20
    normalized_score = max(0, min(max_score, score))

    if normalized_score >= 15:
        prediction = "🏆 STRONG BUY"
        position_type = "AGGRESSIVE LONG"
        risk_label = "LOW–MODERATE RISK"
    elif normalized_score >= 11:
        prediction = "✅ BUY"
        position_type = "STANDARD LONG"
        risk_label = "MODERATE RISK"
    elif normalized_score >= 7:
        prediction = "⚖️ HOLD / NEUTRAL"
        position_type = "NEUTRAL / WAIT"
        risk_label = "BALANCED RISK"
    elif normalized_score >= 4:
        prediction = "⚠️ WEAK / TRIM"
        position_type = "REDUCE / AVOID NEW ENTRIES"
        risk_label = "ELEVATED RISK"
    else:
        prediction = "❌ AVOID"
        position_type = "NO POSITION / EXIT"
        risk_label = "HIGH RISK"

    # trade levels + leverage
    direction = "avoid"
    should_buy_now = False
    use_leverage = False
    leverage_side = None
    suggested_leverage = 1.0

    buy_price = price
    risk_pct = 0.03
    stop_loss = buy_price * (1 - risk_pct)
    take_profit = buy_price * (1 + 2 * risk_pct)

    reward = take_profit - buy_price
    risk = buy_price - stop_loss
    rr = reward / risk if risk > 0 else 0

    max_upside_price = buy_price * 1.15
    max_downside_price = buy_price * 0.85

    if target_price and target_price > 0:
        if target_price > max_upside_price:
            max_upside_price = target_price
        elif target_price < max_downside_price:
            max_downside_price = target_price

    if prediction in ("🏆 STRONG BUY", "✅ BUY") and rr >= 2:
        direction = "long"
        should_buy_now = True
        if normalized_score >= 15:
            use_leverage = True
            leverage_side = "long"
            suggested_leverage = 1.5
        else:
            use_leverage = False
            leverage_side = "long"
            suggested_leverage = 1.0
    elif prediction == "⚖️ HOLD / NEUTRAL":
        direction = "long"
        should_buy_now = False
        use_leverage = False
        leverage_side = "long"
        suggested_leverage = 1.0
    else:
        direction = "avoid"
        should_buy_now = False
        use_leverage = False
        leverage_side = None
        suggested_leverage = 1.0

    reasoning_output = f"""INVESTMENT SCORE: {normalized_score}/{max_score}

🎯 PREDICTION: {prediction}
📍 POSITION: {position_type}
⚠️ RISK PROFILE: {risk_label}

🎯 TARGET BUY PRICE: ${buy_price:.2f}
💰 TARGET SELL PRICE: ${take_profit:.2f}
🛡️ STOP LOSS: ${stop_loss:.2f}
📈 MAX UPSIDE PRICE: ${max_upside_price:.2f}
📉 MAX DOWNSIDE PRICE: ${max_downside_price:.2f}

{chr(10).join(reasons)}"""

    return {
        "prediction": prediction,
        "position_type": position_type,
        "score": normalized_score,
        "max_score": max_score,
        "reasoning": reasoning_output,
        "direction": direction,
        "should_buy_now": should_buy_now,
        "buy_price": buy_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "max_upside_price": max_upside_price,
        "max_downside_price": max_downside_price,
        "use_leverage": use_leverage,
        "leverage_side": leverage_side,
        "suggested_leverage": suggested_leverage,
    }

# ======================= CRYPTO PREDICT ENDPOINT =======================

@app.get("/api/predict")
async def predict(
    symbol: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    db: Session = Depends(get_db),
):
    try:
        user: Optional[User] = None
        if credentials:
            try:
                user = get_current_user(credentials, db)
            except HTTPException:
                user = None

        search_url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}"
        response = requests.get(search_url, timeout=10)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch token data")

        data = response.json()
        pairs = data.get("pairs", [])

        if not pairs:
            raise HTTPException(status_code=404, detail="Token not found on any exchange")

        sol_pairs = [p for p in pairs if p.get("chainId") == "solana"]
        pair = sol_pairs[0] if sol_pairs else pairs[0]

        price = float(pair.get("priceUsd", 0))
        volume24h = float(pair.get("volume", {}).get("h24", 0))
        liquidity = float(pair.get("liquidity", {}).get("usd", 0))
        token_name = pair.get("baseToken", {}).get("name", "Unknown Token")
        base_symbol = pair.get("baseToken", {}).get("symbol", symbol)
        mint_address = pair.get("baseToken", {}).get("address")

        onchain_info = (
            get_token_onchain_info(mint_address)
            if mint_address
            else {"creator": None, "top_holders": [], "lp_locked": True}
        )
        holder_metrics = get_holder_metrics(onchain_info)
        creator_history = None

        if onchain_info.get("creator"):
            creator_history = get_creator_history(onchain_info["creator"])

        prediction_result = predict_trend(
            price, volume24h, liquidity, mint_address, holder_metrics, creator_history
        )

        chart_analysis = prediction_result.get("chart_analysis", "")

        send_discord_notification(
            symbol=base_symbol,
            token_name=token_name,
            price=price,
            prediction=prediction_result["prediction"],
            volume24h=volume24h,
            liquidity=liquidity,
        )

        if user and db:
            try:
                save_analysis_to_history(
                    user=user,
                    analysis_type="crypto",
                    symbol=base_symbol,
                    name=token_name,
                    price=price,
                    prediction=prediction_result["prediction"],
                    confidence=int(prediction_result["confidence"]),
                    position_type=None,
                    db=db,
                )
            except Exception as e:
                print(f"Failed to save crypto analysis history: {e}")

                return {
            "symbol": base_symbol,
            "name": token_name,
            "price": price,
            "volume24h": volume24h,
            "liquidity": liquidity,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "reasoning": prediction_result["reasoning"],
            "highest_price": prediction_result["highest_price"],
            "lowest_price": prediction_result["lowest_price"],
            "chart_analysis": chart_analysis,
            "onchain_info": onchain_info,
            "holder_metrics": holder_metrics,
            "creator_history": creator_history,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ======================= STOCK PREDICT ENDPOINT =======================
def send_stock_discord_notification(
    ticker: str,
    company_name: Optional[str] = None,
    price: Optional[float] = None,
    prediction: Optional[str] = None,
    sector: Optional[str] = None,
    position_type: Optional[str] = None,
):
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
                {
                    "name": "🕒 Time",
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S EST"),
                    "inline": True,
                },
            ],
            "footer": {"text": "Stock Market Analyst"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        if company_name:
            embed["fields"].insert(
                1, {"name": "🏢 Company", "value": company_name, "inline": False}
            )
        if price:
            embed["fields"].append(
                {"name": "💰 Price", "value": f"${price:.2f}", "inline": True}
            )
        if sector:
            embed["fields"].append(
                {"name": "🏭 Sector", "value": sector, "inline": True}
            )
        if prediction:
            embed["fields"].append(
                {"name": "🎯 Prediction", "value": prediction, "inline": False}
            )
        if position_type:
            embed["fields"].append(
                {"name": "📍 Position", "value": position_type, "inline": False}
            )

        payload = {"embeds": [embed]}
        resp = requests.post(DISCORD_WEBHOOK_STOCK, json=payload, timeout=5)
        print("Discord stock webhook status:", resp.status_code, resp.text[:200])
    except Exception as e:
        print(f"Discord stock notification failed: {e}")
        

@app.get("/api/predict-stock")
async def predict_stock(
    ticker: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    db: Session = Depends(get_db),
):
    try:
        user: Optional[User] = None
        if credentials:
            try:
                user = get_current_user(credentials, db)
            except HTTPException:
                user = None

        data = get_stock_data(ticker)
        if not data:
            raise HTTPException(status_code=404, detail="Stock not found")

        price = data["price"]
        fundamentals = get_stock_fundamentals(ticker)
        technicals = get_stock_technicals(ticker)
        news_articles = get_stock_news(ticker)
        news_sentiment = analyze_news_sentiment(ticker, news_articles)

        prediction_result = predict_stock_trend_with_levels(
            ticker, price, fundamentals, technicals, news_sentiment
        )

        send_stock_discord_notification(
            ticker=ticker,
            company_name=data["name"],
            price=price,
            prediction=prediction_result["prediction"],
            sector=data["sector"],
            position_type=prediction_result["position_type"],
        )

        if user and db:
            try:
                save_analysis_to_history(
                    user=user,
                    analysis_type="stock",
                    symbol=ticker,
                    name=data["name"],
                    price=price,
                    prediction=prediction_result["prediction"],
                    confidence=int(
                        prediction_result["score"]
                        / prediction_result["max_score"]
                        * 100
                    ),
                    position_type=prediction_result["position_type"],
                    db=db,
                )
            except Exception as e:
                print(f"Failed to save stock analysis history: {e}")

        return {
            "ticker": ticker,
            "name": data["name"],
            "price": price,
            "sector": data["sector"],
            "industry": data["industry"],
            "prediction": prediction_result["prediction"],
            "position_type": prediction_result["position_type"],
            "score": prediction_result["score"],
            "max_score": prediction_result["max_score"],
            "reasoning": prediction_result["reasoning"],
            "news_sentiment": news_sentiment,
            "direction": prediction_result["direction"],
            "should_buy_now": prediction_result["should_buy_now"],
            "buy_price": prediction_result["buy_price"],
            "stop_loss": prediction_result["stop_loss"],
            "take_profit": prediction_result["take_profit"],
            "max_upside_price": prediction_result["max_upside_price"],
            "max_downside_price": prediction_result["max_downside_price"],
            "use_leverage": prediction_result["use_leverage"],
            "leverage_side": prediction_result["leverage_side"],
            "suggested_leverage": prediction_result["suggested_leverage"],
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stock prediction error: {str(e)}")

# ======================= SUPPORTING ENDPOINTS =======================

ADDRESS_SYMBOL_MAP = {
    "8Jx8AAHj86wbQgUTjGuj6GTTL5Ps3cqxKRTvpaJApump": "PENGUIN",
    "x95HN3DWvbfCBtTjGm587z8suK3ec6cwQwgZNLbWKyp": "HACHI",
    "GtDZKAqvMZMnti46ZewMiXCa4oXF4bZxwQPoKzXPFxZn": "SHITCOIN",
    "GaPbGp23pPuY9QBLPUjUEBn2MKEroTe9Q3M3f2Xpump": "WIF",
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "BONK",
    "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm": "NUB",
    "5UUH9RTDiSpq6HKS6bp4NdU9PNJpXRXuiw6ShBTBhgH2": "TROLL",
}


@app.get("/api/latest-tokens")
async def get_latest_tokens():
    try:
        addresses = ",".join(ADDRESS_SYMBOL_MAP.keys())
        url = f"https://api.dexscreener.com/latest/dex/tokens/{addresses}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"tokens": []}

        data = response.json() or {}
        pairs = data.get("pairs", []) or []

        tokens_by_addr = {}

        for p in pairs:
            base = p.get("baseToken", {}) or {}
            quote = p.get("quoteToken", {}) or {}
            addr = base.get("address")
            price = p.get("priceUsd")
            quote_symbol = (quote.get("symbol") or "").upper()

            if addr not in ADDRESS_SYMBOL_MAP or price is None:
                continue

            try:
                price_f = float(price)
                price_str = (
                    f"${price_f:.6f}" if price_f < 1 else f"${price_f:.4f}"
                )
            except Exception:
                price_str = f"{price}"

            existing = tokens_by_addr.get(addr)
            if existing:
                prev_quote = existing.get("quote_symbol", "")
                if prev_quote in ("SOL", "USDC", "USDT"):
                    continue

            tokens_by_addr[addr] = {
                "symbol": ADDRESS_SYMBOL_MAP[addr],
                "price": price_str,
                "quote_symbol": quote_symbol,
            }

        tokens = []
        for addr in ADDRESS_SYMBOL_MAP.keys():
            t = tokens_by_addr.get(addr)
            if t:
                tokens.append({"symbol": t["symbol"], "price": t["price"]})
            else:
                tokens.append({"symbol": ADDRESS_SYMBOL_MAP[addr], "price": "N/A"})

        return {"tokens": tokens}
    except Exception as e:
        print(f"Latest tokens error: {e}")
        return {"tokens": []}


@app.get("/api/trending-stocks")
async def get_trending_stocks():
    try:
        popular = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN"]
        stocks = []
        for t in popular:
            data = get_stock_data(t)
            if data:
                price_str = f"${data['price']:.2f}"
            else:
                price_str = "--"
            stocks.append({"ticker": t, "price": price_str})
        return {"stocks": stocks}
    except Exception as e:
        print(f"Trending stocks error: {e}")
        return {
            "stocks": [
                {"ticker": t, "price": "--"}
                for t in ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN"]
            ]
        }


@app.get("/api/token-info")
async def get_token_info(symbol: str):
    try:
        search_url = f"https://api.dexscreener.com/latest/dex/search?q={symbol}"
        response = requests.get(search_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch token data")

        data = response.json()
        pairs = data.get("pairs", [])

        if not pairs:
            raise HTTPException(status_code=404, detail="Token not found on any exchange")

        pair = pairs[0]
        token_name = pair.get("baseToken", {}).get("name", "Unknown Token")
        base_symbol = pair.get("baseToken", {}).get("symbol", symbol)
        mint_address = pair.get("baseToken", {}).get("address")

        onchain_info = (
            get_token_onchain_info(mint_address)
            if mint_address
            else {"creator": None, "top_holders": [], "lp_locked": True}
        )
        holder_metrics = get_holder_metrics(onchain_info)

        return {
            "symbol": base_symbol,
            "name": token_name,
            "mint_address": mint_address,
            "onchain_info": onchain_info,
            "holder_metrics": holder_metrics,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token info error: {str(e)}")

# === NEWS TICKERS ===

@app.get("/api/crypto-news")
async def get_crypto_news():
    try:
        news = [
            {
                "headline": "Solana memecoins see surge in volume",
                "source": "On‑chain Watch",
                "url": "https://dexscreener.com/solana",
            },
            {
                "headline": "New Solana DEX launches zero‑fee trading",
                "source": "Solana News",
                "url": "https://solana.com",
            },
            {
                "headline": "DeFi TVL climbs to new monthly high",
                "source": "DeFiPulse",
                "url": "https://defipulse.com",
            },
            {
                "headline": "BTC breaks key resistance level",
                "source": "MarketWire",
                "url": "https://www.coindesk.com",
            },
            {
                "headline": "AI agents begin auto‑trading memecoins",
                "source": "MemeBots",
                "url": "https://twitter.com",
            },
        ]
        return {"news": news}
    except Exception as e:
        print(f"Crypto news error: {e}")
        return {"news": []}


@app.get("/api/market-news")
async def get_market_news():
    try:
        url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
        resp = requests.get(url, timeout=10)
        raw = []
        if resp.status_code == 200:
            raw = resp.json() or []
        else:
            print(f"Finnhub market news error: {resp.status_code} {resp.text[:200]}")

        if raw:
            news = [
                {
                    "headline": item.get("headline", "Market update"),
                    "source": item.get("source", "Finnhub"),
                    "url": item.get("url"),
                }
                for item in raw[:30]
            ]
            return {"news": news}

        fallback = [
            {
                "headline": "S&P 500 holds near recent highs",
                "source": "MarketWire",
                "url": "https://www.marketwatch.com",
            },
            {
                "headline": "Tech stocks lead midday gains",
                "source": "TechDaily",
                "url": "https://www.cnbc.com",
            },
            {
                "headline": "Fed commentary keeps traders cautious",
                "source": "MacroBrief",
                "url": "https://www.bloomberg.com",
            },
        ]
        return {"news": fallback}
    except Exception as e:
        print(f"Market news error: {e}")
        return {"news": []}

# === INDICES & ROOT ===

@app.get("/api/market-indices")
async def get_market_indices():
    try:
        sp = yf.Ticker("^GSPC")
        hist = sp.history(period="2d")
        if len(hist) < 2:
            return {"indices": []}

        latest = hist.iloc[-1]
        prev = hist.iloc[-2]
        price = float(latest["Close"])
        change_pct = float((latest["Close"] - prev["Close"]) / prev["Close"] * 100)

        return {"indices": [{"name": "S&P 500", "price": price, "change": change_pct}]}
    except Exception as e:
        print(f"Market indices error: {e}")
        return {"indices": []}


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.get("/api")
async def get_api():
    return {
        "message": "Market Analyst API - Crypto & Stocks with Auth",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10000)
