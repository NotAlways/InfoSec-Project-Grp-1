import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

MODEL_FILENAME = "anomaly_model.pkl"


def model_path():
    return str(Path(__file__).parent / MODEL_FILENAME)


def save_model(model, path: str = None):
    path = path or model_path()
    joblib.dump(model, path)


def load_model(path: str = None):
    path = path or model_path()
    if not os.path.exists(path):
        return None
    return joblib.load(path)


async def fetch_activity_df(session: AsyncSession):
    """Fetch activity rows and return a pandas DataFrame with basic features."""
    rows = []
    try:
        res = await session.execute(text("""
            SELECT user_id, login_time, ip_address, success 
            FROM user_activity
        """))
        result = res.fetchall()
    except Exception:
        return pd.DataFrame()

    for r in result:
        rows.append({
            "user_id": int(r[0]) if r[0] is not None else 0,
            "login_time": pd.to_datetime(r[1]) if r[1] is not None else pd.NaT,
            "ip_address": r[2],
            "success": bool(r[3]) if r[3] is not None else True,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["hour"] = df["login_time"].dt.hour.fillna(0).astype(int)
    df["failed"] = (~df["success"]).astype(int)
    return df


def train_model(df: pd.DataFrame, contamination: float = 0.05):
    if df.empty:
        return None
    X = df[["hour", "failed"]]
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    return model


def check_anomaly(model, login_time, success: bool):
    if model is None:
        return False
    hour = getattr(login_time, "hour", 0)
    failed = 0 if success else 1
    X_new = [[int(hour), int(failed)]]
    try:
        pred = model.predict(X_new)
        return int(pred[0]) == -1
    except Exception:
        return False
