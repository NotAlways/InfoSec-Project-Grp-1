#!/usr/bin/env python3
"""
Anomaly Detection Model Retraining CLI
--------------------------------------
Fetches user activity logs from PostgreSQL, trains an IsolationForest model,
and saves it to disk. Can be run manually or scheduled via cron/Task Scheduler.

Usage:
    python train_anomaly.py                          # Train with defaults
    python train_anomaly.py --contamination 0.1      # Custom contamination rate
    python train_anomaly.py --output ./models/model.pkl  # Custom output path
    python train_anomaly.py --show-stats             # Display training stats
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Import anomaly and activity modules
from anomaly import fetch_activity_df, train_model, save_model
from activity import init_migration


# Database URL (same as main.py)
DATABASE_URL = "postgresql+asyncpg://postgres:1m1f1b1m@localhost/notevault"


async def retrain_model(
    contamination: float = 0.05,
    output_path: str = None,
    show_stats: bool = False,
):
    """
    Fetch activity data, train model, and save it.
    
    Args:
        contamination: IsolationForest contamination parameter (default 0.05).
        output_path: Custom path to save model (default: backend/anomaly_model.pkl).
        show_stats: Whether to print training statistics.
    
    Returns:
        dict with training result status.
    """
    try:
        # Create async engine
        engine = create_async_engine(DATABASE_URL, echo=False)
        
        # Ensure user_activity table exists
        try:
            await init_migration(engine)
        except Exception as e:
            print(f"[WARN] Migration skipped: {e}")
        
        # Create async session
        SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with SessionLocal() as session:
            print("[INFO] Fetching user activity data...")
            df = await fetch_activity_df(session)
            
            if df.empty:
                print("[WARN] No activity data found. Model not trained.")
                await engine.dispose()
                return {"status": "skipped", "reason": "no_data"}
            
            print(f"[INFO] Fetched {len(df)} activity records")
            
            if show_stats:
                print(f"[STATS] Features used: hour, failed")
                print(f"[STATS] Data shape: {df.shape}")
                print(f"[STATS] Hour range: {df['hour'].min()}-{df['hour'].max()}")
                print(f"[STATS] Success rate: {(1 - df['failed'].mean()) * 100:.1f}%")
            
            print(f"[INFO] Training IsolationForest (contamination={contamination})...")
            model = train_model(df, contamination=contamination)
            
            if model is None:
                print("[ERROR] Model training failed")
                await engine.dispose()
                return {"status": "error", "reason": "training_failed"}
            
            # Save model
            from anomaly import model_path
            save_to = output_path or model_path()
            
            # Ensure directory exists
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            
            save_model(model, save_to)
            print(f"[SUCCESS] Model saved to {save_to}")
            
            await engine.dispose()
            return {
                "status": "success",
                "records": len(df),
                "model_path": save_to,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    except Exception as e:
        print(f"[ERROR] Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Retrain anomaly detection model from user activity logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard retraining
  python train_anomaly.py

  # Custom contamination (5% vs 10% anomalies)
  python train_anomaly.py --contamination 0.1

  # Show training statistics
  python train_anomaly.py --show-stats

  # Save to custom location
  python train_anomaly.py --output /path/to/model.pkl
  
Scheduling (Linux/macOS - crontab):
  # Retrain weekly (every Sunday at 2 AM)
  0 2 * * 0 cd /path/to/backend && python train_anomaly.py >> /var/log/notevault_retrain.log 2>&1
  
Scheduling (Windows - Task Scheduler):
  1. Open Task Scheduler
  2. Create Basic Task: "NoteVault Anomaly Retrain"
  3. Trigger: Weekly, Sunday 2:00 AM
  4. Action: Start a program
     - Program: C:\\Python39\\python.exe
     - Arguments: train_anomaly.py
     - Start in: C:\\path\\to\\backend
        """,
    )
    
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="IsolationForest contamination parameter (0.0-1.0, default 0.05)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for model file (default: backend/anomaly_model.pkl)",
    )
    
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Print training statistics",
    )
    
    args = parser.parse_args()
    
    # Validate contamination
    if not 0 < args.contamination < 1:
        print("[ERROR] Contamination must be between 0 and 1")
        sys.exit(1)
    
    print("=" * 60)
    print("NoteVault Anomaly Detection Model Retrainer")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    
    result = asyncio.run(
        retrain_model(
            contamination=args.contamination,
            output_path=args.output,
            show_stats=args.show_stats,
        )
    )
    
    print("=" * 60)
    print(f"Result: {result['status'].upper()}")
    if result.get("timestamp"):
        print(f"Timestamp: {result['timestamp']}")
    if result.get("records"):
        print(f"Records processed: {result['records']}")
    if result.get("model_path"):
        print(f"Model path: {result['model_path']}")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
