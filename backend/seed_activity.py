#!/usr/bin/env python3
"""
Seed Test Activity Data
-----------------------
Generates realistic user activity logs for testing anomaly detection.
Creates varied patterns: normal logins, unusual times, failed attempts, different IPs.

Usage:
    python seed_activity.py                      # Generate default 100 records
    python seed_activity.py --count 500          # Generate 500 records
    python seed_activity.py --count 50 --clear   # Clear existing data, then generate 50
    python seed_activity.py --anomalies 20       # Include 20 anomalous records
"""

import argparse
import asyncio
import random
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

# Database URL (same as main.py)
DATABASE_URL = "postgresql+asyncpg://postgres:1m1f1b1m@localhost/notevault"


# Sample data pools
VALID_IPS = [
    "192.168.1.10",      # Home IP
    "192.168.1.15",      # Home IP (variant)
    "10.0.0.50",         # Office IP
    "10.0.0.51",         # Office IP (variant)
    "203.0.113.42",      # ISP IP
    "203.0.113.43",      # ISP IP (variant)
]

ANOMALY_IPS = [
    "195.154.1.100",     # Foreign country
    "185.220.101.45",    # Tor exit node
    "89.163.128.1",      # Suspicious
]

DEVICES = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) Safari/604.1",
]


async def clear_activity_data(engine):
    """Delete all records from user_activity table."""
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM user_activity"))
    print("[INFO] Cleared existing activity data")


async def generate_normal_activity(session: AsyncSession, count: int):
    """
    Generate normal activity: business hours, familiar IPs, mostly successful.
    Simulates a typical user with predictable behavior.
    """
    print(f"[INFO] Generating {count} normal activity records...")
    
    for i in range(count):
        # Bias towards business hours (8 AM - 6 PM)
        hour = random.choice(list(range(8, 19)) + list(range(8, 19)))  # More weight on business hours
        minute = random.randint(0, 59)
        
        login_time = datetime.now(timezone.utc) - timedelta(
            days=random.randint(1, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        login_time = login_time.replace(hour=hour, minute=minute, tzinfo=None)  # Remove timezone for DB
        
        # 95% successful, 5% failed
        success = random.random() > 0.05
        ip = random.choice(VALID_IPS)
        device = random.choice(DEVICES)
        
        await session.execute(
            text("""
                INSERT INTO user_activity (user_id, login_time, ip_address, device_info, action, success)
                VALUES (:user_id, :login_time, :ip_address, :device_info, :action, :success)
            """),
            {
                "user_id": random.randint(1, 10),
                "login_time": login_time,
                "ip_address": ip,
                "device_info": device,
                "action": "login",
                "success": success,
            },
        )
        
        if (i + 1) % 50 == 0:
            await session.commit()
            print(f"  - Inserted {i + 1} records")
    
    await session.commit()
    print(f"[SUCCESS] Generated {count} normal records")


async def generate_anomaly_activity(session: AsyncSession, count: int):
    """
    Generate anomalous activity: unusual times, foreign IPs, failed attempts.
    These patterns should be flagged as anomalies by IsolationForest.
    """
    print(f"[INFO] Generating {count} anomalous activity records...")
    
    for i in range(count):
        # Unusual times: late night (2-4 AM) or very early morning (5-6 AM)
        hour = random.choice([2, 3, 4, 5])
        minute = random.randint(0, 59)
        
        login_time = datetime.now(timezone.utc) - timedelta(
            days=random.randint(1, 90),
            hours=random.randint(0, 23),
        )
        login_time = login_time.replace(hour=hour, minute=minute, tzinfo=None)  # Remove timezone for DB
        
        # 60% failed, 40% successful (anomalously high failure rate)
        success = random.random() > 0.6
        ip = random.choice(ANOMALY_IPS)
        device = random.choice(DEVICES)
        
        await session.execute(
            text("""
                INSERT INTO user_activity (user_id, login_time, ip_address, device_info, action, success)
                VALUES (:user_id, :login_time, :ip_address, :device_info, :action, :success)
            """),
            {
                "user_id": random.randint(1, 10),
                "login_time": login_time,
                "ip_address": ip,
                "device_info": device,
                "action": "login",
                "success": success,
            },
        )
        
        if (i + 1) % 25 == 0:
            await session.commit()
            print(f"  - Inserted {i + 1} anomaly records")
    
    await session.commit()
    print(f"[SUCCESS] Generated {count} anomalous records")


async def seed_data(clear: bool = False, normal_count: int = 100, anomaly_count: int = 0):
    """Main seed function."""
    try:
        engine = create_async_engine(DATABASE_URL, echo=False)
        SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        # Optionally clear existing data
        if clear:
            await clear_activity_data(engine)
        
        async with SessionLocal() as session:
            # Generate normal activity
            await generate_normal_activity(session, normal_count)
            
            # Generate anomalous activity if requested
            if anomaly_count > 0:
                await generate_anomaly_activity(session, anomaly_count)
        
        await engine.dispose()
        
        total = normal_count + anomaly_count
        print(f"\n[SUCCESS] Seeded {total} records total")
        print(f"  - Normal: {normal_count}")
        print(f"  - Anomalous: {anomaly_count}")
        return {"status": "success", "records": total}
    
    except Exception as e:
        print(f"[ERROR] Seeding failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate test activity data for anomaly detection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 normal records
  python seed_activity.py

  # Generate 500 normal records
  python seed_activity.py --count 500

  # Generate 80 normal + 20 anomalous
  python seed_activity.py --count 80 --anomalies 20

  # Clear existing data and generate fresh 100 normal
  python seed_activity.py --clear

  # Full test dataset: clear, then 200 normal + 50 anomalies
  python seed_activity.py --clear --count 200 --anomalies 50
        """,
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of normal activity records to generate (default 100)",
    )
    
    parser.add_argument(
        "--anomalies",
        type=int,
        default=0,
        help="Number of anomalous activity records to generate (default 0)",
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing activity data before seeding",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NoteVault Activity Data Seeder")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    
    result = asyncio.run(
        seed_data(
            clear=args.clear,
            normal_count=args.count,
            anomaly_count=args.anomalies,
        )
    )
    
    print("=" * 60)
    print(f"Result: {result['status'].upper()}")
    print("=" * 60)
    
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    exit(main())
