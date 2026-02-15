from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def init_migration(engine):
    """Create user_activity table if it doesn't exist."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS user_activity (
                    id SERIAL PRIMARY KEY,
                    user_id INT NOT NULL,
                    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address VARCHAR(45),
                    device_info TEXT,
                    action VARCHAR(50),
                    success BOOLEAN
                )
                """
            )
        )


async def log_activity(session: AsyncSession, user_id: int, ip: str | None, device: str | None, action: str, success: bool = True):
    """Insert a user activity record using the provided async session."""
    await session.execute(
        text(
            """
            INSERT INTO user_activity (user_id, login_time, ip_address, device_info, action, success)
            VALUES (:user_id, :login_time, :ip_address, :device_info, :action, :success)
            """
        ),
        {
            "user_id": user_id,
            "login_time": datetime.utcnow(),
            "ip_address": ip,
            "device_info": device,
            "action": action,
            "success": success,
        },
    )
    await session.commit()
