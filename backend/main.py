from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime

# Database setup, change to your own password here. Make sure PostgreSQL is running. To be encrypted in the future. 
DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost/notevault"
Base = declarative_base()

# Models
class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NoteSchema(BaseModel):
    title: str
    content: str

class NoteResponse(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# FastAPI app
app = FastAPI(title="NoteVault API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# Routes
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.post("/notes", response_model=NoteResponse)
async def create_note(note: NoteSchema, db: AsyncSession = Depends(get_db)):
    new_note = Note(title=note.title, content=note.content)
    db.add(new_note)
    await db.commit()
    await db.refresh(new_note)
    return new_note

@app.get("/notes", response_model=list[NoteResponse])
async def get_notes():
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note))
        return result.scalars().all()

@app.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(note_id: int):
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        note = result.scalar_one_or_none()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        return note

@app.put("/notes/{note_id}", response_model=NoteResponse)
async def update_note(note_id: int, note: NoteSchema):
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        db_note = result.scalar_one_or_none()
        if not db_note:
            raise HTTPException(status_code=404, detail="Note not found")
        db_note.title = note.title
        db_note.content = note.content
        db_note.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(db_note)
        return db_note

@app.delete("/notes/{note_id}")
async def delete_note(note_id: int):
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        db_note = result.scalar_one_or_none()
        if not db_note:
            raise HTTPException(status_code=404, detail="Note not found")
        await session.delete(db_note)
        await session.commit()
        return {"message": "Note deleted"}
