from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime
from crypto import load_key, encrypt_content, decrypt_content

# Database setup, change to your own password here. Make sure PostgreSQL is running.
DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost/notevault"
Base = declarative_base()

# Load encryption key on startup
encryption_key = None

def get_encryption_key():
    global encryption_key
    if encryption_key is None:
        encryption_key = load_key()
    return encryption_key

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
        from_attributes = True

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
    key = get_encryption_key()
    encrypted_content = encrypt_content(note.content, key)
    new_note = Note(title=note.title, content=encrypted_content)
    db.add(new_note)
    await db.commit()
    await db.refresh(new_note)
    return new_note

@app.get("/notes", response_model=list[NoteResponse])
async def get_notes():
    from sqlalchemy import select
    key = get_encryption_key()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note))
        notes = result.scalars().all()
        # Decrypt content for each note
        for note in notes:
            note.content = decrypt_content(note.content, key)
        return notes

@app.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(note_id: int):
    from sqlalchemy import select
    key = get_encryption_key()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        note = result.scalar_one_or_none()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        note.content = decrypt_content(note.content, key)
        return note

@app.put("/notes/{note_id}", response_model=NoteResponse)
async def update_note(note_id: int, note: NoteSchema):
    from sqlalchemy import select
    key = get_encryption_key()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Note).filter(Note.id == note_id))
        db_note = result.scalar_one_or_none()
        if not db_note:
            raise HTTPException(status_code=404, detail="Note not found")
        db_note.title = note.title
        db_note.content = encrypt_content(note.content, key)
        db_note.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(db_note)
        # Decrypt for response
        db_note.content = decrypt_content(db_note.content, key)
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
