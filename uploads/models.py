from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///voicemail_detector.db')
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Voicemail(Base):
    """Main voicemail record"""
    __tablename__ = "voicemails"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Email metadata
    sender_email = Column(String(255), nullable=False)
    subject = Column(String(500))
    phone_number = Column(String(20))
    email_uid = Column(String(50), unique=True)  # Unique email identifier
    
    # File information
    file_path = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_size = Column(Integer)  # Size in bytes
    
    # Audio transcription
    transcribed_text = Column(Text)
    transcription_language = Column(String(10))  # e.g., "en", "es"
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)

class ProcessingLog(Base):
    """Log of system operations for debugging"""
    __tablename__ = "processing_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    voicemail_id = Column(Integer, ForeignKey("voicemails.id"), nullable=True)
    
    operation = Column(String(100), nullable=False)  # "transcription", "analysis", "email_response"
    status = Column(String(20), nullable=False)  # "success", "error", "warning"
    message = Column(Text)
    error_details = Column(Text)
    processing_time = Column(Float)  # Time in seconds
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

# Database initialization
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session():
    """Get database session for direct use"""
    return SessionLocal()

if __name__ == "__main__":
    # Create tables if running this file directly
    create_tables()
    print("✅ Database tables created successfully!")
    print(f"📁 Database location: {DATABASE_URL}")