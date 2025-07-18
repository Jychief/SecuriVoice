from models import Voicemail, ProcessingLog, get_db_session, create_tables
from datetime import datetime
import os
import traceback
from typing import Optional

def init_database():
    """Initialize the database and create tables"""
    try:
        create_tables()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def save_voicemail(sender_email: str, subject: str, phone_number: str, 
                  email_uid: str, file_path: str, transcribed_text: str = None) -> Optional[int]:
    """
    Save a new voicemail record to the database
    
    Returns:
        The ID of the created voicemail record, or None if failed
    """
    db = get_db_session()
    try:
        # Get file information
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        # Create voicemail record
        voicemail = Voicemail(
            sender_email=sender_email,
            subject=subject,
            phone_number=phone_number,
            email_uid=email_uid,
            file_path=file_path,
            file_name=file_name,
            file_size=file_size,
            transcribed_text=transcribed_text,
            processed_at=datetime.utcnow() if transcribed_text else None
        )
        
        db.add(voicemail)
        db.commit()
        db.refresh(voicemail)
        
        print(f"💾 Saved voicemail to database (ID: {voicemail.id})")
        return voicemail.id
        
    except Exception as e:
        db.rollback()
        print(f"❌ Failed to save voicemail: {e}")
        log_processing_error(None, "save_voicemail", str(e))
        return None
    finally:
        db.close()

def update_transcription(voicemail_id: int, transcribed_text: str, 
                        language: str = None) -> bool:
    """
    Update a voicemail record with transcription results
    """
    db = get_db_session()
    try:
        voicemail = db.query(Voicemail).filter(Voicemail.id == voicemail_id).first()
        if not voicemail:
            print(f"❌ Voicemail {voicemail_id} not found")
            return False
        
        voicemail.transcribed_text = transcribed_text
        voicemail.transcription_language = language
        voicemail.processed_at = datetime.utcnow()
        
        db.commit()
        print(f"💾 Updated transcription for voicemail {voicemail_id}")
        return True
        
    except Exception as e:
        db.rollback()
        print(f"❌ Failed to update transcription: {e}")
        return False
    finally:
        db.close()

def get_voicemail_by_id(voicemail_id: int) -> Optional[Voicemail]:
    """Get a voicemail by its ID"""
    db = get_db_session()
    try:
        return db.query(Voicemail).filter(Voicemail.id == voicemail_id).first()
    finally:
        db.close()

def get_voicemail_by_email_uid(email_uid: str) -> Optional[Voicemail]:
    """Get a voicemail by email UID (to avoid duplicates)"""
    db = get_db_session()
    try:
        return db.query(Voicemail).filter(Voicemail.email_uid == email_uid).first()
    finally:
        db.close()

def get_recent_voicemails(limit: int = 10):
    """Get the most recent voicemails"""
    db = get_db_session()
    try:
        return db.query(Voicemail).order_by(Voicemail.created_at.desc()).limit(limit).all()
    finally:
        db.close()

def log_processing_error(voicemail_id: Optional[int], operation: str, error_message: str):
    """Log processing errors for debugging"""
    db = get_db_session()
    try:
        log_entry = ProcessingLog(
            voicemail_id=voicemail_id,
            operation=operation,
            status="error",
            message="Processing failed",
            error_details=error_message
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"❌ Failed to log error: {e}")
    finally:
        db.close()

def log_processing_success(voicemail_id: int, operation: str, message: str, processing_time: float = None):
    """Log successful operations"""
    db = get_db_session()
    try:
        log_entry = ProcessingLog(
            voicemail_id=voicemail_id,
            operation=operation,
            status="success",
            message=message,
            processing_time=processing_time
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"❌ Failed to log success: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # Test database operations
    if init_database():
        print("🧪 Testing database operations...")
        
        # Test saving a voicemail
        test_id = save_voicemail(
            sender_email="test@example.com",
            subject="Test Voicemail",
            phone_number="5551234567",
            email_uid="test_123",
            file_path="./uploads/test.m4a",
            transcribed_text="This is a test transcription"
        )
        
        if test_id:
            print(f"✅ Test voicemail saved with ID: {test_id}")
            
            # Test retrieving
            voicemail = get_voicemail_by_id(test_id)
            if voicemail:
                print(f"✅ Retrieved voicemail: {voicemail.sender_email}")
        
        print("🧪 Database test complete!")