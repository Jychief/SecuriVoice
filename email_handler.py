import imaplib
import email
import os
import time
from dotenv import load_dotenv
from threading import Event
import traceback
from speech_to_text import transcribe_audio

load_dotenv()

EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
UPLOAD_DIR = os.getenv('UPLOAD_DIR', './uploads')

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global stop event for clean shutdown
stop_event = Event()

def process_email(uid, mail):
    """Process a single email and extract voicemail data"""
    print(f"Processing email with UID: {uid}")
    try:
        # Fetch the email
        status, msg_data = mail.fetch(uid, '(RFC822)')
        if status != 'OK':
            print(f"Failed to fetch email {uid}")
            return
            
        msg = email.message_from_bytes(msg_data[0][1])

        # Extract basic info
        sender_email = msg['From']
        subject = msg['Subject'] or "voicemail"
        phone_number = None
        saved_file = None

        print(f"Email from: {sender_email}")
        print(f"Subject: {subject}")

        # Walk through all parts of the email
        for part in msg.walk():
            # Look for phone number in plain text
            if part.get_content_type() == "text/plain" and phone_number is None:
                try:
                    text_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    phone_number = text_content.strip()
                    print(f"Found text content: {phone_number[:100]}...")  # First 100 chars
                except Exception as e:
                    print(f"Error reading text content: {e}")
            
            # Look for .m4a attachments
            if part.get('Content-Disposition'):
                filename = part.get_filename()
                if filename and filename.lower().endswith('.m4a'):
                    try:
                        # Create safe filename
                        safe_subject = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).strip()
                        safe_subject = safe_subject or "voicemail"
                        saved_file = os.path.join(UPLOAD_DIR, f"{safe_subject}_{uid.decode()}.m4a")
                        
                        # Save the attachment
                        with open(saved_file, 'wb') as f:
                            f.write(part.get_payload(decode=True))
                        print(f"Saved attachment: {saved_file}")
                        
                    except Exception as e:
                        print(f"Error saving attachment: {e}")

        print(f"✅ Processed voicemail from {sender_email}")
        if phone_number:
            print(f"   Phone: {phone_number}")
        if saved_file:
            print(f"   File: {saved_file}")
            
            # Transcribe the audio file
            try:
                print("🎤 Transcribing audio...")
                transcribed_text = transcribe_audio(saved_file)
                print("✅ Transcription complete:")
                print(f"📝 Text: {transcribed_text}")
                
                # TODO: Add phishing analysis here
                # TODO: Send response email with results
                
            except Exception as e:
                print(f"❌ Transcription failed: {e}")
                traceback.print_exc()
        
        # Mark email as read
        mail.store(uid, '+FLAGS', '\\Seen')
        print(f"   Marked as read")
        
    except Exception as e:
        print(f"❌ Error processing email {uid}: {e}")
        traceback.print_exc()

def polling_loop():
    """Main polling loop - checks for new emails every 15 seconds"""
    print("🚀 Starting email polling system...")
    
    last_email_count = 0
    processed_uids = set()
    
    while not stop_event.is_set():
        mail = None
        try:
            # Connect to email server (silently during normal operation)
            mail = imaplib.IMAP4_SSL(EMAIL_HOST)
            mail.login(EMAIL_USER, EMAIL_PASS)
            mail.select('inbox')
            
            # Get current email count
            status, data = mail.search(None, 'ALL')
            current_count = len(data[0].split()) if data[0] else 0
            
            # Check if we have new emails
            if current_count > last_email_count:
                new_emails = current_count - last_email_count
                print(f"🔔 Found {new_emails} new emails! Total: {current_count}")
                
                # Get all unseen emails
                status, unseen_data = mail.search(None, 'UNSEEN')
                if unseen_data[0]:
                    unseen_uids = unseen_data[0].split()
                    print(f"📬 Processing {len(unseen_uids)} unseen emails...")
                    
                    for uid in unseen_uids:
                        if uid not in processed_uids:
                            process_email(uid, mail)
                            processed_uids.add(uid)
                            time.sleep(1)  # Brief pause between emails
                
                last_email_count = current_count
                
            elif current_count == last_email_count:
                print(f"⏳ No new emails. Total: {current_count} (waiting 15 seconds...)")
            else:
                # Email count decreased (emails were deleted)
                print(f"📉 Email count changed: {last_email_count} → {current_count}")
                last_email_count = current_count
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("🔄 Will retry in 30 seconds...")
            stop_event.wait(30)
            continue
            
        finally:
            # Always clean up connection
            if mail:
                try:
                    mail.close()
                    mail.logout()
                except:
                    pass
        
        # Wait 15 seconds before next check
        if not stop_event.is_set():
            stop_event.wait(15)
    
    print("🛑 Email polling stopped")

def start_email_monitoring():
    """Start the email monitoring system - called by Flask app"""
    print("🔧 Initializing email monitoring...")
    
    # Test connection first
    if test_connection():
        print("✅ Email connection test passed")
        print("🔄 Starting continuous polling...")
        polling_loop()
    else:
        print("❌ Email connection test failed - monitoring not started")

def test_connection():
    """Test email connection without polling"""
    try:
        print("🧪 Testing email connection...")
        mail = imaplib.IMAP4_SSL(EMAIL_HOST)
        print("✅ Connected to server")
        
        mail.login(EMAIL_USER, EMAIL_PASS)
        print("✅ Login successful")
        
        mail.select('inbox')
        print("✅ Inbox selected")
        
        # Check email counts
        status, data = mail.search(None, 'ALL')
        total = len(data[0].split()) if data[0] else 0
        
        status, data = mail.search(None, 'UNSEEN')
        unseen = len(data[0].split()) if data[0] else 0
        
        print(f"📊 Total emails: {total}")
        print(f"📊 Unseen emails: {unseen}")
        
        mail.close()
        mail.logout()
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Test connection first
    if test_connection():
        print("\n" + "="*50)
        print("Starting polling loop...")
        print("Send yourself an email with .m4a attachment to test!")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        try:
            polling_loop()
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            stop_event.set()
            print("✅ Stopped")