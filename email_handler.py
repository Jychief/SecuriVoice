import imaplib
import email
import os
import time
from dotenv import load_dotenv
from threading import Event
import traceback
from speech_to_text import transcribe_audio
from text_analysis import analyze_voicemail_text
from audio_analysis import analyze_voicemail_audio
from email_response import send_analysis_report
from datetime import datetime
import logging

load_dotenv()

EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
UPLOAD_DIR = os.getenv('UPLOAD_DIR', './uploads')

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global stop event for clean shutdown
stop_event = Event()

# Configure logging to reduce noise
logging.getLogger('speech_to_text').setLevel(logging.WARNING)
logging.getLogger('text_analysis').setLevel(logging.WARNING)
logging.getLogger('audio_analysis').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

def process_email(uid, mail):
    """Process a single email and extract voicemail data"""
    print(f"\n📧 Processing email UID: {uid.decode()}")
    try:
        # Fetch the email
        status, msg_data = mail.fetch(uid, '(RFC822)')
        if status != 'OK':
            print(f"❌ Failed to fetch email {uid}")
            return
            
        msg = email.message_from_bytes(msg_data[0][1])

        # Extract basic info
        sender_email = msg['From']
        subject = msg['Subject'] or "voicemail"
        phone_number = None
        saved_file = None

        print(f"   From: {sender_email}")
        print(f"   Subject: {subject}")

        # Walk through all parts of the email
        for part in msg.walk():
            # Look for phone number in plain text
            if part.get_content_type() == "text/plain" and phone_number is None:
                try:
                    text_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    phone_number = text_content.strip()
                    print(f"   Phone: {phone_number}")
                except Exception as e:
                    print(f"   ⚠️ Error reading text content: {e}")
            
            # Look for audio attachments (.m4a, .wav, .mp3, etc.)
            if part.get('Content-Disposition'):
                filename = part.get_filename()
                if filename and any(filename.lower().endswith(ext) for ext in ['.m4a', '.wav', '.mp3', '.flac', '.ogg']):
                    try:
                        # Create safe filename with original extension
                        safe_subject = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).strip()
                        safe_subject = safe_subject or "voicemail"
                        
                        # Preserve original file extension
                        file_ext = os.path.splitext(filename)[1].lower()
                        saved_file = os.path.join(UPLOAD_DIR, f"{safe_subject}_{uid.decode()}{file_ext}")
                        
                        # Save the attachment
                        with open(saved_file, 'wb') as f:
                            f.write(part.get_payload(decode=True))
                        print(f"   💾 Saved: {os.path.basename(saved_file)}")
                        
                    except Exception as e:
                        print(f"   ❌ Error saving attachment: {e}")

        # Process the voicemail if we have both phone number and audio file
        if saved_file and phone_number:
            try:
                print(f"🎤 Processing audio...")
                # Convert audio to text using Whisper
                transcript = transcribe_audio(saved_file)
                print(f"📝 Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
                
                # Analyze the transcript for phishing indicators
                print(f"🔍 Analyzing content...")
                db_id, text_analysis = analyze_voicemail_text(transcript, phone_number, saved_file)
                
                print(f"📊 Text Risk: {text_analysis.risk_score}/10 | Indicators: {len(text_analysis.indicators)}")
                
                # Analyze audio for AI-generated voice detection
                print(f"🎵 Analyzing audio...")
                try:
                    overall_risk_score, audio_analysis = analyze_voicemail_audio(saved_file, db_id)
                    
                    ai_status = "AI" if audio_analysis.is_ai_generated else "Human"
                    print(f"🤖 Audio: {ai_status} | Risk: {audio_analysis.risk_score}/10 | Overall: {overall_risk_score}/10")
                    
                except Exception as audio_error:
                    print(f"⚠️ Audio analysis failed: {audio_error}")
                    overall_risk_score = text_analysis.risk_score
                    audio_analysis = None
                
                # Send analysis report email back to sender
                try:
                    print(f"📤 Sending report...")
                    
                    # Prepare voicemail data for email report
                    voicemail_data = {
                        'phone_number': phone_number,
                        'transcribed_text': transcript,
                        'file_name': os.path.basename(saved_file),
                        'processed_at': datetime.now(),
                        'risk_score': text_analysis.risk_score,
                        'indicators': text_analysis.indicators,
                        'explanation': text_analysis.explanation,
                        'caller_id_mismatch': text_analysis.caller_id_mismatch,
                        'transcript_numbers': text_analysis.transcript_numbers,
                        'overall_risk_score': overall_risk_score,
                        'audio_analysis': audio_analysis,
                        'has_audio_analysis': audio_analysis is not None
                    }
                    
                    # Send the analysis report email
                    email_sent = send_analysis_report(sender_email, voicemail_data)
                    
                    if email_sent:
                        print(f"✅ Report sent to {sender_email}")
                    else:
                        print(f"❌ Failed to send report")
                        
                except Exception as email_error:
                    print(f"❌ Email error: {email_error}")
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
        else:
            missing = []
            if not saved_file:
                missing.append("audio file")
            if not phone_number:
                missing.append("phone number")
            print(f"⚠️ Missing: {', '.join(missing)}")

        # Mark email as read
        mail.store(uid, '+FLAGS', '\\Seen')
        print(f"✅ Processing complete\n")
        
    except Exception as e:
        print(f"❌ Error processing email {uid}: {e}")

def get_initial_email_state():
    """Get initial email state to establish baseline"""
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_HOST)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select('inbox')
        
        # Get all current UIDs and unseen count
        status, all_data = mail.search(None, 'ALL')
        all_uids = set(all_data[0].split()) if all_data[0] else set()
        
        status, unseen_data = mail.search(None, 'UNSEEN')
        unseen_uids = set(unseen_data[0].split()) if unseen_data[0] else set()
        
        mail.close()
        mail.logout()
        
        return all_uids, unseen_uids
        
    except Exception as e:
        print(f"❌ Failed to get initial email state: {e}")
        return set(), set()

def polling_loop():
    """Main polling loop - checks for new emails every 15 seconds"""
    print("🔗 Connecting to email server...")
    
    # Get initial state
    known_uids, initial_unseen = get_initial_email_state()
    processed_uids = set()
    
    if known_uids:
        print(f"✅ Connected successfully")
        print(f"📬 Found {len(known_uids)} existing emails ({len(initial_unseen)} unread)")
        
        # Process any existing unseen emails
        if initial_unseen:
            print(f"🔄 Processing {len(initial_unseen)} existing unread emails...")
            try:
                mail = imaplib.IMAP4_SSL(EMAIL_HOST)
                mail.login(EMAIL_USER, EMAIL_PASS)
                mail.select('inbox')
                
                for uid in initial_unseen:
                    process_email(uid, mail)
                    processed_uids.add(uid)
                    time.sleep(1)
                
                mail.close()
                mail.logout()
                print("✅ Existing emails processed")
                
            except Exception as e:
                print(f"❌ Error processing existing emails: {e}")
    else:
        print(f"✅ Connected successfully")
        print(f"📬 No existing emails found")
    
    print(f"⏳ Monitoring for new emails... (checking every 15 seconds)")
    print("-" * 50)
    
    connection_logged = True
    check_count = 0
    
    while not stop_event.is_set():
        mail = None
        try:
            # Connect to email server
            mail = imaplib.IMAP4_SSL(EMAIL_HOST)
            mail.login(EMAIL_USER, EMAIL_PASS)
            mail.select('inbox')
            
            # Get current UIDs
            status, current_data = mail.search(None, 'ALL')
            current_uids = set(current_data[0].split()) if current_data[0] else set()
            
            # Find truly NEW emails (UIDs we haven't seen before)
            new_uids = current_uids - known_uids
            
            if new_uids:
                print(f"🔔 {len(new_uids)} NEW email{'s' if len(new_uids) != 1 else ''} arrived!")
                
                for uid in new_uids:
                    if uid not in processed_uids:
                        process_email(uid, mail)
                        processed_uids.add(uid)
                        time.sleep(1)
                
                # Update known UIDs
                known_uids.update(new_uids)
                
            else:
                # Show periodic status (every 4th check = once per minute)
                check_count += 1
                if check_count % 4 == 0:
                    total_emails = len(current_uids)
                    print(f"⏳ Monitoring... ({total_emails} total emails)")
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("🔄 Retrying in 30 seconds...")
            connection_logged = False
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
    
    print("🛑 Email monitoring stopped")

def start_email_monitoring():
    """Start the email monitoring system"""
    try:
        polling_loop()
    except KeyboardInterrupt:
        print("\n🛑 Stopping email monitoring...")
        stop_event.set()
        print("✅ Email monitoring stopped")

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