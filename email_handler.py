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
    print(f"\n📧 Processing email with UID: {uid.decode()}")
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
                print(f"🎤 Converting speech-to-text...")
                # Convert audio to text using Whisper
                transcript = transcribe_audio(saved_file)
                print(f"📝 Transcript ({len(transcript)} chars): {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
                
                # Analyze the transcript for phishing indicators
                print(f"🔍 Analyzing text for phishing indicators...")
                db_id, text_analysis = analyze_voicemail_text(transcript, phone_number, saved_file)
                
                print(f"📊 Text Analysis - Risk: {text_analysis.risk_score}/10 | Indicators: {len(text_analysis.indicators)} | DB ID: {db_id}")
                
                # Analyze audio for AI-generated voice detection
                print(f"🎵 Analyzing audio for AI voice detection...")
                try:
                    overall_risk_score, audio_analysis = analyze_voicemail_audio(saved_file, db_id)
                    
                    ai_status = "AI" if audio_analysis.is_ai_generated else "Human"
                    print(f"🤖 Audio Analysis - Voice: {ai_status} ({audio_analysis.confidence_score:.3f}) | Risk: {audio_analysis.risk_score}/10 | Overall: {overall_risk_score}/10")
                    
                except Exception as audio_error:
                    print(f"⚠️ Audio analysis failed: {audio_error}")
                    overall_risk_score = text_analysis.risk_score
                    audio_analysis = None
                
                # Send analysis report email back to sender
                try:
                    print(f"📤 Sending analysis report...")
                    
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
                        print(f"❌ Failed to send report to {sender_email}")
                        
                except Exception as email_error:
                    print(f"❌ Email sending error: {email_error}")
                
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
        print(f"✅ Completed processing\n")
        
    except Exception as e:
        print(f"❌ Error processing email {uid}: {e}")
        traceback.print_exc()

def polling_loop():
    """Main polling loop - checks for new emails every 15 seconds"""
    print("🚀 Starting email polling system...")
    
    last_email_count = 0
    processed_uids = set()
    connection_logged = False
    
    while not stop_event.is_set():
        mail = None
        try:
            # Connect to email server (only log first connection)
            if not connection_logged:
                print("📧 Connecting to email server...")
            mail = imaplib.IMAP4_SSL(EMAIL_HOST)
            mail.login(EMAIL_USER, EMAIL_PASS)
            mail.select('inbox')
            
            if not connection_logged:
                print("✅ Connected successfully")
                connection_logged = True
            
            # Get current email count
            status, data = mail.search(None, 'ALL')
            current_count = len(data[0].split()) if data[0] else 0
            
            # Check if we have new emails
            if current_count > last_email_count:
                new_emails = current_count - last_email_count
                print(f"🔔 Found {new_emails} new email{'s' if new_emails != 1 else ''}! Total: {current_count}")
                
                # Get all unseen emails
                status, unseen_data = mail.search(None, 'UNSEEN')
                if unseen_data[0]:
                    unseen_uids = unseen_data[0].split()
                    print(f"📬 Processing {len(unseen_uids)} unseen email{'s' if len(unseen_uids) != 1 else ''}...")
                    
                    for uid in unseen_uids:
                        if uid not in processed_uids:
                            process_email(uid, mail)
                            processed_uids.add(uid)
                            time.sleep(1)  # Brief pause between emails
                
                last_email_count = current_count
                
            elif current_count == last_email_count:
                # Only show waiting message every 4th check (every minute) to reduce spam
                if int(time.time()) % 60 < 15:  # Show once per minute
                    print(f"⏳ No new emails. Total: {current_count} (checking every 15s...)")
            else:
                # Email count decreased (emails were deleted)
                print(f"📉 Email count changed: {last_email_count} → {current_count}")
                last_email_count = current_count
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("🔄 Will retry in 30 seconds...")
            connection_logged = False  # Reset to log reconnection
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