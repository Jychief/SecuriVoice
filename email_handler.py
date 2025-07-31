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
from email_response import send_analysis_report, extract_tracking_code
from datetime import datetime
import logging
import sqlite3
import re

load_dotenv()

EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
UPLOAD_DIR = os.getenv('UPLOAD_DIR', './uploads')
FROM_EMAIL = os.getenv('FROM_EMAIL', EMAIL_USER)

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global stop event for clean shutdown
stop_event = Event()

# Configure logging to reduce noise
logging.getLogger('speech_to_text').setLevel(logging.WARNING)
logging.getLogger('text_analysis').setLevel(logging.WARNING)
logging.getLogger('audio_analysis').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

def update_database_schema():
    """Update database schema to include tracking code and community permission tracking"""
    try:
        db_path = "voicemail_analysis.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check which columns already exist
            cursor.execute("PRAGMA table_info(voicemail_submissions)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # Add new columns for tracking and community sharing
            new_columns = [
                ("tracking_code", "TEXT DEFAULT NULL"),
                ("community_permission", "BOOLEAN DEFAULT FALSE"),
                ("permission_granted_at", "TIMESTAMP DEFAULT NULL"),
                ("permission_email_uid", "TEXT DEFAULT NULL"),
                ("shared_to_community", "BOOLEAN DEFAULT FALSE"),
                ("shared_at", "TIMESTAMP DEFAULT NULL")
            ]
            
            columns_added = 0
            for column_name, column_def in new_columns:
                if column_name not in existing_columns:
                    try:
                        cursor.execute(f'ALTER TABLE voicemail_submissions ADD COLUMN {column_name} {column_def}')
                        columns_added += 1
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            raise
            
            conn.commit()
            
            if columns_added > 0:
                print(f"✅ Added {columns_added} new columns for tracking and community permissions")
                
    except Exception as e:
        print(f"❌ Failed to update database schema: {e}")
        raise

def save_tracking_code_to_db(db_id: int, tracking_code: str) -> bool:
    """
    Save tracking code to database for the specific submission
    
    Args:
        db_id: Database ID of the submission
        tracking_code: Generated tracking code
        
    Returns:
        True if saved successfully
    """
    try:
        db_path = "voicemail_analysis.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE voicemail_submissions 
                SET tracking_code = ?
                WHERE id = ?
            ''', (tracking_code, db_id))
            
            conn.commit()
            
            if cursor.rowcount > 0:
                print(f"✅ Tracking code {tracking_code} saved for submission ID: {db_id}")
                return True
            else:
                print(f"❌ No submission found with ID: {db_id}")
                return False
                
    except Exception as e:
        print(f"❌ Error saving tracking code: {e}")
        return False

def is_reply_to_securivoice(msg, sender_email: str) -> tuple:
    """
    Check if an email is a reply to a SecuriVoice analysis report and extract tracking code
    
    Args:
        msg: Email message object
        sender_email: Email address of the sender
        
    Returns:
        Tuple of (is_reply_boolean, tracking_code_or_none)
    """
    try:
        # Get and decode the subject line properly
        raw_subject = msg.get('Subject', '')
        
        # Decode the subject if it's encoded
        if raw_subject:
            from email.header import decode_header
            decoded_parts = decode_header(raw_subject)
            subject_parts = []
            
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        part = part.decode(encoding)
                    else:
                        part = part.decode('utf-8', errors='ignore')
                subject_parts.append(part)
            
            subject = ''.join(subject_parts)
        else:
            subject = ''
        
        print(f"🔍 Checking if reply:")
        print(f"   Raw subject: {raw_subject}")
        print(f"   Decoded subject: {subject}")
        
        if not subject.lower().startswith('re:'):
            print(f"❌ Not a reply - no 'Re:' prefix")
            return False, None
            
        if 'securivoice' not in subject.lower():
            print(f"❌ Not a SecuriVoice reply - no 'securivoice' in subject")
            return False, None
        
        print(f"✅ Subject indicates SecuriVoice reply")
        
        # Extract tracking code from subject
        tracking_code = extract_tracking_code(subject)
        
        if not tracking_code:
            print(f"❌ No tracking code found in subject")
            return False, None
        
        print(f"✅ Found tracking code: {tracking_code}")
        
        # Verify this tracking code exists in our database and matches the sender
        if verify_tracking_code_ownership(tracking_code, sender_email):
            print(f"✅ Tracking code verified for sender {sender_email}")
            return True, tracking_code
        else:
            print(f"❌ Tracking code {tracking_code} not found or doesn't belong to {sender_email}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error checking if reply to SecuriVoice: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def verify_tracking_code_ownership(tracking_code: str, sender_email: str) -> bool:
    """
    Verify that the tracking code belongs to a submission from the given sender email
    and that permission hasn't already been granted
    
    Args:
        tracking_code: Tracking code to verify
        sender_email: Email address of the sender
        
    Returns:
        True if tracking code is valid and belongs to sender
    """
    try:
        db_path = "voicemail_analysis.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Find submission with this tracking code and sender email
            cursor.execute('''
                SELECT id, phone_number, transcript, community_permission, submission_date 
                FROM voicemail_submissions 
                WHERE tracking_code = ? AND sender_email = ? AND community_permission = FALSE
            ''', (tracking_code, sender_email))
            
            result = cursor.fetchone()
            
            if result:
                print(f"✅ Found matching submission:")
                print(f"   ID: {result[0]}")
                print(f"   Phone: {result[1]}")
                print(f"   Transcript: {result[2][:50]}...")
                print(f"   Permission: {result[3]}")
                print(f"   Date: {result[4]}")
                return True
            else:
                print(f"❌ No matching submission found or permission already granted")
                
                # Check if tracking code exists at all
                cursor.execute('''
                    SELECT id, sender_email, community_permission 
                    FROM voicemail_submissions 
                    WHERE tracking_code = ?
                ''', (tracking_code,))
                
                check_result = cursor.fetchone()
                if check_result:
                    print(f"📋 Tracking code exists but:")
                    print(f"   Sender email: {check_result[1]} (expected: {sender_email})")
                    print(f"   Permission already granted: {check_result[2]}")
                else:
                    print(f"📋 Tracking code {tracking_code} not found in database")
                
                return False
            
    except Exception as e:
        print(f"❌ Error verifying tracking code: {e}")
        return False

def find_submission_by_tracking_code(tracking_code: str, sender_email: str) -> int:
    """
    Find the submission ID by tracking code and sender email
    
    Args:
        tracking_code: Tracking code from the email
        sender_email: Email address of the sender
        
    Returns:
        Database ID of the submission, or None if not found
    """
    try:
        print(f"🔍 Looking for submission with tracking code: {tracking_code}")
        
        db_path = "voicemail_analysis.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, phone_number, transcript, community_permission, submission_date 
                FROM voicemail_submissions 
                WHERE tracking_code = ? AND sender_email = ? AND community_permission = FALSE
            ''', (tracking_code, sender_email))
            
            result = cursor.fetchone()
            if result:
                print(f"✅ Found submission with tracking code:")
                print(f"   ID: {result[0]}")
                print(f"   Phone: {result[1]}")
                print(f"   Transcript: {result[2][:50]}...")
                print(f"   Permission: {result[3]}")
                print(f"   Date: {result[4]}")
                return result[0]
            else:
                print(f"❌ No submission found with tracking code {tracking_code} for {sender_email}")
                return None
            
    except Exception as e:
        print(f"❌ Error finding submission by tracking code: {e}")
        return None

def grant_community_permission(submission_id: int, email_uid: str, tracking_code: str) -> bool:
    """
    Grant community permission for a specific submission
    
    Args:
        submission_id: Database ID of the submission
        email_uid: UID of the permission email
        tracking_code: Tracking code for verification
        
    Returns:
        True if permission granted successfully
    """
    try:
        db_path = "voicemail_analysis.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE voicemail_submissions 
                SET community_permission = TRUE,
                    permission_granted_at = ?,
                    permission_email_uid = ?
                WHERE id = ? AND tracking_code = ?
            ''', (datetime.now(), email_uid, submission_id, tracking_code))
            
            conn.commit()
            
            if cursor.rowcount > 0:
                print(f"✅ Community permission granted for submission ID: {submission_id} (tracking: {tracking_code})")
                return True
            else:
                print(f"❌ No submission found with ID: {submission_id} and tracking code: {tracking_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error granting community permission: {e}")
        return False

def get_submission_details(submission_id: int) -> dict:
    """
    Get submission details for confirmation
    
    Args:
        submission_id: Database ID of the submission
        
    Returns:
        Dictionary with submission details
    """
    try:
        db_path = "voicemail_analysis.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT phone_number, transcript, text_risk_score, 
                       overall_risk_score, submission_date, tracking_code
                FROM voicemail_submissions 
                WHERE id = ?
            ''', (submission_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'phone_number': result[0],
                    'transcript': result[1],
                    'text_risk_score': result[2],
                    'overall_risk_score': result[3],
                    'submission_date': result[4],
                    'tracking_code': result[5]
                }
            return None
            
    except Exception as e:
        print(f"❌ Error getting submission details: {e}")
        return None

def send_permission_confirmation(recipient_email: str, submission_details: dict) -> bool:
    """
    Send a confirmation email when community permission is granted
    
    Args:
        recipient_email: Email address to send confirmation to
        submission_details: Details of the submission
        
    Returns:
        True if email sent successfully
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Email configuration
        SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
        SMTP_USER = os.getenv('SMTP_USER')
        SMTP_PASS = os.getenv('SMTP_PASS')
        FROM_NAME = os.getenv('FROM_NAME', 'SecuriVoice Analysis System')
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{FROM_NAME} <{FROM_EMAIL}>"
        msg['To'] = recipient_email
        
        # Include tracking code in subject
        tracking_code = submission_details.get('tracking_code', 'Unknown')
        msg['Subject'] = f"✅ SecuriVoice - Community Sharing Permission Confirmed [{tracking_code}]"
        
        # Get submission info
        phone_number = submission_details.get('phone_number', 'Unknown')
        risk_score = submission_details.get('overall_risk_score') or submission_details.get('text_risk_score', 'Unknown')
        submission_date = submission_details.get('submission_date', 'Unknown')
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
                .header {{ background-color: #28a745; color: white; padding: 20px; text-align: center; border-radius: 8px; }}
                .content {{ padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin: 20px 0; }}
                .success-box {{ background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px; border: 1px solid #c3e6cb; margin: 15px 0; }}
                .info-box {{ background-color: #e7f3ff; padding: 15px; border-radius: 8px; margin: 15px 0; }}
                .tracking-code {{ background-color: #2c3e50; color: white; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 1.1em; text-align: center; margin: 10px 0; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>✅ Community Sharing Confirmed</h1>
                <p>Thank you for helping protect others!</p>
                <div class="tracking-code">Submission: {tracking_code}</div>
            </div>
            
            <div class="content">
                <div class="success-box">
                    <h3>🤝 Permission Granted Successfully</h3>
                    <p>Your specific voicemail submission ({tracking_code}) will now be shared anonymously with the SecuriVoice community to help others identify and avoid similar scam attempts.</p>
                </div>
                
                <div class="info-box">
                    <h4>📋 Submission Details</h4>
                    <p><strong>Submission ID:</strong> {tracking_code}</p>
                    <p><strong>Phone Number:</strong> {phone_number}</p>
                    <p><strong>Risk Score:</strong> {risk_score}/10</p>
                    <p><strong>Original Submission:</strong> {submission_date}</p>
                    <p><strong>Permission Granted:</strong> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
                </div>
                
                <div class="info-box">
                    <h4>🔒 Privacy Protection</h4>
                    <p>Only this specific submission will appear on the community page with:</p>
                    <ul>
                        <li>✅ The voicemail transcript and analysis</li>
                        <li>✅ The caller's phone number</li>
                        <li>✅ Risk assessment and indicators</li>
                        <li>❌ <strong>NO personal information about you</strong></li>
                    </ul>
                    <p><em>Other submissions you may have require separate permission.</em></p>
                </div>
                
                <p>View community submissions at: <strong>https://your-domain.com/community</strong></p>
                
                <div class="success-box">
                    <p><strong>Thank you for contributing to community safety!</strong> Your submission helps others recognize and avoid voice-based scams.</p>
                </div>
            </div>
            
            <div class="footer">
                <p>SecuriVoice - Protecting communities from voice-based phishing attacks</p>
                <p>This confirmation is for submission {tracking_code} only.</p>
                <p>This is an automated confirmation email.</p>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        text_content = f"""
SecuriVoice - Community Sharing Permission Confirmed
Submission: {tracking_code}

Thank you for helping protect others!

Your specific voicemail submission ({tracking_code}) will now be shared anonymously with the SecuriVoice community.

Submission Details:
- Submission ID: {tracking_code}
- Phone Number: {phone_number}
- Risk Score: {risk_score}/10
- Original Submission: {submission_date}
- Permission Granted: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}

Privacy Protection:
Only this specific submission will include the voicemail content, phone number, and analysis results, but NO personal information about you will be shared.

Other submissions you may have require separate permission.

View community submissions: https://your-domain.com/community

Thank you for contributing to community safety!

SecuriVoice Team
        """
        
        # Attach both versions
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        
        print(f"✅ Permission confirmation sent to {recipient_email} for {tracking_code}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send permission confirmation: {e}")
        return False

def process_permission_reply(uid, mail, sender_email: str, tracking_code: str):
    """Process a reply email granting community permission for specific submission"""
    print(f"🤝 Processing permission reply from: {sender_email}")
    print(f"🔍 For tracking code: {tracking_code}")
    
    try:
        # Find the specific submission by tracking code
        print(f"🔍 Step 1: Finding submission by tracking code...")
        submission_id = find_submission_by_tracking_code(tracking_code, sender_email)
        
        if not submission_id:
            print(f"❌ No pending submission found for tracking code {tracking_code}")
            print(f"📧 This submission may already have permission or the tracking code is invalid")
            return
        
        print(f"✅ Found submission ID: {submission_id}")
        
        # Grant permission for this specific submission
        print(f"🔍 Step 2: Granting permission...")
        if grant_community_permission(submission_id, uid.decode(), tracking_code):
            print(f"✅ Permission granted successfully")
            
            # Get submission details for confirmation
            print(f"🔍 Step 3: Getting submission details...")
            submission_details = get_submission_details(submission_id)
            
            if submission_details:
                print(f"✅ Retrieved submission details")
                
                # Send confirmation email
                print(f"🔍 Step 4: Sending confirmation email...")
                confirmation_sent = send_permission_confirmation(sender_email, submission_details)
                
                if confirmation_sent:
                    print(f"✅ Confirmation email sent successfully")
                    print(f"🤝 Community permission granted for submission {submission_id} (tracking: {tracking_code})")
                else:
                    print(f"❌ Failed to send confirmation email")
            else:
                print(f"❌ Could not retrieve submission details for ID: {submission_id}")
        else:
            print(f"❌ Failed to grant permission for submission {submission_id}")
            
        # Mark reply email as read
        print(f"🔍 Step 5: Marking reply email as read...")
        mail.store(uid, '+FLAGS', '\\Seen')
        print(f"✅ Reply email marked as read")
        
    except Exception as e:
        print(f"❌ Error processing permission reply: {e}")
        import traceback
        traceback.print_exc()

def process_email(uid, mail):
    """Process a single email and extract voicemail data or handle permission replies"""
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
        
        # Extract email address from "Name <email@domain.com>" format
        email_match = re.search(r'<([^>]+)>', sender_email)
        if email_match:
            clean_sender_email = email_match.group(1)
        else:
            clean_sender_email = sender_email
        
        print(f"   From: {clean_sender_email}")
        print(f"   Subject: {subject}")

        # Check if this is a reply to SecuriVoice (permission grant)
        is_reply, tracking_code = is_reply_to_securivoice(msg, clean_sender_email)
        
        if is_reply and tracking_code:
            print(f"🤝 Detected permission reply for tracking code: {tracking_code}")
            process_permission_reply(uid, mail, clean_sender_email, tracking_code)
            return
        else:
            print("📧 Not a permission reply, processing as new voicemail submission")

        # Original voicemail processing logic
        phone_number = None
        saved_file = None

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
                db_id, text_analysis = analyze_voicemail_text(transcript, phone_number, saved_file, clean_sender_email)
                
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
                
                # Send analysis report email back to sender (with tracking code)
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
                    
                    # Send the analysis report email and get tracking code
                    email_sent, tracking_code = send_analysis_report(clean_sender_email, voicemail_data, db_id)
                    
                    if email_sent and tracking_code:
                        print(f"✅ Report sent to {clean_sender_email}")
                        print(f"🔍 Tracking code: {tracking_code}")
                        
                        # Save tracking code to database
                        save_tracking_code_to_db(db_id, tracking_code)
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

def polling_loop():
    """Main polling loop - checks for new emails every 15 seconds"""
    print("🚀 Starting email monitoring...")
    
    # Update database schema for tracking and community permissions
    update_database_schema()
    
    last_email_count = 0
    processed_uids = set()
    connection_logged = False
    check_count = 0
    
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
                print(f"🔔 Found {new_emails} total email{'s' if new_emails != 1 else ''}!")
                
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
                check_count += 1
                if check_count % 4 == 0:  # Show once per minute (every 4 checks)
                    print(f"⏳ Monitoring... Total emails: {current_count}")
            else:
                # Email count decreased (emails were deleted)
                print(f"📉 Email count changed: {last_email_count} → {current_count}")
                last_email_count = current_count
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("🔄 Retrying in 30 seconds...")
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
        print("Or reply to a SecuriVoice report to test permission granting!")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        try:
            polling_loop()
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            stop_event.set()
            print("✅ Stopped")