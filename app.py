from flask import Flask, render_template, send_file, abort
from email_handler import start_email_monitoring
from text_analysis import TextAnalyzer
import threading
import sqlite3
import json
import os
from datetime import datetime

app = Flask(__name__)

def get_community_submissions(limit: int = 20):
    """
    Get community submissions that have permission granted
    
    Args:
        limit: Maximum number of submissions to return
        
    Returns:
        List of submission dictionaries
    """
    try:
        db_path = "voicemail_analysis.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get submissions with community permission
            cursor.execute('''
                SELECT id, phone_number, transcript, text_risk_score,
                       text_analysis, text_indicators, caller_id_mismatch,
                       transcript_numbers, audio_risk_score, 
                       overall_risk_score, submission_date,
                       audio_analysis, audio_indicators, audio_confidence,
                       is_ai_generated, permission_granted_at, audio_file_path
                FROM voicemail_submissions 
                WHERE community_permission = TRUE
                ORDER BY permission_granted_at DESC 
                LIMIT ?
            ''', (limit,))
            
            submissions = []
            for row in cursor.fetchall():
                # Parse JSON fields safely
                try:
                    text_indicators = json.loads(row[5]) if row[5] else []
                except:
                    text_indicators = []
                
                try:
                    transcript_numbers = json.loads(row[7]) if row[7] else []
                except:
                    transcript_numbers = []
                
                try:
                    audio_indicators = row[12].split(', ') if row[12] else []
                except:
                    audio_indicators = []
                
                # Calculate final risk score
                text_risk = row[3] or 0
                audio_risk = row[8] or 0
                overall_risk = row[9] or text_risk
                
                # Format phone number
                phone_number = row[1] or "Unknown"
                clean_number = ''.join(filter(str.isdigit, phone_number))
                if len(clean_number) == 10:
                    formatted_phone = f"({clean_number[:3]}) {clean_number[3:6]}-{clean_number[6:]}"
                else:
                    formatted_phone = phone_number
                
                # Format submission date
                try:
                    if isinstance(row[10], str):
                        sub_date = datetime.fromisoformat(row[10].replace('Z', '+00:00'))
                    else:
                        sub_date = row[10]
                    formatted_date = sub_date.strftime("%B %d, %Y")
                except:
                    formatted_date = "Unknown date"
                
                # Determine AI status
                if row[14]:  # is_ai_generated
                    confidence_pct = f"{row[13]:.1%}" if row[13] else "Unknown"
                    ai_status = f"AI-Generated Voice (Confidence: {confidence_pct})"
                    ai_class = "ai-detected"
                elif audio_risk:
                    ai_status = "Human Voice Detected"
                    ai_class = "human-detected"
                else:
                    ai_status = "Voice Analysis Pending"
                    ai_class = "analysis-pending"
                
                # Get risk class
                if overall_risk >= 8:
                    risk_class = "risk-high"
                elif overall_risk >= 5:
                    risk_class = "risk-medium"
                else:
                    risk_class = "risk-low"
                
                submissions.append({
                    'id': row[0],
                    'phone_number': formatted_phone,
                    'transcript': row[2],
                    'text_risk_score': text_risk,
                    'text_analysis': row[4] or 'No analysis available',
                    'text_indicators': text_indicators,
                    'caller_id_mismatch': bool(row[6]),
                    'transcript_numbers': transcript_numbers,
                    'audio_risk_score': audio_risk,
                    'overall_risk_score': overall_risk,
                    'submission_date': formatted_date,
                    'audio_analysis': row[11] or 'No audio analysis available',
                    'audio_indicators': audio_indicators,
                    'audio_confidence': row[13] or 0,
                    'is_ai_generated': bool(row[14]) if row[14] is not None else False,
                    'permission_granted_at': row[15],
                    'audio_file_path': row[16],  # Include audio file path
                    'ai_status': ai_status,
                    'ai_class': ai_class,
                    'risk_class': risk_class
                })
            
            return submissions
            
    except Exception as e:
        print(f"❌ Failed to get community submissions: {e}")
        return []

@app.route('/')
def home():
    """Serve the main homepage"""
    return render_template('home.html')

@app.route('/community')
def community():
    """Serve the community submissions page with real data"""
    submissions = get_community_submissions()
    return render_template('community.html', submissions=submissions)

@app.route('/api/audio/<int:submission_id>')
def get_audio(submission_id):
    """Serve audio files for community submissions with permission"""
    try:
        db_path = "voicemail_analysis.db"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get audio file path for submissions with community permission
            cursor.execute('''
                SELECT audio_file_path FROM voicemail_submissions 
                WHERE id = ? AND community_permission = TRUE
            ''', (submission_id,))
            
            result = cursor.fetchone()
            if not result or not result[0]:
                abort(404)
            
            audio_file_path = result[0]
            
            # Check if file exists
            if not os.path.exists(audio_file_path):
                abort(404)
            
            # Determine MIME type based on file extension
            file_ext = os.path.splitext(audio_file_path)[1].lower()
            mime_type_map = {
                '.m4a': 'audio/mp4',
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.flac': 'audio/flac',
                '.ogg': 'audio/ogg'
            }
            
            mime_type = mime_type_map.get(file_ext, 'audio/mpeg')
            
            return send_file(audio_file_path, mimetype=mime_type)
            
    except Exception as e:
        print(f"❌ Error serving audio file: {e}")
        abort(500)

if __name__ == "__main__":
    print("🚀 Starting SecuriVoice")
    print("📝 Make sure your .env file is configured")
    
    # Initialize text analysis database
    print("💾 Initializing database...")
    try:
        analyzer = TextAnalyzer()
        print("✅ Database ready")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        print("❌ Exiting...")
        exit(1)
        
    # Start email monitoring in background thread
    email_thread = threading.Thread(target=start_email_monitoring, daemon=True)
    email_thread.start()
    
    print("✅ Email monitoring started")
    print("🌐 Homepage: http://localhost:5000")
    print("🤝 Community: http://localhost:5000/community")
    
    app.run(debug=False, host='0.0.0.0', port=5000)