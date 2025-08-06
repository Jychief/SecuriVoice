SecuriVoice - Voicemail Phishing Detector
🛡️ AI-powered voicemail analysis system that detects phishing attempts using advanced text and audio analysis
SecuriVoice helps users identify voice-based phishing (vishing) attacks by analyzing both the content and audio characteristics of suspicious voicemails. The system combines OpenAI's GPT-4o-mini for text analysis with VoiceGUARD (Wav2Vec2) for AI voice detection.
🌟 Features

📝 Advanced Text Analysis: Uses GPT-4o-mini to identify phishing tactics, urgency language, and social engineering techniques
🎵 AI Voice Detection: Employs VoiceGUARD to detect AI-generated voices commonly used in vishing attacks
📞 Caller ID Verification: Checks for phone number spoofing by comparing caller ID with numbers mentioned in voicemails
📧 Email Integration: Automatically processes voicemails sent via email and returns detailed analysis reports
🤝 Community Sharing: Optional anonymous sharing of results to help protect others from similar scams
🌐 Web Interface: Clean, responsive web interface for viewing community submissions

🚀 Quick Start
Prerequisites

Python 3.8+
OpenAI API key
Gmail account with app password (for email processing)
Audio files in supported formats (.m4a, .wav, .mp3, .flac, .ogg)

Installation

Clone the repository
bashgit clone <your-repo-url>
cd securivoice

Install dependencies
bashpip install -r requirements.txt

Set up environment variables
Create a .env file in the project root:
env# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Email Configuration (Gmail)
EMAIL_HOST=imap.gmail.com
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password

# SMTP Configuration (for sending reports)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
FROM_EMAIL=your_email@gmail.com
FROM_NAME=SecuriVoice Analysis System

# File Storage
UPLOAD_DIR=./uploads

Run the application
bashpython app.py


Using SecuriVoice
Method 1: Email Submission (Recommended)

Forward suspicious voicemails to your configured email address
Important: Include the caller's 10-digit phone number (no formatting) in the email body
Attach the voicemail audio file
Receive automated analysis report via email

Method 2: Direct Analysis
bash# Analyze text only
python text_analysis.py

# Analyze audio only
python audio_analysis.py your_voicemail.m4a

# Full transcription
python speech_to_text.py your_voicemail.m4a
🏗️ Architecture
Core Components

speech_to_text.py: OpenAI Whisper-based audio transcription
text_analysis.py: GPT-4o-mini powered phishing detection
audio_analysis.py: VoiceGUARD AI voice detection
email_handler.py: Automated email processing and monitoring
email_response.py: Analysis report generation and delivery
app.py: Flask web application for community submissions

Analysis Pipeline

Audio Transcription: Whisper converts voicemail to text
Text Analysis: GPT-4o-mini identifies phishing indicators
Audio Analysis: VoiceGUARD detects AI-generated voices
Risk Scoring: Combined analysis produces overall risk score (1-10)
Report Generation: Detailed HTML report with recommendations

📊 Risk Assessment
Risk Levels

🔴 High Risk (8-10): Strong phishing indicators detected
🟡 Medium Risk (5-7): Some suspicious elements found
🟢 Low Risk (1-4): Appears legitimate

Detection Indicators
Text Analysis:

Urgency tactics ("immediate action required")
Authority impersonation (banks, IRS, government)
Threat language (account suspension, legal action)
Information requests (personal data, passwords)
Caller ID spoofing

Audio Analysis:

AI-generated voice detection
Unnatural audio characteristics
Background noise analysis
Voice quality metrics

🤝 Community Features
Users can optionally share their analysis results anonymously to help protect others:

Receive analysis report via email
Reply to grant permission for community sharing
View shared submissions at /community endpoint
Help others recognize similar scam patterns

🔧 Configuration
Email Setup (Gmail)

Enable 2-factor authentication
Generate app password: Google Account → Security → App passwords
Use app password in .env file

Database

SQLite database automatically created (voicemail_analysis.db)
Stores analysis results, tracking codes, and community permissions
Schema automatically updates on startup

🛡️ Privacy & Security

Email Processing: Emails marked as read after processing
File Storage: Audio files stored locally in uploads directory
Community Sharing: Requires explicit user permission via email reply
Tracking Codes: Unique codes ensure only specific submissions are shared
No Personal Data: Community submissions show only voicemail content and analysis

📋 Requirements
Python Dependencies

torch - PyTorch for ML models
transformers - Hugging Face transformers for VoiceGUARD
whisper - OpenAI Whisper for speech-to-text
openai - OpenAI API client
librosa - Audio processing
flask - Web framework
python-dotenv - Environment variable management
numpy - Numerical computing
sqlite3 - Database (built-in)

See requirements.txt for complete list.
🚨 Important Notes
Phone Number Format

Always include the caller's phone number in email submissions
Use 10-digit format with no spaces, hyphens, or formatting
Example: 5551234567 (not 555-123-4567)

Supported Audio Formats

.m4a (recommended for voicemails)
.wav, .mp3, .flac, .ogg

🤖 AI Models Used

OpenAI Whisper: Speech-to-text transcription
OpenAI GPT-4o-mini: Text-based phishing detection
VoiceGUARD (Wav2Vec2): AI-generated voice detection
