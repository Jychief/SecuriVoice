import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv
import traceback

load_dotenv()

# Email configuration for sending responses
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USER = os.getenv('SMTP_USER')  # Your sending email
SMTP_PASS = os.getenv('SMTP_PASS')  # Your app password
FROM_EMAIL = os.getenv('FROM_EMAIL', SMTP_USER)
FROM_NAME = os.getenv('FROM_NAME', 'SecuriVoice Analysis System')

def generate_analysis_report(voicemail_data: dict) -> str:
    """
    Generate HTML analysis report template
    
    Args:
        voicemail_data: Dictionary containing voicemail information
        
    Returns:
        HTML formatted report string
    """
    
    # Extract data with defaults
    phone_number = voicemail_data.get('phone_number', 'Not provided')
    transcribed_text = voicemail_data.get('transcribed_text', 'Transcription failed')
    file_name = voicemail_data.get('file_name', 'Unknown file')
    processed_at = voicemail_data.get('processed_at', datetime.now())
    
    # TODO: Replace these with actual analysis results
    risk_score = 0.75  # Placeholder - will come from analysis module
    risk_level = "MEDIUM"  # Placeholder
    suspicious_indicators = [
        "Urgency language detected",
        "Request for personal information", 
        "Unfamiliar caller ID"
    ]  # Placeholder
    
    # Format the processed date
    if isinstance(processed_at, str):
        processed_date = processed_at
    else:
        processed_date = processed_at.strftime("%B %d, %Y at %I:%M %p")
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; border-radius: 8px; }}
            .content {{ padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin: 20px 0; }}
            .risk-score {{ font-size: 24px; font-weight: bold; text-align: center; padding: 15px; border-radius: 8px; }}
            .risk-medium {{ background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }}
            .risk-high {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
            .risk-low {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
            .section {{ margin: 20px 0; padding: 15px; background-color: white; border-radius: 8px; border-left: 4px solid #3498db; }}
            .indicator {{ background-color: #fff3cd; padding: 8px; margin: 5px 0; border-radius: 4px; border-left: 3px solid #ffc107; }}
            .transcription {{ background-color: #e9ecef; padding: 15px; border-radius: 8px; font-style: italic; }}
            .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            .warning {{ background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛡️ SecuriVoice Analysis Report</h1>
            <p>Voicemail Phishing Detection Results</p>
        </div>
        
        <div class="content">
            <div class="risk-score risk-{risk_level.lower()}">
                Risk Score: {risk_score:.0%} ({risk_level} RISK)
            </div>
            
            <div class="section">
                <h3>📞 Voicemail Details</h3>
                <p><strong>Caller Phone Number:</strong> {phone_number}</p>
                <p><strong>File Analyzed:</strong> {file_name}</p>
                <p><strong>Analysis Date:</strong> {processed_date}</p>
            </div>
            
            <div class="section">
                <h3>📝 Transcription</h3>
                <div class="transcription">
                    "{transcribed_text}"
                </div>
            </div>
            
            <div class="section">
                <h3>⚠️ Suspicious Indicators Detected</h3>
                {"".join(f'<div class="indicator">• {indicator}</div>' for indicator in suspicious_indicators)}
            </div>
            
            <div class="section">
                <h3>🎯 Analysis Summary</h3>
                <p>Based on our analysis, this voicemail shows <strong>{risk_level.lower()} risk</strong> characteristics commonly associated with phishing attempts.</p>
                
                {'''<div class="warning">
                    <strong>⚠️ CAUTION:</strong> This voicemail contains multiple red flags typically seen in phishing scams. 
                    Do not provide personal information or call back using the number provided in the message.
                </div>''' if risk_score > 0.6 else ''}
                
                <h4>Recommendations:</h4>
                <ul>
                    <li>🚫 Do not call back using the number provided in the voicemail</li>
                    <li>🏦 Contact your bank/institution directly using official numbers</li>
                    <li>🔍 Verify any claims through official channels</li>
                    <li>🗑️ Delete the voicemail if confirmed as phishing</li>
                </ul>
            </div>
            
            <div class="section">
                <h3>🤖 About This Analysis</h3>
                <p>This report was generated by SecuriVoice, an AI-powered voicemail phishing detection system. 
                The analysis combines speech pattern recognition, content analysis, and known phishing indicators 
                to assess the likelihood that a voicemail is a phishing attempt.</p>
                
                <p><em>Note: This is an automated analysis. Always use your judgment and verify through official channels 
                when dealing with financial or personal information requests.</em></p>
            </div>
        </div>
        
        <div class="footer">
            <p>SecuriVoice - Protecting you from voice-based phishing attacks</p>
            <p>This report was generated automatically. Please do not reply to this email.</p>
        </div>
    </body>
    </html>
    """
    
    return html_report

def send_analysis_report(recipient_email: str, voicemail_data: dict) -> bool:
    """
    Send analysis report via email
    
    Args:
        recipient_email: Email address to send report to
        voicemail_data: Dictionary containing voicemail analysis data
        
    Returns:
        True if email sent successfully, False otherwise
    """
    
    try:
        # Generate the report
        html_report = generate_analysis_report(voicemail_data)
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{FROM_NAME} <{FROM_EMAIL}>"
        msg['To'] = recipient_email
        msg['Subject'] = "🛡️ SecuriVoice Analysis Report - Your Voicemail Analysis Results"
        
        # Create plain text version
        text_content = f"""
SecuriVoice Analysis Report

Your voicemail from {voicemail_data.get('phone_number', 'Unknown')} has been analyzed.

Transcription: {voicemail_data.get('transcribed_text', 'Transcription failed')}

Risk Assessment: MEDIUM RISK (75%)

This voicemail shows characteristics commonly associated with phishing attempts.

Recommendations:
- Do not call back using the number provided
- Contact institutions directly using official numbers
- Verify any claims through official channels

This is an automated analysis from SecuriVoice.
        """
        
        # Attach both versions
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_report, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        print(f"📧 Sending analysis report to {recipient_email}...")
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        
        print(f"✅ Analysis report sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send analysis report to {recipient_email}: {e}")
        traceback.print_exc()
        return False

def test_email_system():
    """Test the email response system"""
    
    # Test data
    test_voicemail_data = {
        'phone_number': '833-622-3359',
        'transcribed_text': 'Hello, this is Mia Hayes from Spartan Finance. This is a reminder that you are currently pre-approved for a personal loan up to $60,000. After reviewing your credit profile, I believe this is a perfect time to take advantage of this offer. You can reach me directly at 833-622-3359. I will be looking forward to your call.',
        'file_name': 'test_voicemail.m4a',
        'processed_at': datetime.now()
    }
    
    # Generate report (don't send, just preview)
    report = generate_analysis_report(test_voicemail_data)
    
    # Save to file for preview
    with open('sample_report.html', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Sample report generated: sample_report.html")
    print("🔍 Open this file in your browser to preview the report")
    
    # Uncomment to test actual email sending:
    # return send_analysis_report("your-test-email@example.com", test_voicemail_data)

if __name__ == "__main__":
    test_email_system()