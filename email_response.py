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

def get_risk_level(risk_score: int) -> tuple:
    """
    Convert risk score to level and color
    
    Args:
        risk_score: Risk score from 1-10
        
    Returns:
        Tuple of (risk_level, risk_class)
    """
    if risk_score >= 8:
        return "HIGH", "high"
    elif risk_score >= 5:
        return "MEDIUM", "medium"
    else:
        return "LOW", "low"

def generate_analysis_report(voicemail_data: dict) -> str:
    """
    Generate HTML analysis report template with audio analysis and community sharing info
    
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
    
    # Get text analysis results
    text_risk_score = voicemail_data.get('risk_score', 5)
    indicators = voicemail_data.get('indicators', [])
    explanation = voicemail_data.get('explanation', 'No explanation available')
    
    # Get audio analysis results (NEW)
    audio_analysis = voicemail_data.get('audio_analysis')
    has_audio_analysis = voicemail_data.get('has_audio_analysis', False)
    overall_risk_score = voicemail_data.get('overall_risk_score', text_risk_score)
    
    # Use overall risk score for display if available, otherwise text risk score
    display_risk_score = overall_risk_score if has_audio_analysis else text_risk_score
    risk_level, risk_class = get_risk_level(display_risk_score)
    
    # Format the processed date
    if isinstance(processed_at, str):
        processed_date = processed_at
    else:
        processed_date = processed_at.strftime("%B %d, %Y at %I:%M %p")
    
    # Generate audio analysis section
    audio_section = ""
    if has_audio_analysis and audio_analysis:
        ai_status = "AI-Generated" if audio_analysis.is_ai_generated else "Human Voice"
        confidence_pct = f"{audio_analysis.confidence_score:.1%}"
        
        audio_section = f"""
        <div class="section">
            <h3>🎵 Audio Analysis (VoiceGUARD AI Detection)</h3>
            <div class="audio-analysis-box">
                <div class="audio-result {'ai-detected' if audio_analysis.is_ai_generated else 'human-detected'}">
                    <h4>Voice Type: {ai_status}</h4>
                    <p><strong>Confidence:</strong> {confidence_pct}</p>
                    <p><strong>Audio Risk Score:</strong> {audio_analysis.risk_score}/10</p>
                </div>
                
                {'<div class="audio-indicators"><h4>Audio Indicators:</h4>' + "".join(f'<div class="indicator">• {indicator}</div>' for indicator in audio_analysis.indicators) + '</div>' if audio_analysis.indicators else ''}
                
                <div class="audio-explanation">
                    <strong>Audio Analysis:</strong> {audio_analysis.explanation}
                </div>
            </div>
        </div>
        """
    
    # Generate combined risk breakdown
    risk_breakdown_section = ""
    if has_audio_analysis:
        risk_breakdown_section = f"""
        <div class="risk-breakdown-detailed">
            <h4>Risk Score Breakdown:</h4>
            <div class="breakdown-item">📝 <strong>Text Analysis:</strong> {text_risk_score}/10</div>
            <div class="breakdown-item">🎵 <strong>Audio Analysis:</strong> {audio_analysis.risk_score if audio_analysis else 'N/A'}/10</div>
            <div class="breakdown-item total">🎯 <strong>Overall Risk:</strong> {overall_risk_score}/10</div>
            <p class="breakdown-note">Overall score is calculated as: 60% text analysis + 40% audio analysis</p>
        </div>
        """
    else:
        risk_breakdown_section = f"""
        <div class="risk-breakdown-detailed">
            <h4>Risk Score Breakdown:</h4>
            <div class="breakdown-item">📝 <strong>Text Analysis:</strong> {text_risk_score}/10</div>
            <div class="breakdown-item">🎵 <strong>Audio Analysis:</strong> Not available</div>
            <div class="breakdown-item total">🎯 <strong>Total Risk:</strong> {text_risk_score}/10</div>
            <p class="breakdown-note">Audio analysis requires additional processing time and may be included in future updates.</p>
        </div>
        """
    
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
            .explanation {{ background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
            
            /* Audio analysis styles */
            .audio-analysis-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; }}
            .audio-result {{ padding: 15px; border-radius: 8px; margin-bottom: 15px; text-align: center; }}
            .ai-detected {{ background-color: #f8d7da; color: #721c24; border: 2px solid #dc3545; }}
            .human-detected {{ background-color: #d4edda; color: #155724; border: 2px solid #28a745; }}
            .audio-indicators {{ margin: 15px 0; }}
            .audio-explanation {{ background-color: #e7f3ff; padding: 10px; border-radius: 6px; margin-top: 15px; }}
            .risk-breakdown-detailed {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }}
            .breakdown-item {{ padding: 8px 0; border-bottom: 1px solid #dee2e6; }}
            .breakdown-item.total {{ font-weight: bold; border-bottom: none; color: #2c3e50; }}
            .breakdown-note {{ font-size: 0.9rem; color: #666; font-style: italic; margin-top: 10px; }}
            
            /* NEW: Community sharing section styles */
            .community-section {{ background-color: #e7f9ff; color: #004085; padding: 20px; border-radius: 8px; margin: 20px 0; border: 2px solid #b3d9ff; }}
            .community-highlight {{ background-color: #cce7ff; padding: 15px; border-radius: 6px; margin: 15px 0; text-align: center; }}
            .reply-instruction {{ background-color: #fff; padding: 15px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #007bff; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛡️ SecuriVoice Analysis Report</h1>
            <p>Voicemail Phishing Detection Results</p>
        </div>
        
        <div class="content">
            <div class="risk-score risk-{risk_class}">
                Overall Risk Score: {display_risk_score}/10 ({risk_level} RISK)
            </div>
            
            {risk_breakdown_section}
            
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
                <h3>📊 Text Analysis</h3>
                {'<h4>⚠️ Suspicious Text Indicators Detected</h4>' + "".join(f'<div class="indicator">• {indicator}</div>' for indicator in indicators) if indicators else '<h4>✅ No Major Text Indicators Detected</h4><p>No significant phishing indicators were found in the voicemail text.</p>'}
                
                <div class="explanation">
                    <strong>Text Analysis:</strong> {explanation}
                    {f'<br><br><strong>⚠️ Caller ID Mismatch:</strong> The caller ID does not match any phone numbers mentioned in the voicemail (Found: {", ".join(voicemail_data.get("transcript_numbers", []))}). This suggests phone number spoofing.' if voicemail_data.get("caller_id_mismatch", False) else ''}
                </div>
            </div>
            
            {audio_section}
            
            <div class="section">
                <h3>🎯 Recommendations</h3>
                
                {f'''<div class="warning">
                    <strong>⚠️ HIGH RISK ALERT:</strong> This voicemail shows strong indicators of a phishing attempt. 
                    {'The use of AI-generated voice technology is particularly concerning as this is a common tactic in modern vishing attacks.' if has_audio_analysis and audio_analysis and audio_analysis.is_ai_generated else 'Exercise extreme caution and do not provide any personal information.'}
                </div>
                <h4>Immediate Actions:</h4>
                <ul>
                    <li>🚫 <strong>DO NOT</strong> call back using the number provided in the voicemail</li>
                    <li>🚫 <strong>DO NOT</strong> provide any personal or financial information</li>
                    <li>🏦 Contact your bank/institution directly using official numbers from their website</li>
                    <li>🗑️ Delete this voicemail immediately</li>
                    <li>📢 Consider reporting this number to authorities</li>
                    {'<li>🤖 Be aware that AI voice technology is being used - this may sound very convincing</li>' if has_audio_analysis and audio_analysis and audio_analysis.is_ai_generated else ''}
                </ul>''' if display_risk_score >= 8 else f'''<h4>Recommended Actions:</h4>
                <ul>
                    <li>🔍 Verify any claims through official channels before taking action</li>
                    <li>🏦 Contact institutions directly using official numbers from their website</li>
                    <li>⚠️ Be cautious about providing personal information</li>
                    {'<li>🗑️ Consider deleting this voicemail if you determine it is suspicious</li>' if display_risk_score >= 5 else '<li>✅ This appears to be a legitimate message, but always verify when in doubt</li>'}
                    {'<li>🤖 Note: AI voice detection indicates this may be synthetically generated</li>' if has_audio_analysis and audio_analysis and audio_analysis.is_ai_generated else ''}
                </ul>'''}
            </div>
            
            <!-- NEW: Community Sharing Section -->
            <div class="community-section">
                <h3>🤝 Help Protect Others - Community Submissions</h3>
                <p>Your analysis has been completed and is helping protect you from potential scams. You can also help protect others by sharing your submission anonymously with the SecuriVoice community.</p>
                
                <div class="community-highlight">
                    <h4>📊 Community Benefits</h4>
                    <p>Anonymous submissions help others recognize similar scam patterns and improve our detection algorithms.</p>
                </div>
                
                <div class="reply-instruction">
                    <h4>🔄 How to Share Your Submission</h4>
                    <p><strong>Simply reply to this email with any message to give permission for anonymous sharing.</strong></p>
                    <ul>
                        <li>✅ Your submission will be posted anonymously (no personal information shared)</li>
                        <li>✅ Only the voicemail content, phone number, and analysis results will be shown</li>
                        <li>✅ This helps others learn to identify similar scam attempts</li>
                    </ul>
                    <p><em>No reply needed if you prefer to keep your submission private.</em></p>
                </div>
                
                <p>View community submissions at: <strong>https://your-domain.com/community</strong></p>
            </div>
            
            <div class="section">
                <h3>🤖 About This Analysis</h3>
                <p>This report was generated by SecuriVoice, an AI-powered voicemail phishing detection system. 
                The analysis combines speech-to-text conversion with advanced AI analysis to identify patterns and 
                language commonly used in phishing attempts{', plus VoiceGUARD AI voice detection to identify synthetically generated voices commonly used in modern vishing attacks' if has_audio_analysis else ''}.</p>
                
                <p><strong>Analysis Components:</strong></p>
                <ul>
                    <li>📝 <strong>Text Analysis:</strong> OpenAI GPT-4o-mini analyzes transcript content for phishing indicators</li>
                    {'<li>🎵 <strong>Audio Analysis:</strong> VoiceGUARD (Wav2Vec2) detects AI-generated voices and audio anomalies</li>' if has_audio_analysis else '<li>🎵 <strong>Audio Analysis:</strong> Currently processing (may be included in future reports)</li>'}
                    <li>📞 <strong>Caller ID Verification:</strong> Checks for phone number spoofing attempts</li>
                </ul>
                
                <p><em>Note: This is an automated analysis powered by AI. While highly accurate, always use your 
                judgment and verify through official channels when dealing with financial or personal information requests.</em></p>
            </div>
        </div>
        
        <div class="footer">
            <p>SecuriVoice - Protecting you from voice-based phishing attacks</p>
            <p>Enhanced with AI voice detection technology</p>
            <p><strong>Reply to this email to share your submission anonymously with the community</strong></p>
            <p>This report was generated automatically.</p>
        </div>
    </body>
    </html>
    """
    
    return html_report

def send_analysis_report(recipient_email: str, voicemail_data: dict) -> bool:
    """
    Send analysis report via email (UPDATED with community sharing info)
    
    Args:
        recipient_email: Email address to send report to
        voicemail_data: Dictionary containing voicemail analysis data
        
    Returns:
        True if email sent successfully, False otherwise
    """
    
    try:
        # Generate the report
        html_report = generate_analysis_report(voicemail_data)
        
        # Get risk level for subject line (use overall risk if available)
        has_audio_analysis = voicemail_data.get('has_audio_analysis', False)
        display_risk_score = voicemail_data.get('overall_risk_score') if has_audio_analysis else voicemail_data.get('risk_score', 5)
        risk_level, _ = get_risk_level(display_risk_score)
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{FROM_NAME} <{FROM_EMAIL}>"
        msg['To'] = recipient_email
        
        # Enhanced subject line
        subject_prefix = "🛡️ SecuriVoice Analysis Report"
        if has_audio_analysis:
            audio_analysis = voicemail_data.get('audio_analysis')
            if audio_analysis and audio_analysis.is_ai_generated:
                subject_prefix += " [AI VOICE DETECTED]"
        
        msg['Subject'] = f"{subject_prefix} - {risk_level} Risk Detected"
        
        # Create enhanced plain text version with community info
        indicators_text = "\n".join([f"- {indicator}" for indicator in voicemail_data.get('indicators', [])])
        explanation = voicemail_data.get('explanation', 'No explanation available')
        caller_id_mismatch = voicemail_data.get('caller_id_mismatch', False)
        
        # Add audio analysis to plain text
        audio_text = ""
        if has_audio_analysis:
            audio_analysis = voicemail_data.get('audio_analysis')
            if audio_analysis:
                ai_status = "AI-Generated" if audio_analysis.is_ai_generated else "Human Voice"
                confidence_pct = f"{audio_analysis.confidence_score:.1%}"
                audio_text = f"""
Audio Analysis (VoiceGUARD):
- Voice Type: {ai_status}
- Confidence: {confidence_pct}
- Audio Risk Score: {audio_analysis.risk_score}/10
- Audio Indicators: {', '.join(audio_analysis.indicators)}
- Audio Analysis: {audio_analysis.explanation}
"""
        
        text_content = f"""
SecuriVoice Analysis Report

Your voicemail from {voicemail_data.get('phone_number', 'Unknown')} has been analyzed.

Overall Risk Score: {display_risk_score}/10 ({risk_level} RISK)

Risk Breakdown:
- Text Analysis: {voicemail_data.get('risk_score', 'N/A')}/10
- Audio Analysis: {voicemail_data.get('audio_analysis').risk_score if has_audio_analysis and voicemail_data.get('audio_analysis') else 'N/A'}/10

Transcription: {voicemail_data.get('transcribed_text', 'Transcription failed')}

Text Indicators Detected:
{indicators_text if indicators_text else 'No major text indicators detected'}

Text Analysis: {explanation}

{audio_text}

{f'CALLER ID MISMATCH: The caller ID does not match phone numbers mentioned in the voicemail. This suggests phone number spoofing.' if caller_id_mismatch else ''}

Recommendations:
- Verify any claims through official channels
- Contact institutions directly using official numbers
- Be cautious about providing personal information
{'- Be aware that AI voice technology may have been used' if has_audio_analysis and voicemail_data.get('audio_analysis') and voicemail_data.get('audio_analysis').is_ai_generated else ''}

=== HELP PROTECT OTHERS ===
Want to help others avoid similar scams? Simply REPLY to this email with any message to share your submission anonymously with the community. Your personal information will never be shared.

View community submissions: https://your-domain.com/community

This is an automated analysis from SecuriVoice with enhanced AI voice detection.
        """
        
        # Attach both versions
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_report, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to send analysis report to {recipient_email}: {e}")
        traceback.print_exc()
        return False

def test_email_system():
    """Test the email response system with audio analysis"""
    
    # Test data with mock audio analysis
    class MockAudioAnalysis:
        def __init__(self):
            self.is_ai_generated = True
            self.confidence_score = 0.92
            self.risk_score = 8
            self.indicators = ['HIGH CONFIDENCE AI-GENERATED VOICE', 'UNNATURAL SILENCE/NO BACKGROUND NOISE']
            self.explanation = 'VoiceGUARD detected this as an AI-generated voice with 92% confidence. AI-generated voices are commonly used in vishing attacks to impersonate legitimate organizations. The high confidence suggests this is very likely a synthetic voice used for fraudulent purposes.'
    
    test_voicemail_data = {
        'phone_number': '833-622-3359',
        'transcribed_text': 'Hello, this is Mia Hayes from Spartan Finance. This is a reminder that you are currently pre-approved for a personal loan up to $60,000. After reviewing your credit profile, I believe this is a perfect time to take advantage of this offer. You can reach me directly at 833-622-3359. I will be looking forward to your call.',
        'file_name': 'test_voicemail.m4a',
        'processed_at': datetime.now(),
        'risk_score': 7,
        'indicators': ['CALLBACK PRESSURE', 'REWARD/URGENCY'],
        'explanation': 'The voicemail creates a sense of urgency by presenting an "exclusive opportunity" and encourages immediate action by asking the recipient to call back.',
        'caller_id_mismatch': False,
        'transcript_numbers': ['833-622-3359'],
        # NEW: Audio analysis data
        'has_audio_analysis': True,
        'audio_analysis': MockAudioAnalysis(),
        'overall_risk_score': 9  # Combined score
    }
    
    # Generate report (don't send, just preview)
    report = generate_analysis_report(test_voicemail_data)
    
    # Save to file for preview
    with open('sample_report_with_community.html', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Sample report with community sharing generated: sample_report_with_community.html")
    print("🔍 Open this file in your browser to preview the enhanced report")
    
    # Uncomment to test actual email sending:
    # return send_analysis_report("your-test-email@example.com", test_voicemail_data)

if __name__ == "__main__":
    test_email_system()