import openai
import os
import json
import logging
import sqlite3
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import hashlib

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API configuration
openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class PhishingAnalysis:
    """Data class for phishing analysis results"""
    transcript: str
    risk_score: int  # 1-10 scale
    indicators: List[str]
    explanation: str
    caller_id_mismatch: bool  # New field for phone number mismatch
    transcript_numbers: List[str]  # Phone numbers found in transcript

class TextAnalyzer:
    """
    Text-based phishing detection using OpenAI GPT-4o-mini
    """
    
    def __init__(self, db_path: str = "voicemail_analysis.db"):
        """
        Initialize the text analyzer
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create voicemail_submissions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS voicemail_submissions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        phone_number TEXT NOT NULL,
                        transcript TEXT NOT NULL,
                        transcript_hash TEXT UNIQUE NOT NULL,
                        audio_file_path TEXT,
                        submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        text_risk_score INTEGER,
                        text_analysis TEXT,
                        text_indicators TEXT,
                        caller_id_mismatch BOOLEAN DEFAULT FALSE,
                        transcript_numbers TEXT,
                        audio_risk_score INTEGER DEFAULT NULL,
                        overall_risk_score INTEGER DEFAULT NULL,
                        analysis_date TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("✅ Database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """
        Extract phone numbers from text using regex patterns
        
        Args:
            text: Text to search for phone numbers
            
        Returns:
            List of phone numbers found in various formats
        """
        # Common phone number patterns
        patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',           # 123-456-7890
            r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',     # (123) 456-7890
            r'\b\(\d{3}\)\s*\d{3}\s*\d{4}\b',   # (123) 456 7890
            r'\b\d{3}\s*\d{3}\s*\d{4}\b',       # 123 456 7890
            r'\b\d{3}\.\d{3}\.\d{4}\b',         # 123.456.7890
            r'\b\d{10}\b',                      # 1234567890
            r'\b1[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',  # 1-123-456-7890 or 1 123 456 7890
        ]
        
        found_numbers = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Clean the number (remove formatting)
                clean_number = re.sub(r'[^\d]', '', match)
                # Remove leading 1 if present (country code)
                if len(clean_number) == 11 and clean_number.startswith('1'):
                    clean_number = clean_number[1:]
                # Only keep 10-digit numbers
                if len(clean_number) == 10:
                    found_numbers.append(clean_number)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_numbers))
    
    def check_caller_id_mismatch(self, submitted_number: str, transcript: str) -> Tuple[bool, List[str]]:
        """
        Check if the submitted caller ID matches phone numbers mentioned in transcript
        
        Args:
            submitted_number: Phone number from caller ID
            transcript: Voicemail transcript text
            
        Returns:
            Tuple of (is_mismatch, list_of_transcript_numbers)
        """
        # Clean the submitted number
        clean_submitted = re.sub(r'[^\d]', '', submitted_number)
        if len(clean_submitted) == 11 and clean_submitted.startswith('1'):
            clean_submitted = clean_submitted[1:]
        
        # Extract numbers from transcript
        transcript_numbers = self.extract_phone_numbers(transcript)
        
        logger.info(f"📞 Caller ID: {clean_submitted}")
        logger.info(f"📝 Numbers in transcript: {transcript_numbers}")
        
        # If no numbers in transcript, no mismatch can be determined
        if not transcript_numbers:
            return False, transcript_numbers
        
        # Check if submitted number matches any number in transcript
        is_mismatch = clean_submitted not in transcript_numbers
        
        if is_mismatch:
            logger.warning(f"⚠️ Caller ID mismatch detected! Caller: {clean_submitted}, Transcript: {transcript_numbers}")
        else:
            logger.info(f"✅ Caller ID matches transcript number")
        
        return is_mismatch, transcript_numbers

    def _create_analysis_prompt(self, transcript: str, phone_mismatch_info: str = "") -> str:
        """
        Create a detailed prompt for GPT-4o-mini to analyze the transcript
        
        Args:
            transcript: The voicemail transcript to analyze
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are an expert cybersecurity analyst specializing in voice-based phishing (vishing) detection. Analyze the following voicemail transcript for potential phishing indicators and scam tactics.

TRANSCRIPT TO ANALYZE:
"{transcript}"

{phone_mismatch_info}

Please provide a comprehensive analysis in the following JSON format:

{{
    "risk_score": <integer 1-10 where 10 is highest risk>,
    "indicators": [
        "<specific phishing indicator found>",
        "<another indicator>"
    ],
    "explanation": "<detailed 2-3 sentence explanation of why this is/isn't likely phishing>"
}}

ANALYSIS CRITERIA:
1. URGENCY TACTICS: Phrases like "immediate action required", "within 24 hours", "final notice"
2. AUTHORITY IMPERSONATION: Claims to be from banks, IRS, government agencies, tech support
3. THREAT LANGUAGE: Mentions of account suspension, legal action, arrest, fines
4. INFORMATION REQUESTS: Asking for personal data, passwords, account numbers, SSN
5. CALLBACK PRESSURE: Demanding immediate callback, providing suspicious phone numbers
6. FEAR TACTICS: Creating panic about security breaches, suspicious activity
7. REWARD/URGENCY: Offering prizes, refunds, or benefits with time pressure
8. GENERIC GREETINGS: Lack of personalization, no specific account details
9. GRAMMAR/LANGUAGE: Poor grammar, unusual phrasing, robotic language
10. VERIFICATION REQUESTS: Asking to "verify" information they should already have
11. CALLER ID SPOOFING: Mismatched phone numbers between caller ID and callback numbers

LEGITIMATE INDICATORS:
- Specific account references, appointment confirmations
- Professional language without urgency
- Clear identification with verifiable contact info
- No requests for sensitive information
- Reasonable timeframes for responses
- Caller ID matches callback numbers mentioned

Respond ONLY with the JSON object, no additional text.
"""
        return prompt
    
    def analyze_transcript(self, transcript: str, submitted_phone: Optional[str] = None) -> PhishingAnalysis:
        """
        Analyze a voicemail transcript for phishing indicators
        
        Args:
            transcript: The transcript text to analyze
            submitted_phone: The phone number from caller ID (optional)
            
        Returns:
            PhishingAnalysis object with results
        """
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        
        logger.info(f"🔍 Analyzing transcript: {transcript[:100]}...")
        
        # Check for caller ID mismatch if phone number provided
        caller_id_mismatch = False
        transcript_numbers = []
        phone_mismatch_info = ""
        
        if submitted_phone:
            caller_id_mismatch, transcript_numbers = self.check_caller_id_mismatch(submitted_phone, transcript)
            if caller_id_mismatch:
                phone_mismatch_info = f"\nIMPORTANT: CALLER ID MISMATCH DETECTED - The caller ID ({submitted_phone}) does not match any phone numbers mentioned in the voicemail transcript ({', '.join(transcript_numbers)}). This is a major red flag for phone number spoofing."
        
        try:
            # Create the analysis prompt with phone mismatch info
            prompt = self._create_analysis_prompt(transcript, phone_mismatch_info)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert analyzing voicemail transcripts for phishing indicators. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1000
            )
            
            # Parse the JSON response
            analysis_text = response.choices[0].message.content.strip()
            logger.info(f"📋 Raw AI response: {analysis_text}")
            
            # Clean up response (remove code blocks if present)
            if analysis_text.startswith("```json"):
                analysis_text = analysis_text[7:]
            if analysis_text.endswith("```"):
                analysis_text = analysis_text[:-3]
            
            analysis_data = json.loads(analysis_text)
            
            # Get indicators from AI response
            indicators = analysis_data.get('indicators', [])
            risk_score = analysis_data.get('risk_score', 5)
            
            # Add caller ID mismatch to indicators if detected (avoid duplicates)
            if caller_id_mismatch:
                # Check if any form of caller ID spoofing is already detected
                spoofing_indicators = [ind for ind in indicators if 'CALLER' in ind.upper() or 'SPOOFING' in ind.upper()]
                if not spoofing_indicators:
                    indicators.append("CALLER ID SPOOFING")
                # Increase risk score for caller ID mismatch
                risk_score = min(10, risk_score + 2)
                logger.info(f"🚨 Added CALLER ID SPOOFING indicator, risk score increased to {risk_score}")
            
            # Remove any duplicate indicators while preserving order
            unique_indicators = []
            for indicator in indicators:
                if indicator not in unique_indicators:
                    unique_indicators.append(indicator)
            
            logger.info(f"🔍 Final indicators: {unique_indicators}")
            
            # Create PhishingAnalysis object
            analysis = PhishingAnalysis(
                transcript=transcript,
                risk_score=risk_score,
                indicators=unique_indicators,
                explanation=analysis_data.get('explanation', 'Analysis completed'),
                caller_id_mismatch=caller_id_mismatch,
                transcript_numbers=transcript_numbers
            )
            
            logger.info(f"✅ Analysis complete - Risk Score: {analysis.risk_score}/10")
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse AI response: {e}")
            # Return a fallback analysis
            return PhishingAnalysis(
                transcript=transcript,
                risk_score=5,
                indicators=["Analysis parsing failed"],
                explanation="Unable to complete automated analysis",
                caller_id_mismatch=False,
                transcript_numbers=[]
            )
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise
    
    def save_analysis(self, phone_number: str, analysis: PhishingAnalysis, 
                     audio_file_path: Optional[str] = None) -> int:
        """
        Save the analysis results to the database
        
        Args:
            phone_number: Phone number of the caller
            analysis: PhishingAnalysis object
            audio_file_path: Optional path to the audio file
            
        Returns:
            ID of the saved record
        """
        try:
            # Create a hash of the transcript for deduplication
            transcript_hash = hashlib.sha256(analysis.transcript.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if this transcript already exists
                cursor.execute('SELECT id FROM voicemail_submissions WHERE transcript_hash = ?', 
                             (transcript_hash,))
                existing = cursor.fetchone()
                
                if existing:
                    logger.info(f"📋 Transcript already analyzed (ID: {existing[0]})")
                    return existing[0]
                
                # Insert new analysis
                cursor.execute('''
                    INSERT INTO voicemail_submissions (
                        phone_number, transcript, transcript_hash, audio_file_path,
                        text_risk_score, text_analysis, text_indicators, 
                        caller_id_mismatch, transcript_numbers, analysis_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    phone_number,
                    analysis.transcript,
                    transcript_hash,
                    audio_file_path,
                    analysis.risk_score,
                    analysis.explanation,
                    json.dumps(analysis.indicators),
                    analysis.caller_id_mismatch,
                    json.dumps(analysis.transcript_numbers),
                    datetime.now()
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"✅ Analysis saved to database (ID: {record_id})")
                return record_id
                
        except Exception as e:
            logger.error(f"❌ Failed to save analysis: {e}")
            raise
    
    def get_all_submissions(self, limit: int = 50) -> List[Dict]:
        """
        Get all submissions for display
        
        Args:
            limit: Maximum number of submissions to return
            
        Returns:
            List of submission dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, phone_number, transcript, text_risk_score,
                           text_analysis, text_indicators, caller_id_mismatch,
                           transcript_numbers, audio_risk_score, 
                           overall_risk_score, submission_date
                    FROM voicemail_submissions 
                    ORDER BY submission_date DESC 
                    LIMIT ?
                ''', (limit,))
                
                submissions = []
                for row in cursor.fetchall():
                    submissions.append({
                        'id': row[0],
                        'phone_number': row[1],
                        'transcript': row[2],
                        'text_risk_score': row[3],
                        'text_analysis': row[4],
                        'text_indicators': json.loads(row[5]) if row[5] else [],
                        'caller_id_mismatch': bool(row[6]),
                        'transcript_numbers': json.loads(row[7]) if row[7] else [],
                        'audio_risk_score': row[8],
                        'overall_risk_score': row[9],
                        'submission_date': row[10]
                    })
                
                logger.info(f"📋 Retrieved {len(submissions)} submissions")
                return submissions
                
        except Exception as e:
            logger.error(f"❌ Failed to get submissions: {e}")
            return []


def analyze_voicemail_text(transcript: str, phone_number: str, 
                          audio_file_path: Optional[str] = None) -> Tuple[int, PhishingAnalysis]:
    """
    Convenience function to analyze voicemail text and save to database
    
    Args:
        transcript: The voicemail transcript
        phone_number: Caller's phone number
        audio_file_path: Optional path to audio file
        
    Returns:
        Tuple of (database_id, analysis_object)
    """
    analyzer = TextAnalyzer()
    analysis = analyzer.analyze_transcript(transcript, phone_number)
    db_id = analyzer.save_analysis(phone_number, analysis, audio_file_path)
    return db_id, analysis


if __name__ == "__main__":
    # Test the analyzer
    test_transcript = """
    Hello, this is an urgent message regarding your bank account. 
    Suspicious activity has been detected, and immediate action is required. 
    Please call us back at this number to verify your information and avoid 
    permanent account suspension. Thank you.
    """
    
    try:
        analyzer = TextAnalyzer()
        analysis = analyzer.analyze_transcript(test_transcript, "5551234567")
        
        print("="*50)
        print("PHISHING ANALYSIS RESULTS")
        print("="*50)
        print(f"Risk Score: {analysis.risk_score}/10")
        print(f"\nIndicators: {', '.join(analysis.indicators)}")
        print(f"\nExplanation: {analysis.explanation}")
        print(f"\nCaller ID Mismatch: {analysis.caller_id_mismatch}")
        print(f"Numbers in Transcript: {analysis.transcript_numbers}")
        
        # Save to database
        db_id = analyzer.save_analysis("5551234567", analysis)
        print(f"\n✅ Saved to database with ID: {db_id}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")