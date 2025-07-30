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
                        analysis_date TIMESTAMP,
                        sender_email TEXT DEFAULT NULL
                    )
                ''')
                
                # Check which columns already exist
                cursor.execute("PRAGMA table_info(voicemail_submissions)")
                existing_columns = {row[1] for row in cursor.fetchall()}
                
                # Add missing columns for community sharing and sender tracking
                new_columns = [
                    ("sender_email", "TEXT DEFAULT NULL"),
                    ("community_permission", "BOOLEAN DEFAULT FALSE"),
                    ("permission_granted_at", "TIMESTAMP DEFAULT NULL"),
                    ("permission_email_uid", "TEXT DEFAULT NULL"),
                    ("shared_to_community", "BOOLEAN DEFAULT FALSE"),
                    ("shared_at", "TIMESTAMP DEFAULT NULL"),
                    ("audio_risk_score", "INTEGER DEFAULT NULL"),
                    ("overall_risk_score", "INTEGER DEFAULT NULL"),
                    ("audio_analysis", "TEXT DEFAULT NULL"),
                    ("audio_indicators", "TEXT DEFAULT NULL"), 
                    ("audio_confidence", "REAL DEFAULT NULL"),
                    ("is_ai_generated", "BOOLEAN DEFAULT NULL"),
                    ("audio_metrics", "TEXT DEFAULT NULL"),
                    ("audio_analysis_date", "TIMESTAMP DEFAULT NULL")
                ]
                
                columns_added = 0
                for column_name, column_def in new_columns:
                    if column_name not in existing_columns:
                        try:
                            cursor.execute(f'ALTER TABLE voicemail_submissions ADD COLUMN {column_name} {column_def}')
                            columns_added += 1
                            logger.info(f"✅ Added column: {column_name}")
                        except sqlite3.OperationalError as e:
                            if "duplicate column name" not in str(e):
                                raise
                
                conn.commit()
                
                if columns_added > 0:
                    logger.info(f"✅ Database schema updated - added {columns_added} columns")
                else:
                    logger.info("✅ Database schema is up to date")
                
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
            # NEW: Handle spoken numbers with dashes between each digit
            r'\b1-\d-\d-\d-\d{7}\b',            # 1-8-0-0-5550199 (partial)
            r'\b1-\d-\d-\d-\d-\d-\d-\d-\d-\d-\d\b',  # 1-8-0-0-5-5-5-0-1-9-9
            r'\b\d-\d-\d-\d-\d{6}\b',           # 8-0-0-555199 (8 digit start)
            r'\b\d-\d-\d-\d-\d-\d-\d-\d-\d-\d\b',    # 8-0-0-5-5-5-0-1-9-9
        ]
        
        found_numbers = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Clean the number (remove formatting)
                clean_number = re.sub(r'[^\d]', '', match)
                
                # Handle different length numbers
                if len(clean_number) == 11 and clean_number.startswith('1'):
                    # Remove leading 1 (country code)
                    clean_number = clean_number[1:]
                elif len(clean_number) == 10:
                    # Perfect 10-digit number
                    pass
                elif len(clean_number) > 10:
                    # Try to extract a 10-digit number from longer sequences
                    # Look for patterns like area code + 7 digits
                    if len(clean_number) >= 10:
                        # Take the last 10 digits (most common case)
                        clean_number = clean_number[-10:]
                else:
                    # Too short, skip
                    continue
                
                # Only keep 10-digit numbers
                if len(clean_number) == 10 and clean_number.isdigit():
                    found_numbers.append(clean_number)
        
        # Advanced pattern: Look for spoken number sequences in transcripts
        # Handle cases like "call us back at one eight zero zero five five five zero one nine nine"
        spoken_patterns = self._extract_spoken_numbers(text)
        found_numbers.extend(spoken_patterns)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_numbers))
    
    def _extract_spoken_numbers(self, text: str) -> List[str]:
        """
        Extract phone numbers from spoken/written out numbers
        
        Args:
            text: Text containing potentially spoken numbers
            
        Returns:
            List of phone numbers extracted from spoken format
        """
        found_numbers = []
        
        # Word to digit mapping
        word_to_digit = {
            'zero': '0', 'oh': '0', 'o': '0',
            'one': '1', 'won': '1',
            'two': '2', 'to': '2', 'too': '2',
            'three': '3', 'tree': '3',
            'four': '4', 'for': '4', 'fore': '4',
            'five': '5',
            'six': '6', 'six': '6',
            'seven': '7',
            'eight': '8', 'ate': '8',
            'nine': '9'
        }
        
        # Look for sequences like "one eight zero zero five five five..."
        # Split text into words and look for number word sequences
        words = re.findall(r'\b\w+\b', text.lower())
        
        number_sequences = []
        current_sequence = []
        
        for word in words:
            if word in word_to_digit:
                current_sequence.append(word_to_digit[word])
            elif word.isdigit() and len(word) == 1:
                current_sequence.append(word)
            else:
                if len(current_sequence) >= 7:  # At least 7 digits for a phone number
                    number_sequences.append(''.join(current_sequence))
                current_sequence = []
        
        # Check final sequence
        if len(current_sequence) >= 7:
            number_sequences.append(''.join(current_sequence))
        
        # Process found sequences
        for sequence in number_sequences:
            if len(sequence) == 10:
                found_numbers.append(sequence)
            elif len(sequence) == 11 and sequence.startswith('1'):
                found_numbers.append(sequence[1:])  # Remove leading 1
            elif len(sequence) > 10:
                # Try to find a 10-digit subsequence
                for i in range(len(sequence) - 9):
                    substr = sequence[i:i+10]
                    if substr[0] != '0' and substr[0] != '1':  # Valid area code
                        found_numbers.append(substr)
                        break
        
        return found_numbers
    
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
        logger.info(f"📝 Numbers found in transcript: {transcript_numbers}")
        
        # Debug: Show what we're looking for in the raw transcript
        logger.info(f"📄 Transcript excerpt: {transcript[:200]}...")
        
        # If no numbers in transcript, no mismatch can be determined
        if not transcript_numbers:
            logger.info(f"ℹ️ No phone numbers found in transcript")
            return False, transcript_numbers
        
        # Check if submitted number matches any number in transcript
        is_mismatch = clean_submitted not in transcript_numbers
        
        if is_mismatch:
            logger.warning(f"⚠️ Caller ID mismatch detected! Caller: {clean_submitted}, Transcript: {transcript_numbers}")
        else:
            logger.info(f"✅ Caller ID matches transcript number")
        
        return is_mismatch, transcript_numbers

    def _standardize_indicator(self, indicator: str) -> str:
        """
        Standardize indicator formatting to title case
        
        Args:
            indicator: Raw indicator string
            
        Returns:
            Standardized indicator string
        """
        # Convert to title case but preserve certain words in uppercase
        indicator = indicator.strip()
        
        # Special handling for common abbreviations/terms that should stay uppercase
        uppercase_words = ['AI', 'ID', 'IRS', 'FBI', 'CIA', 'SSN', 'PIN', 'ATM', 'URL', 'IP', 'DNS']
        
        # Split by spaces and process each word
        words = indicator.split()
        standardized_words = []
        
        for word in words:
            # Remove any trailing punctuation for comparison
            clean_word = word.rstrip('.,!?:;')
            trailing_punct = word[len(clean_word):]
            
            if clean_word.upper() in uppercase_words:
                standardized_words.append(clean_word.upper() + trailing_punct)
            else:
                standardized_words.append(clean_word.title() + trailing_punct)
        
        return ' '.join(standardized_words)

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

For indicators, use clear, descriptive phrases in title case format, such as:
- "Threat Language"
- "Fear Tactics" 
- "Urgency Tactics"
- "Authority Impersonation"
- "Callback Pressure"
- "Information Requests"
- "Generic Greetings"
- "Caller ID Spoofing"

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
            
            # Get indicators from AI response and standardize them
            raw_indicators = analysis_data.get('indicators', [])
            standardized_indicators = [self._standardize_indicator(ind) for ind in raw_indicators]
            
            risk_score = analysis_data.get('risk_score', 5)
            
            # Add caller ID mismatch to indicators if detected (avoid duplicates)
            if caller_id_mismatch:
                # Check if any form of caller ID spoofing is already detected
                spoofing_indicators = [ind for ind in standardized_indicators if 'caller' in ind.lower() and 'spoofing' in ind.lower()]
                if not spoofing_indicators:
                    standardized_indicators.append("Caller ID Spoofing")
                # Increase risk score for caller ID mismatch
                risk_score = min(10, risk_score + 2)
                logger.info(f"🚨 Added Caller ID Spoofing indicator, risk score increased to {risk_score}")
            
            # Remove any duplicate indicators while preserving order
            unique_indicators = []
            for indicator in standardized_indicators:
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
                indicators=["Analysis Parsing Failed"],
                explanation="Unable to complete automated analysis",
                caller_id_mismatch=False,
                transcript_numbers=[]
            )
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise
    
    def save_analysis(self, phone_number: str, analysis: PhishingAnalysis, 
                     audio_file_path: Optional[str] = None, sender_email: Optional[str] = None) -> int:
        """
        Save the analysis results to the database
        
        Args:
            phone_number: Phone number of the caller
            analysis: PhishingAnalysis object
            audio_file_path: Optional path to the audio file
            sender_email: Email address of the person who submitted this
            
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
                        caller_id_mismatch, transcript_numbers, analysis_date, sender_email
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    datetime.now(),
                    sender_email
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
                          audio_file_path: Optional[str] = None, sender_email: Optional[str] = None) -> Tuple[int, PhishingAnalysis]:
    """
    Convenience function to analyze voicemail text and save to database
    
    Args:
        transcript: The voicemail transcript
        phone_number: Caller's phone number
        audio_file_path: Optional path to audio file
        sender_email: Email address of the submitter
        
    Returns:
        Tuple of (database_id, analysis_object)
    """
    analyzer = TextAnalyzer()
    analysis = analyzer.analyze_transcript(transcript, phone_number)
    db_id = analyzer.save_analysis(phone_number, analysis, audio_file_path, sender_email)
    return db_id, analysis


if __name__ == "__main__":
    # Test the analyzer
    test_transcript = """
    Hello, this is David from the Fraud Prevention Department at your bank. 
    We've detected suspicious activity on your account, ending in 4782. 
    Someone has attempted to make a large purchase from your account just 20 minutes ago. 
    For your security, we've temporarily frozen your account. 
    To verify your identity and restore access, you need to call us back immediately at 1-8-0-0-5550199.
    Please have your full account number, social security number, and the pin you use for online banking ready.
    """
    
    try:
        analyzer = TextAnalyzer()
        analysis = analyzer.analyze_transcript(test_transcript, "7747739012")
        
        print("="*50)
        print("PHISHING ANALYSIS RESULTS")
        print("="*50)
        print(f"Risk Score: {analysis.risk_score}/10")
        print(f"\nIndicators: {', '.join(analysis.indicators)}")
        print(f"\nExplanation: {analysis.explanation}")
        print(f"\nCaller ID Mismatch: {analysis.caller_id_mismatch}")
        print(f"Numbers in Transcript: {analysis.transcript_numbers}")
        
        # Save to database
        db_id = analyzer.save_analysis("7747739012", analysis, sender_email="test@example.com")
        print(f"\n✅ Saved to database with ID: {db_id}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")