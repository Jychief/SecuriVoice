import torch
import numpy as np
import logging
import os
import sqlite3
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import warnings
import librosa
import soundfile as sf

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging - QUIETER for production
logging.basicConfig(level=logging.WARNING)  # Changed from INFO to WARNING
logger = logging.getLogger(__name__)

@dataclass
class AudioAnalysis:
    """Data class for audio analysis results"""
    audio_file_path: str
    is_ai_generated: bool
    confidence_score: float  # 0-1 scale
    risk_score: int  # 1-10 scale
    indicators: list
    explanation: str
    audio_quality_metrics: Dict

class AudioAnalyzer:
    """
    Audio-based phishing detection using VoiceGUARD (Wav2Vec2-based classifier)
    to detect AI-generated voices commonly used in vishing attacks
    """
    
    def __init__(self, model_name: str = "Mrkomiljon/voiceGUARD", db_path: str = "voicemail_analysis.db"):
        """
        Initialize the audio analyzer
        
        Args:
            model_name: HuggingFace model name for voice detection
            db_path: Path to SQLite database file
        """
        self.model_name = model_name
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.feature_extractor = None
        
        # Only log initialization once per session
        if not hasattr(AudioAnalyzer, '_initialized'):
            # QUIETER - no device logging unless error
            AudioAnalyzer._initialized = True
        
        self._load_model()
    
    def _load_model(self):
        """Load the VoiceGUARD model and feature extractor"""
        try:
            # Only log loading if not already cached
            if not hasattr(AudioAnalyzer, '_model_loaded'):
                # QUIETER - minimal loading message
                AudioAnalyzer._model_loaded = True
            
            # Load the feature extractor and model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Only log success once
            if AudioAnalyzer._model_loaded:
                # QUIETER - no success message
                AudioAnalyzer._model_loaded = False  # Prevent further logging
            
        except Exception as e:
            logger.error(f"❌ Failed to load VoiceGUARD model: {e}")
            raise
    
    def _preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio file for model input using librosa (Railway-compatible)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Load audio file using librosa (supports many formats without FFmpeg)
            # librosa automatically handles format conversion
            target_sample_rate = 16000
            
            # Load and resample in one step
            waveform, sample_rate = librosa.load(
                audio_path, 
                sr=target_sample_rate,  # Automatically resample to 16kHz
                mono=True  # Convert to mono
            )
            
            # Ensure we have a 1D array
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=0)
            
            # QUIETER - no preprocessing logs
            return waveform, target_sample_rate
            
        except Exception as e:
            logger.error(f"❌ Audio preprocessing failed: {e}")
            raise
    
    def _calculate_audio_quality_metrics(self, waveform: np.ndarray, sample_rate: int) -> Dict:
        """
        Calculate basic audio quality metrics using librosa (Railway-compatible)
        
        Args:
            waveform: Audio waveform numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of audio quality metrics
        """
        try:
            # Calculate basic metrics
            duration = len(waveform) / sample_rate
            rms_energy = np.sqrt(np.mean(waveform ** 2))
            zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(waveform)))) / 2
            
            # Calculate spectral features using librosa
            # Spectral centroid (frequency center of mass)
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
            spectral_centroid_mean = np.mean(spectral_centroid)
            
            # Dynamic range
            dynamic_range = float(np.max(waveform) - np.min(waveform))
            
            # Additional librosa features for better analysis
            try:
                # Spectral rolloff
                spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)
                spectral_rolloff_mean = np.mean(spectral_rolloff)
                
                # MFCC features (first coefficient represents energy)
                mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=1)
                mfcc_energy = np.mean(mfccs[0])
                
            except Exception:
                spectral_rolloff_mean = 0.0
                mfcc_energy = 0.0
            
            metrics = {
                'duration_seconds': float(duration),
                'rms_energy': float(rms_energy),
                'zero_crossing_rate': float(zero_crossing_rate),
                'spectral_centroid_mean': float(spectral_centroid_mean),
                'spectral_rolloff_mean': float(spectral_rolloff_mean),
                'mfcc_energy': float(mfcc_energy),
                'dynamic_range': dynamic_range,
                'sample_rate': int(sample_rate),
                'audio_length': int(len(waveform))
            }
            
            # QUIETER - no metrics logging
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Audio metrics calculation failed: {e}")
            return {
                'duration_seconds': 0.0,
                'rms_energy': 0.0,
                'zero_crossing_rate': 0.0,
                'spectral_centroid_mean': 0.0,
                'dynamic_range': 0.0,
                'sample_rate': 16000,
                'audio_length': 0
            }
    
    def analyze_audio(self, audio_path: str) -> AudioAnalysis:
        """
        Analyze audio file for AI-generated voice detection
        
        Args:
            audio_path: Path to audio file (.m4a, .wav, .mp3, etc.)
            
        Returns:
            AudioAnalysis object with results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # QUIETER - no analysis start message
        
        try:
            # Preprocess audio
            waveform, sample_rate = self._preprocess_audio(audio_path)
            
            # Calculate audio quality metrics
            audio_metrics = self._calculate_audio_quality_metrics(waveform, sample_rate)
            
            # QUIETER - no debug logging of audio characteristics
            
            # Prepare input for model using numpy array
            inputs = self.feature_extractor(
                waveform,  # waveform is now numpy array
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get prediction
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
                
                # Multi-class interpretation for VoiceGUARD
                if predicted_class >= 3:  # Classes 3, 4, 5, 6 likely indicate AI/synthetic
                    is_ai_generated = True
                else:  # Classes 0, 1, 2 likely indicate human
                    is_ai_generated = False
                
                # Adjust confidence based on class prediction
                if predicted_class >= 4 and confidence > 0.9:
                    confidence = min(confidence, 0.95)
                elif predicted_class == 3 and confidence > 0.8:
                    confidence = min(confidence, 0.9)
                elif predicted_class <= 2:
                    confidence = min(confidence, 0.98)
            
            # Heuristic checks for AI characteristics
            heuristic_ai_score = self._calculate_heuristic_ai_score(audio_metrics, waveform)
            
            # Enhanced decision logic: combine model prediction with heuristics
            original_prediction = is_ai_generated
            
            if not is_ai_generated and heuristic_ai_score >= 3:  # Model says human but heuristics suggest AI
                if confidence < 0.9:  # Not very confident in human prediction
                    is_ai_generated = True
                    confidence = 0.6 + (heuristic_ai_score * 0.1)  # Moderate confidence based on heuristics
                    # QUIETER - no override logging
            
            elif is_ai_generated and heuristic_ai_score >= 2:  # Model says AI and heuristics agree
                confidence = min(confidence + (heuristic_ai_score * 0.05), 0.98)  # Boost confidence
            
            elif is_ai_generated and heuristic_ai_score == 0:  # Model says AI but no heuristic evidence
                if confidence < 0.7:  # Low confidence AI prediction
                    confidence *= 0.8  # Reduce confidence slightly
            
            # QUIETER - no VoiceGUARD result logging
            
            # Calculate risk score and indicators
            risk_score, indicators, explanation = self._calculate_audio_risk(
                is_ai_generated, confidence, audio_metrics, heuristic_ai_score
            )
            
            # Create analysis result
            analysis = AudioAnalysis(
                audio_file_path=audio_path,
                is_ai_generated=is_ai_generated,
                confidence_score=confidence,
                risk_score=risk_score,
                indicators=indicators,
                explanation=explanation,
                audio_quality_metrics=audio_metrics
            )
            
            # QUIETER - no completion message
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Audio analysis failed: {e}")
            # Return fallback analysis
            return AudioAnalysis(
                audio_file_path=audio_path,
                is_ai_generated=False,
                confidence_score=0.5,
                risk_score=5,
                indicators=["Analysis failed"],
                explanation="Unable to complete audio analysis",
                audio_quality_metrics={}
            )
    
    def _calculate_heuristic_ai_score(self, audio_metrics: Dict, waveform: np.ndarray) -> int:
        """
        Calculate heuristic AI score based on audio characteristics
        
        Args:
            audio_metrics: Audio quality metrics
            waveform: Raw audio waveform
            
        Returns:
            Heuristic AI score (0-5, higher = more likely AI)
        """
        score = 0
        
        try:
            duration = audio_metrics.get('duration_seconds', 0)
            rms_energy = audio_metrics.get('rms_energy', 0)
            zero_crossing_rate = audio_metrics.get('zero_crossing_rate', 0)
            spectral_centroid = audio_metrics.get('spectral_centroid_mean', 0)
            dynamic_range = audio_metrics.get('dynamic_range', 0)
            
            # 1. Unnatural silence/background noise (common in TTS)
            if rms_energy < 0.005:  # Very quiet background
                score += 1
                # QUIETER - no heuristic logging
            
            # 2. Unnatural zero crossing rate (robotic speech patterns)
            if zero_crossing_rate > 0.25 or zero_crossing_rate < 0.05:
                score += 1
            
            # 3. Limited dynamic range (compressed/processed audio)
            if dynamic_range < 0.2:
                score += 1
            
            # 4. Spectral characteristics (unnatural frequency distribution)
            if spectral_centroid > 3000 or spectral_centroid < 500:
                score += 1
            
            # 5. Advanced analysis: Check for periodic patterns (TTS artifacts)
            try:
                # Calculate autocorrelation to detect unnatural periodicity
                autocorr = np.correlate(waveform, waveform, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Look for strong periodic patterns (excluding fundamental frequency)
                if len(autocorr) > 1000:
                    # Check for artificial periodicity in mid-range
                    mid_range = autocorr[200:800]
                    if len(mid_range) > 0:
                        periodicity_strength = np.max(mid_range) / np.mean(mid_range)
                        if periodicity_strength > 3.0:  # Strong artificial periodicity
                            score += 1
                            
            except Exception as e:
                pass  # Silent failure for periodicity analysis
            
            # QUIETER - no total heuristic score logging
            return score
            
        except Exception as e:
            logger.error(f"❌ Heuristic analysis failed: {e}")
            return 0
    
    def _calculate_audio_risk(self, is_ai_generated: bool, confidence: float, 
                            audio_metrics: Dict, heuristic_score: int = 0) -> Tuple[int, list, str]:
        """
        Calculate risk score based on audio analysis results
        
        Args:
            is_ai_generated: Whether audio is AI-generated
            confidence: Model confidence score
            audio_metrics: Audio quality metrics
            heuristic_score: Heuristic AI detection score
            
        Returns:
            Tuple of (risk_score, indicators, explanation)
        """
        indicators = []
        risk_score = 1
        
        # QUIETER - no risk calculation logging
        
        # AI-generated voice detection (ENHANCED SCORING)
        if is_ai_generated:
            if confidence > 0.85:  # Very high confidence (85%+)
                risk_score += 7  # Major risk increase (was +6)
                indicators.append("HIGH CONFIDENCE AI-GENERATED VOICE")
            elif confidence > 0.7:  # High confidence (70-85%)
                risk_score += 6  # Increased from +5
                indicators.append("LIKELY AI-GENERATED VOICE")
            elif confidence > 0.6:  # Moderate confidence (60-70%)
                risk_score += 5  # Increased from +4
                indicators.append("POSSIBLE AI-GENERATED VOICE")
            else:  # Lower confidence (50-60%)
                risk_score += 4  # Increased from +3
                indicators.append("SUSPECTED AI-GENERATED VOICE")
            
            # Extra penalty for extremely high confidence
            if confidence > 0.95:
                risk_score += 1
                indicators.append("EXTREMELY HIGH AI CONFIDENCE")
        
        # Add risk based on heuristic analysis
        if heuristic_score >= 4:
            risk_score += 2
            indicators.append("STRONG AI VOICE CHARACTERISTICS")
        elif heuristic_score >= 3:
            risk_score += 1
            indicators.append("MULTIPLE AI VOICE INDICATORS")
        elif heuristic_score >= 2:
            indicators.append("SOME AI VOICE INDICATORS")
        
        # Audio quality indicators that might suggest synthetic audio
        duration = audio_metrics.get('duration_seconds', 0)
        rms_energy = audio_metrics.get('rms_energy', 0)
        zero_crossing_rate = audio_metrics.get('zero_crossing_rate', 0)
        dynamic_range = audio_metrics.get('dynamic_range', 0)
        
        # Additional risk factors (but lower weight if already AI-detected)
        additional_risk = 0
        
        # Very short duration (robocalls are often brief)
        if duration < 15:
            additional_risk += 1
            indicators.append("VERY SHORT DURATION")
        
        # Unusual audio characteristics
        if rms_energy < 0.01:  # Very low energy might indicate processed audio
            additional_risk += 1
            indicators.append("LOW AUDIO ENERGY")
        
        if zero_crossing_rate > 0.3:  # High ZCR might indicate synthetic speech
            additional_risk += 1
            indicators.append("HIGH ZERO CROSSING RATE")
        
        if dynamic_range < 0.1:  # Low dynamic range suggests compressed/processed audio
            additional_risk += 1
            indicators.append("LOW DYNAMIC RANGE")
        
        # Background noise analysis
        if rms_energy > 0 and rms_energy < 0.005:
            additional_risk += 1
            indicators.append("UNNATURAL SILENCE/NO BACKGROUND NOISE")
        
        # If AI is detected, reduce weight of additional factors (they're less important)
        if is_ai_generated:
            additional_risk = min(additional_risk, 2)  # Cap additional risk when AI detected
        
        risk_score += additional_risk
        
        # Cap risk score at 10
        risk_score = min(risk_score, 10)
        
        # QUIETER - no final audio risk logging
        
        # Generate explanation
        if is_ai_generated:
            explanation = f"VoiceGUARD detected this as an AI-generated voice with {confidence:.1%} confidence. "
            if confidence > 0.9:
                explanation += "This extremely high confidence strongly indicates the use of text-to-speech or voice cloning technology. "
            elif confidence > 0.8:
                explanation += "This high confidence strongly suggests synthetic voice generation. "
            else:
                explanation += "This indicates likely synthetic voice generation. "
                
            if heuristic_score >= 3:
                explanation += f"Additionally, heuristic analysis found {heuristic_score} AI voice characteristics. "
            
            explanation += "AI-generated voices are commonly used in vishing attacks to impersonate legitimate organizations and bypass voice recognition systems."
            
            if risk_score >= 8:
                explanation += " This represents a very high risk for voice-based fraud."
            elif risk_score >= 6:
                explanation += " This represents a high risk for voice-based fraud."
                
        else:
            explanation = f"VoiceGUARD classified this as a human voice with {confidence:.1%} confidence. "
            if heuristic_score >= 3:
                explanation += f"However, heuristic analysis found {heuristic_score} characteristics often associated with AI-generated voices, which raises some concerns. "
            explanation += "Other audio characteristics are also analyzed for potential signs of manipulation or robocall patterns."
            
            if any("DURATION" in ind or "ENERGY" in ind or "RANGE" in ind for ind in indicators):
                explanation += " Some audio quality metrics suggest this could still be from an automated system."
        
        return risk_score, indicators, explanation
    
    def save_audio_analysis(self, analysis: AudioAnalysis, db_record_id: int) -> bool:
        """
        Save audio analysis results to the database
        
        Args:
            analysis: AudioAnalysis object
            db_record_id: ID of existing record to update
            
        Returns:
            True if saved successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update the existing record with audio analysis results
                cursor.execute('''
                    UPDATE voicemail_submissions 
                    SET audio_risk_score = ?,
                        audio_analysis = ?,
                        audio_indicators = ?,
                        audio_confidence = ?,
                        is_ai_generated = ?,
                        audio_metrics = ?,
                        overall_risk_score = ?,
                        audio_analysis_date = ?
                    WHERE id = ?
                ''', (
                    analysis.risk_score,
                    analysis.explanation,
                    str(analysis.indicators),  # Convert list to string
                    analysis.confidence_score,
                    analysis.is_ai_generated,
                    str(analysis.audio_quality_metrics),  # Convert dict to string
                    None,  # Will calculate overall risk score separately
                    datetime.now(),
                    db_record_id
                ))
                
                # Calculate and update overall risk score
                cursor.execute('SELECT text_risk_score FROM voicemail_submissions WHERE id = ?', (db_record_id,))
                result = cursor.fetchone()
                
                if result:
                    text_risk_score = result[0] or 0
                    # Overall risk is weighted average: 60% text, 40% audio
                    overall_risk = int((text_risk_score * 0.6) + (analysis.risk_score * 0.4))
                    overall_risk = min(overall_risk, 10)  # Cap at 10
                    
                    cursor.execute('UPDATE voicemail_submissions SET overall_risk_score = ? WHERE id = ?', 
                                 (overall_risk, db_record_id))
                
                conn.commit()
                # QUIETER - no save success logging
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save audio analysis: {e}")
            return False
    
    def update_database_schema(self):
        """Update database schema to include audio analysis fields"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check which columns already exist
                cursor.execute("PRAGMA table_info(voicemail_submissions)")
                existing_columns = {row[1] for row in cursor.fetchall()}
                
                # Add new columns for audio analysis
                new_columns = [
                    ("audio_risk_score", "INTEGER DEFAULT NULL"),
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
                        except sqlite3.OperationalError as e:
                            if "duplicate column name" not in str(e):
                                raise
                
                conn.commit()
                
                # Only log if we actually added columns
                if columns_added > 0:
                    # QUIETER - no schema update logging
                    pass
                
        except Exception as e:
            logger.error(f"❌ Failed to update database schema: {e}")
            raise


def analyze_voicemail_audio(audio_file_path: str, db_record_id: int) -> Tuple[int, AudioAnalysis]:
    """
    Convenience function to analyze voicemail audio and save to database
    
    Args:
        audio_file_path: Path to audio file
        db_record_id: Database record ID to update
        
    Returns:
        Tuple of (updated_overall_risk_score, analysis_object)
    """
    try:
        # Initialize analyzer and update schema
        analyzer = AudioAnalyzer()
        analyzer.update_database_schema()
        
        # Analyze audio
        analysis = analyzer.analyze_audio(audio_file_path)
        
        # Save results
        analyzer.save_audio_analysis(analysis, db_record_id)
        
        # Get updated overall risk score
        with sqlite3.connect(analyzer.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT overall_risk_score FROM voicemail_submissions WHERE id = ?', (db_record_id,))
            result = cursor.fetchone()
            overall_risk = result[0] if result else analysis.risk_score
        
        return overall_risk, analysis
        
    except Exception as e:
        logger.error(f"❌ Audio analysis failed: {e}")
        # Return fallback
        fallback_analysis = AudioAnalysis(
            audio_file_path=audio_file_path,
            is_ai_generated=False,
            confidence_score=0.5,
            risk_score=5,
            indicators=["Analysis failed"],
            explanation="Audio analysis could not be completed",
            audio_quality_metrics={}
        )
        return 5, fallback_analysis


if __name__ == "__main__":
    # Test the audio analyzer
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
        if os.path.exists(test_file):
            try:
                analyzer = AudioAnalyzer()
                analysis = analyzer.analyze_audio(test_file)
                
                print("="*50)
                print("AUDIO ANALYSIS RESULTS")
                print("="*50)
                print(f"File: {analysis.audio_file_path}")
                print(f"AI Generated: {analysis.is_ai_generated}")
                print(f"Confidence: {analysis.confidence_score:.3f}")
                print(f"Risk Score: {analysis.risk_score}/10")
                print(f"Indicators: {', '.join(analysis.indicators)}")
                print(f"Explanation: {analysis.explanation}")
                print(f"Audio Metrics: {analysis.audio_quality_metrics}")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
        else:
            print(f"❌ Audio file '{test_file}' not found")
            print("Please provide a valid audio file path")
    else:
        print("🎵 SecuriVoice Audio Analysis")
        print("="*30)
        print("Usage: python audio_analysis.py <audio_file>")
        print("")
        print("Examples:")
        print("  python audio_analysis.py my_voicemail.m4a")
        print("  python audio_analysis.py test_audio.wav")
        print("  python audio_analysis.py sample.mp3")
        print("")
        print("Supported formats: .m4a, .wav, .mp3, .flac, .ogg")
        print("")
        print("This will analyze the audio file for AI-generated voice detection")
        print("and display detailed results including risk scoring.")