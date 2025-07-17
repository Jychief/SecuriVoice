import whisper
import os
import logging
from typing import Optional, Dict, Any
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToText:
    """
    Simple Speech-to-Text processor using OpenAI Whisper ASR model
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the speech-to-text processor
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run on ("cpu", "cuda", or None for auto-detect)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        logger.info(f"Initializing Whisper model '{model_size}' on device '{self.device}'")
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"✅ Whisper model '{self.model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file (.m4a, .wav, .mp3, etc.)
            language: Optional language hint (e.g., "en" for English)
            
        Returns:
            Transcribed text as string
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Validate file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"🎤 Transcribing: {audio_path}")
        
        try:
            # Transcribe with Whisper
            options = {}
            if language:
                options["language"] = language
            
            result = self.model.transcribe(audio_path, **options)
            
            # Return just the text
            transcribed_text = result["text"].strip()
            
            logger.info(f"✅ Transcription complete: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Simple function to transcribe audio file to text
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        
    Returns:
        Transcribed text
    """
    stt = SpeechToText(model_size=model_size)
    return stt.transcribe(audio_path)


if __name__ == "__main__":
    # Test with a sample file
    test_file = "sample_voicemail.m4a"
    
    if os.path.exists(test_file):
        try:
            text = transcribe_audio(test_file)
            print("="*50)
            print("TRANSCRIPTION:")
            print("="*50)
            print(text)
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print(f"Test file '{test_file}' not found")
        print("Please provide a sample audio file to test")