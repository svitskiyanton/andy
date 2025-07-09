#!/usr/bin/env python3
"""
STT Writer Fixed - Simple and Working
Uses the same approach as the working voice changer files
"""

import os
import pyaudio
import time
import numpy as np
from google.cloud import speech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class STTWriterFixed:
    def __init__(self):
        # STT settings - same as working voice changers
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.CHUNK_SIZE = 1024  # Same as working versions
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Google Cloud Speech client
        self.speech_client = None
        self.init_google_client()
        
        # File settings
        self.TEXT_FILE = "voice_changer_text.txt"
        
        # Control
        self.running = True
        
        # Text accumulation
        self.current_phrase = ""
        self.silence_start = None
        self.max_silence_duration = 2.0  # Back to 2 seconds like working versions
        
        # Audio buffer settings - same as working voice changers
        self.BUFFER_DURATION = 3.0  # 3 seconds like working versions
        self.BUFFER_SIZE = int(self.SAMPLE_RATE * self.BUFFER_DURATION)
        self.SILENCE_DURATION = 1.0
        self.SILENCE_THRESHOLD = 0.01
        
        # Audio buffers
        self.audio_buffer = []
        self.silence_buffer = []
    
    def init_google_client(self):
        """Initialize Google Cloud Speech client"""
        try:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path:
                self.speech_client = speech.SpeechClient()
                print("✅ STT: Google Cloud Speech client initialized")
            else:
                print("⚠️  STT: Google Cloud credentials not found")
        except Exception as e:
            print(f"❌ STT: Failed to initialize Google Cloud client: {e}")
    
    def transcribe_audio(self, audio_chunk):
        """Transcribe audio chunk using Google Speech-to-Text - same as working versions"""
        if not self.speech_client:
            return None
        
        try:
            # Create audio content
            audio = speech.RecognitionAudio(content=audio_chunk)
            
            # Configure recognition - same as working voice changers
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.SAMPLE_RATE,
                language_code="ru-RU",  # Russian
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            # Perform recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            # Extract transcribed text - same as working versions
            transcribed_text = ""
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript + " "
            
            text = transcribed_text.strip()
            if text:
                return text
            else:
                return None
            
        except Exception as e:
            print(f"❌ STT: Transcription error: {e}")
            return None
    
    def write_phrase_to_file(self, phrase):
        """Append complete phrase to file for TTS reader"""
        try:
            with open(self.TEXT_FILE, 'a', encoding='utf-8') as f:
                f.write(phrase + "\n")  # Add newline to separate phrases
            print(f"📝 Appended phrase to file: '{phrase}'")
        except Exception as e:
            print(f"❌ Error writing to file: {e}")
    
    def run(self):
        """Main STT writer loop - using the working approach with silence detection"""
        print("🎤 STT Writer Fixed: Starting...")
        print(" Writing to file:", self.TEXT_FILE)
        print("📝 Using silence detection like working voice changers")
        print("=" * 50)
        
        try:
            # Initialize audio input - same as working versions
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("✅ STT: Audio input stream started")
            
            while self.running:
                try:
                    # Read audio chunk
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Add to buffers
                    self.audio_buffer.extend(audio_chunk)
                    self.silence_buffer.extend(audio_chunk)
                    
                    # Check for silence to trigger processing - same as working versions
                    if len(self.silence_buffer) >= int(self.SAMPLE_RATE * self.SILENCE_DURATION):
                        recent_audio = list(self.silence_buffer)[-int(self.SAMPLE_RATE * self.SILENCE_DURATION):]
                        audio_level = np.sqrt(np.mean(np.array(recent_audio, dtype=np.float32) ** 2))
                    
                        # If silence detected and we have enough audio, process it
                        if audio_level < self.SILENCE_THRESHOLD * 32768 and len(self.audio_buffer) >= self.BUFFER_SIZE:
                            # Extract audio for processing
                            stt_chunk = list(self.audio_buffer)
                            self.audio_buffer.clear()
                            self.silence_buffer.clear()
                            
                            # Convert to bytes
                            stt_audio = np.array(stt_chunk, dtype=np.int16).tobytes()
                            
                            print(f"🎤 STT: Processing {len(stt_audio)} bytes of audio (silence detected)...")
                            
                            # Transcribe audio
                            transcribed_text = self.transcribe_audio(stt_audio)
                            
                            if transcribed_text:
                                print(f"📝 STT: Transcribed: '{transcribed_text}'")
                                
                                # Accumulate text for complete phrases
                                if self.current_phrase:
                                    self.current_phrase += " " + transcribed_text
                                else:
                                    self.current_phrase = transcribed_text
                                
                                print(f"📚 STT: Current phrase: '{self.current_phrase}'")
                                
                                # Reset silence timer
                                self.silence_start = None
                                
                            else:
                                print("🔇 STT: No speech detected in chunk")
                                
                                # Start silence timer if not already started
                                if self.silence_start is None:
                                    self.silence_start = time.time()
                                else:
                                    # Check if silence duration exceeded
                                    silence_duration = time.time() - self.silence_start
                                    if silence_duration >= self.max_silence_duration and self.current_phrase:
                                        print(f"⏰ STT: Long silence detected ({silence_duration:.1f}s), writing complete phrase")
                                        self.write_phrase_to_file(self.current_phrase)
                                        self.current_phrase = ""
                                        self.silence_start = None
                    
                    time.sleep(0.01)  # Small delay
                    
                except Exception as e:
                    print(f"❌ STT: Error in audio processing: {e}")
                    break
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            self.audio.terminate()
            print("✅ STT Writer stopped")
            
        except Exception as e:
            print(f"❌ STT Writer error: {e}")

def main():
    """Main function"""
    # Check required environment variables
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
        print("   Please set the path to your Google Cloud service account key")
        return
    
    print("✅ Prerequisites check passed")
    
    # Create and run STT writer
    stt_writer = STTWriterFixed()
    
    try:
        stt_writer.run()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping STT writer...")
        stt_writer.running = False
        print(" STT writer stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()