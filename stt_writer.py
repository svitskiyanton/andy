#!/usr/bin/env python3
"""
STT Writer - Transcribes audio and writes to text file
"""

import os
import pyaudio
import time
import numpy as np
from google.cloud import speech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class STTWriter:
    def __init__(self):
        # STT settings
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.CHUNK_SIZE = 1024 * 4
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Google Cloud Speech client
        self.speech_client = None
        self.init_google_client()
        
        # File settings
        self.TEXT_FILE = "voice_changer_text.txt"
        
        # Control
        self.running = True
    
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
        """Transcribe audio chunk using Google Speech-to-Text"""
        if not self.speech_client:
            return None
        
        try:
            # Create audio content
            audio = speech.RecognitionAudio(content=audio_chunk)
            
            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.SAMPLE_RATE,
                language_code="ru-RU",  # Russian
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            # Perform recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            # Extract transcribed text
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
    
    def write_to_file(self, text):
        """Write text to file"""
        try:
            with open(self.TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"📝 Wrote to file: '{text}'")
        except Exception as e:
            print(f"❌ Error writing to file: {e}")
    
    def run(self):
        """Main STT writer loop"""
        print("🎤 STT Writer: Starting...")
        print("📁 Writing to file:", self.TEXT_FILE)
        print("=" * 50)
        
        try:
            # Initialize audio input
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("✅ STT: Audio input stream started")
            
            # Buffer for accumulating audio chunks
            audio_buffer = b""
            buffer_duration = 1.0  # Process every 1 second for better responsiveness
            chunk_duration = self.CHUNK_SIZE / self.SAMPLE_RATE
            chunks_per_buffer = int(buffer_duration / chunk_duration)
            
            chunk_count = 0
            silence_count = 0
            max_silence_chunks = 3  # Process after 3 chunks of silence
            
            while self.running:
                try:
                    # Read audio chunk
                    audio_chunk = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_buffer += audio_chunk
                    chunk_count += 1
                    
                    # Check audio level for silence detection
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                    audio_level = np.abs(audio_data).mean()
                    
                    # Process buffer when we have enough chunks OR after silence
                    should_process = (chunk_count >= chunks_per_buffer) or (silence_count >= max_silence_chunks and len(audio_buffer) > 0)
                    
                    if audio_level < 500:  # Silence threshold
                        silence_count += 1
                    else:
                        silence_count = 0
                    
                    if should_process and audio_buffer:
                        print(f"🎤 STT: Processing {len(audio_buffer)} bytes of audio (level: {audio_level:.0f})...")
                        
                        # Transcribe audio
                        transcribed_text = self.transcribe_audio(audio_buffer)
                        
                        if transcribed_text:
                            print(f"📝 STT: Transcribed: '{transcribed_text}'")
                            
                            # Write to file
                            self.write_to_file(transcribed_text)
                        else:
                            print("🔇 STT: No speech detected")
                        
                        # Reset buffer
                        audio_buffer = b""
                        chunk_count = 0
                        silence_count = 0
                    
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
    stt_writer = STTWriter()
    
    try:
        stt_writer.run()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping STT writer...")
        stt_writer.running = False
        print("👋 STT writer stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 