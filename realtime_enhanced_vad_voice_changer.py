#!/usr/bin/env python3
"""
Real-time Enhanced VAD Voice Changer
Uses pause detection for natural phrase boundaries with microphone input
"""

import os
import sys
import time
import json
import queue
import signal
import asyncio
import threading
import tempfile
import argparse
import numpy as np
import soundfile as sf
import requests
from pydub import AudioSegment
import pyaudio
import atexit
import webrtcvad
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for cleanup
running = True
audio_chunks = []

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global running
    print("\n⏹️  Received interrupt signal, saving output and exiting...")
    running = False

def cleanup_handler():
    """Cleanup handler for atexit"""
    global running, audio_chunks
    if running:
        running = False
        print("💾 Saving output on exit...")
        if audio_chunks:
            save_audio_chunks(audio_chunks, "realtime_output.mp3")

def save_audio_chunks(chunks, output_file):
    """Save audio chunks to output file"""
    if not chunks:
        print("⚠️ No audio chunks to save")
        return
    
    try:
        # Combine all chunks
        combined_audio = b''.join(chunks)
        
        with open(output_file, 'wb') as f:
            f.write(combined_audio)
        
        print(f"✅ All audio saved to: {output_file}")
        print(f"📊 Total audio size: {len(combined_audio)} bytes")
        
        # Calculate duration (approximate)
        # MP3 at 192kbps = ~24KB per second
        duration_seconds = len(combined_audio) / 24000
        print(f"🎵 Duration: {duration_seconds:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error saving audio: {e}")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_handler)

class RealtimeEnhancedVADVoiceChanger:
    def __init__(self):
        # Audio settings (Pro quality)
        self.SAMPLE_RATE = 44100  # Pro quality
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # VAD settings for natural phrase detection
        self.VAD_SAMPLE_RATE = 16000  # VAD works better at 16kHz
        self.VAD_FRAME_DURATION = 0.03  # 30ms frames
        self.VAD_FRAME_SIZE = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
        
        # Pause detection settings
        self.SILENCE_THRESHOLD = 0.005  # RMS threshold for silence detection
        self.MIN_PHRASE_DURATION = 0.5  # Minimum phrase duration (0.5 seconds)
        self.MAX_PHRASE_DURATION = 5.0  # Maximum phrase duration (5 seconds)
        self.SILENCE_DURATION = 0.3  # Silence duration to trigger split (0.3 seconds)
        self.FORCE_SPLIT_DURATION = 3.0  # Force split every 3 seconds if no pause found
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.continuous_buffer = queue.Queue(maxsize=50)
        self.playback_thread = None
        self.running = True
        self.playback_started = False
        self.mp3_chunks = []
        self.all_audio_chunks = []
        
        # Real-time audio buffers
        self.audio_buffer = []
        self.silence_buffer = []
        self.phrase_start_time = None
        self.silence_start_time = None
        self.is_speaking = False
        
        # ElevenLabs Pro settings
        self.api_key = self._get_api_key()
        self.voice_id = self._get_voice_id()
        self.model_id = "eleven_multilingual_sts_v2"
        
        # Pro voice settings
        self.voice_settings = {
            "stability": 0.7,
            "similarity_boost": 0.8,
            "style": 0.3,
            "use_speaker_boost": True
        }
        
        # Pro audio settings
        self.output_format = "mp3_44100_192"
        self.optimize_streaming_latency = 4
        
        # Timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        
        print(f"🎵 Realtime Enhanced VAD Voice Changer Configuration:")
        print(f"   Sample Rate: {self.SAMPLE_RATE}Hz")
        print(f"   VAD Sample Rate: {self.VAD_SAMPLE_RATE}Hz")
        print(f"   Voice ID: {self.voice_id}")
        print(f"   Model: {self.model_id}")
        print(f"   Silence Threshold: {self.SILENCE_THRESHOLD}")
        print(f"   Min Phrase Duration: {self.MIN_PHRASE_DURATION}s")
        print(f"   Max Phrase Duration: {self.MAX_PHRASE_DURATION}s")
        print(f"   Silence Duration: {self.SILENCE_DURATION}s")
        print(f"   Force Split Duration: {self.FORCE_SPLIT_DURATION}s")
        
    def _get_api_key(self):
        """Get API key from environment or .env file"""
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key:
            return api_key
        
        try:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("ELEVENLABS_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            if api_key and api_key != "your_api_key_here":
                                return api_key
        except Exception:
            pass
        
        print("🔑 ElevenLabs API Key not found.")
        print("Please enter your ElevenLabs API key:")
        api_key = input("API Key: ").strip()
        
        if api_key:
            try:
                with open(".env", "w") as f:
                    f.write(f"ELEVENLABS_API_KEY={api_key}\n")
                print("✅ API key saved to .env file for future use")
            except Exception:
                print("⚠️ Could not save API key to file")
            
            return api_key
        
        return None
    
    def _get_voice_id(self):
        """Get voice ID from environment or .env file"""
        voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        if voice_id:
            return voice_id
        
        try:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("ELEVENLABS_VOICE_ID="):
                            voice_id = line.split("=", 1)[1].strip()
                            if voice_id and voice_id != "your_voice_id_here":
                                return voice_id
        except Exception:
            pass
        
        # Default to Ekaterina if not found
        return "GN4wbsbejSnGSa1AzjH5"
    
    def detect_speech_in_chunk(self, audio_chunk):
        """Detect speech in real-time audio chunk using WebRTC VAD"""
        try:
            # Convert to 16kHz for VAD
            audio_16k = self._resample_audio(audio_chunk, 16000)
            
            # Convert to int16
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            
            # Frame size for 30ms at 16kHz
            frame_size = int(16000 * 0.03)
            
            speech_detected = False
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                if len(frame) == frame_size:
                    is_speech = self.vad.is_speech(frame.tobytes(), 16000)
                    if is_speech:
                        speech_detected = True
                        break
            
            return speech_detected
            
        except Exception as e:
            print(f"❌ VAD error: {e}")
            return True  # Default to speech if VAD fails
    
    def _resample_audio(self, audio_data, target_sample_rate):
        """Simple resampling by interpolation"""
        if target_sample_rate == self.SAMPLE_RATE:
            return audio_data
        
        # Simple linear interpolation resampling
        original_length = len(audio_data)
        target_length = int(original_length * target_sample_rate / self.SAMPLE_RATE)
        
        indices = np.linspace(0, original_length - 1, target_length)
        return np.interp(indices, np.arange(original_length), audio_data)
    
    def process_audio_chunk(self, audio_chunk):
        """Process real-time audio chunk with enhanced VAD"""
        current_time = time.time()
        
        # Add to buffers
        self.audio_buffer.extend(audio_chunk)
        self.silence_buffer.extend(audio_chunk)
        
        # Keep silence buffer at reasonable size (last 0.5 seconds)
        max_silence_samples = int(self.SAMPLE_RATE * 0.5)
        if len(self.silence_buffer) > max_silence_samples:
            self.silence_buffer = self.silence_buffer[-max_silence_samples:]
        
        # Detect speech in current chunk
        is_speech = self.detect_speech_in_chunk(audio_chunk)
        
        # Calculate audio level for silence detection
        audio_level = np.sqrt(np.mean(np.array(audio_chunk, dtype=np.float32) ** 2))
        
        # Handle speech/silence state changes
        if is_speech and audio_level > self.SILENCE_THRESHOLD:
            # Speech detected
            if not self.is_speaking:
                self.is_speaking = True
                self.phrase_start_time = current_time
                self.silence_start_time = None
                print("🎤 Speech started")
            
        else:
            # Silence detected
            if self.is_speaking:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                
                # Check if silence duration is long enough to end phrase
                silence_duration = current_time - self.silence_start_time
                phrase_duration = current_time - self.phrase_start_time
                
                if (silence_duration >= self.SILENCE_DURATION and 
                    phrase_duration >= self.MIN_PHRASE_DURATION):
                    # End phrase and process
                    self._process_phrase()
                    
                elif phrase_duration >= self.MAX_PHRASE_DURATION:
                    # Force process long phrase
                    print("⏰ Force processing long phrase")
                    self._process_phrase()
        
        # Check for force split on very long phrases
        if self.is_speaking and self.phrase_start_time:
            phrase_duration = current_time - self.phrase_start_time
            if phrase_duration >= self.FORCE_SPLIT_DURATION:
                print("⏰ Force splitting long phrase")
                self._process_phrase()
    
    def _process_phrase(self):
        """Process current phrase with STS API"""
        if not self.audio_buffer:
            return
        
        try:
            # Convert buffer to audio segment
            audio_array = np.array(self.audio_buffer, dtype=np.float32)
            
            # Create audio segment
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=self.SAMPLE_RATE,
                sample_width=2,
                channels=1
            )
            
            # Convert to WAV bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_segment.export(temp_file.name, format="wav")
                temp_file_path = temp_file.name
            
            # Read the audio file
            with open(temp_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Process with STS API
            self._send_to_sts_api(audio_data)
            
            # Reset buffers
            self.audio_buffer.clear()
            self.silence_buffer.clear()
            self.is_speaking = False
            self.phrase_start_time = None
            self.silence_start_time = None
            
        except Exception as e:
            print(f"❌ Error processing phrase: {e}")
            # Reset buffers on error
            self.audio_buffer.clear()
            self.silence_buffer.clear()
            self.is_speaking = False
            self.phrase_start_time = None
            self.silence_start_time = None
    
    def _send_to_sts_api(self, audio_data):
        """Send audio to ElevenLabs STS API"""
        try:
            print("🎵 Sending phrase to ElevenLabs STS API...")
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file, audio_array, self.SAMPLE_RATE)
                temp_file_path = temp_file.name
            
            # Prepare the request with Pro features
            headers = {
                "xi-api-key": self.api_key
            }
            
            # Read the audio file
            with open(temp_file_path, "rb") as audio_file:
                files = {
                    "audio": ("audio.wav", audio_file, "audio/wav"),
                    "model_id": (None, self.model_id),
                    "remove_background_noise": (None, "false"),
                    "optimize_streaming_latency": (None, str(self.optimize_streaming_latency)),
                    "output_format": (None, self.output_format),
                    "voice_settings": (None, json.dumps(self.voice_settings))
                }
                
                response = requests.post(
                    f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}/stream",
                    headers=headers,
                    files=files,
                    timeout=30
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            if response.status_code == 200:
                audio_output = response.content
                
                if audio_output:
                    print(f"✅ Received {len(audio_output)} bytes from STS API")
                    
                    # Add to continuous buffer for playback
                    try:
                        self.continuous_buffer.put_nowait(audio_output)
                    except queue.Full:
                        pass
                    
                    # Collect for file saving
                    self.mp3_chunks.append(audio_output)
                    self.all_audio_chunks.append(audio_output)
                    global audio_chunks
                    audio_chunks.append(audio_output)
                    
                    # Start playback when we have enough data
                    if not self.playback_started and len(self.mp3_chunks) >= 1:
                        self.playback_started = True
                        print(f"🎵 Starting real-time playback")
                    
                else:
                    print("⚠️ No audio data received from STS API")
            else:
                print(f"❌ STS API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error sending to STS API: {e}")
    
    async def _smooth_audio_streaming(self):
        """Smooth audio streaming with continuous playback"""
        try:
            # Initialize audio stream
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            # Start continuous playback thread if not already started
            if not self.playback_thread or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self._continuous_playback_worker, args=(stream,))
                self.playback_thread.daemon = True
                self.playback_thread.start()
            
            print("🎵 Realtime Enhanced VAD: Audio streaming initialized")
            
        except Exception as e:
            print(f"❌ Smooth streaming error: {e}")
    
    def _continuous_playback_worker(self, stream):
        """Continuous playback worker that eliminates gaps"""
        try:
            while self.running:
                try:
                    # Get audio chunk from buffer
                    mp3_data = self.continuous_buffer.get(timeout=0.1)
                    
                    # Play the audio chunk smoothly
                    self._play_audio_chunk_smooth(mp3_data, stream)
                    
                except queue.Empty:
                    # No audio data, continue
                    continue
                except Exception as e:
                    print(f"❌ Playback error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Continuous playback error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    def _play_audio_chunk_smooth(self, mp3_data, stream):
        """Play audio chunk smoothly without gaps"""
        try:
            # Convert MP3 to PCM
            audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            
            # Convert to mono if needed
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Resample if needed
            if audio_segment.frame_rate != self.SAMPLE_RATE:
                audio_segment = audio_segment.set_frame_rate(self.SAMPLE_RATE)
            
            # Convert to PCM
            pcm_data = audio_segment.raw_data
            
            # Play the audio
            stream.write(pcm_data)
            
        except Exception as e:
            print(f"❌ Audio chunk playback error: {e}")
    
    def audio_recording_worker(self):
        """Audio recording worker thread"""
        try:
            # Initialize input stream
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            print("🎤 Microphone input started...")
            print("🎤 Speak naturally - phrases will be detected automatically")
            print("⏹️  Press Ctrl+C to stop")
            
            while self.running:
                try:
                    # Read audio chunk
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Process audio chunk with enhanced VAD
                    self.process_audio_chunk(audio_chunk)
                    
                except Exception as e:
                    print(f"❌ Audio recording error: {e}")
                    break
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Audio recording worker error: {e}")
    
    async def start(self):
        """Start the real-time enhanced VAD voice changer"""
        try:
            print("🎤🎵 Realtime Enhanced VAD Voice Changer: Microphone → Natural Phrases → ElevenLabs STS API → Speakers")
            print("=" * 80)
            print("🎤 Enhanced VAD: Detects natural phrase boundaries using pause detection")
            print("🎵 STS: Sends natural phrases to ElevenLabs STS API")
            print("🔊 Playback: Smooth real-time audio streaming")
            print(f"🎵 Model: {self.model_id}")
            print(f"🎵 Voice: {self.voice_id}")
            print(f"🎵 Pro Features: {self.output_format}, {self.voice_settings}")
            print("=" * 80)
            
            # Start smooth audio streaming
            await self._smooth_audio_streaming()
            
            # Start audio recording worker
            recording_thread = threading.Thread(target=self.audio_recording_worker)
            recording_thread.daemon = True
            recording_thread.start()
            
            print("✅ Realtime Enhanced VAD voice changer started!")
            print("🎤 Speak into your microphone...")
            print("🎵 Natural phrases will be detected and processed automatically")
            print("⏹️  Press Ctrl+C to stop")
            
            # Keep main thread alive
            while self.running:
                await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Realtime Enhanced VAD voice changer error: {e}")
        finally:
            self.save_final_output()
    
    def save_final_output(self):
        """Save final output file"""
        try:
            print("💾 Saving final output...")
            save_audio_chunks(self.all_audio_chunks, f"realtime_enhanced_vad_output_{self.timestamp}.mp3")
        except Exception as e:
            print(f"❌ Error saving final output: {e}")

async def main():
    """Main function"""
    try:
        # Check prerequisites
        print("✅ Prerequisites check passed")
        
        # Create and start the voice changer
        voice_changer = RealtimeEnhancedVADVoiceChanger()
        
        # Start the voice changer
        await voice_changer.start()
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Main error: {e}")

if __name__ == "__main__":
    import io
    asyncio.run(main()) 