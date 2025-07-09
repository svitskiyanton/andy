#!/usr/bin/env python3
"""
Simple Multi-Process Voice Changer
Robust implementation with proper inter-process communication
"""

import multiprocessing as mp
import time
import numpy as np
import pyaudio
import os
import signal
import sys

class SimpleMultiProcessVoiceChanger:
    def __init__(self):
        """Initialize the voice changer"""
        # Audio settings
        self.SAMPLE_RATE = 16000
        self.CHUNK_SIZE = 1024
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # Process management
        self.is_running = False
        self.processes = []
        
        # Shared queues - single queue for STT to TTS communication
        self.audio_queue = mp.Queue(maxsize=10)  # Audio chunks from main to STT
        self.shared_queue = mp.Queue(maxsize=20)  # STT results to TTS
        
        # Statistics
        self.session_start_time = None
        self.total_phrases_processed = 0
        
        # Audio interface
        self.audio = pyaudio.PyAudio()
        
        print("🎤 Simple Multi-Process Voice Changer initialized")
        print("🔧 Using single shared queue for lower latency")
    
    def start_streaming(self):
        """Start the voice changer"""
        print("\n🚀 Starting multi-process voice changer...")
        print("📋 Press Ctrl+C to stop")
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            self.session_start_time = time.time()
            self.is_running = True
            
            # Start processes
            self._start_processes()
            
            # Start audio recording in main process
            self._start_audio_recording()
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.stop_streaming()
    
    def stop_streaming(self):
        """Stop the voice changer"""
        print("\n🛑 Stopping multi-process voice changer...")
        self.is_running = False
        
        # Stop processes
        self._stop_processes()
        
        # Print statistics
        self._print_stats()
        
        print("✅ Multi-process voice changer stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n🛑 Received stop signal...")
        self.stop_streaming()
        sys.exit(0)
    
    def _start_processes(self):
        """Start STT and TTS processes"""
        print("🔄 Starting processes...")
        
        # Start STT process
        stt_process = mp.Process(
            target=stt_worker,
            args=(self.audio_queue, self.shared_queue)
        )
        stt_process.start()
        self.processes.append(stt_process)
        
        # Start TTS process
        tts_process = mp.Process(
            target=tts_worker,
            args=(self.shared_queue,)
        )
        tts_process.start()
        self.processes.append(tts_process)
        
        print(f"✅ Started {len(self.processes)} processes")
    
    def _stop_processes(self):
        """Stop all processes"""
        print("🛑 Stopping processes...")
        
        # Send stop signals
        try:
            self.audio_queue.put(None, timeout=1.0)
        except:
            pass
        
        try:
            self.shared_queue.put(None, timeout=1.0)
        except:
            pass
        
        # Wait for processes to finish
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
                if process.is_alive():
                    process.kill()
        
        print("✅ All processes stopped")
    
    def _start_audio_recording(self):
        """Start audio recording"""
        print("🎤 Audio recording started...")
        
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            # Process in 2-second chunks for lower latency
            chunk_duration = 2.0
            chunk_samples = int(self.SAMPLE_RATE * chunk_duration)
            current_chunk = []
            
            while self.is_running:
                try:
                    # Read audio chunk
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Add to current chunk
                    current_chunk.extend(audio_chunk)
                    
                    # When chunk is full, send to STT process
                    if len(current_chunk) >= chunk_samples:
                        # Convert to bytes and send to STT process
                        try:
                            audio_bytes = np.array(current_chunk, dtype=np.int16).tobytes()
                            self.audio_queue.put_nowait(audio_bytes)
                        except:
                            # Queue full, skip this chunk
                            pass
                        current_chunk = []
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"❌ Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
    
    def _print_stats(self):
        """Print session statistics"""
        if self.session_start_time:
            session_duration = time.time() - self.session_start_time
            print(f"\n📊 Session Duration: {session_duration:.1f}s")
            print(f"📊 Total Phrases: {self.total_phrases_processed}")
            print(f"📊 Phrases per Minute: {self.total_phrases_processed / (session_duration / 60):.1f}")

def stt_worker(audio_queue, shared_queue):
    """STT process worker - sends results directly to shared queue"""
    print("🔍 STT worker started...")
    
    # Import here to avoid issues in main process
    from google.cloud import speech
    from google.oauth2 import service_account
    
    # Initialize Google client with service account
    try:
        # Try to use service account key file first
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path and os.path.exists(credentials_path):
            print(f"🔑 Using service account key: {credentials_path}")
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            speech_client = speech.SpeechClient(credentials=credentials)
        else:
            # Fallback to default credentials
            print("🔑 Using default credentials")
            speech_client = speech.SpeechClient()
        
        print("✅ Google STT client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Google STT: {e}")
        print("💡 Try setting GOOGLE_APPLICATION_CREDENTIALS to your service account key file path")
        return
    
    while True:
        try:
            # Get audio chunk
            audio_chunk = audio_queue.get(timeout=1.0)
            
            if audio_chunk is None:
                print("🛑 STT worker received stop signal")
                break
            
            # Process STT
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_level = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Only process if audio level is above threshold
            if audio_level < 50:
                continue
            
            audio = speech.RecognitionAudio(content=audio_chunk)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="ru-RU",
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            response = speech_client.recognize(config=config, audio=audio)
            
            transcribed_text = ""
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript + " "
            
            if transcribed_text.strip():
                # Send directly to shared queue for TTS
                shared_queue.put({
                    'text': transcribed_text.strip(),
                    'timestamp': time.time()
                })
                print(f"🔍 STT → TTS: '{transcribed_text.strip()}'")
            
        except Exception as e:
            if "Empty" not in str(e):
                print(f"❌ STT worker error: {e}")
            continue
    
    print("✅ STT worker stopped")

def tts_worker(shared_queue):
    """TTS process worker - receives from shared queue"""
    print("🎵 TTS worker started...")
    
    # Import here to avoid issues in main process
    import websockets
    import asyncio
    import base64
    import json
    import os
    import numpy as np
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    model_id = "eleven_multilingual_v2"
    
    async def tts_stream(text):
        uri = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
            f"?model_id={model_id}&optimize_streaming_latency=4"
        )
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        try:
            print(f"🎵 TTS processing: '{text}'")
            
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                # Send initialization
                init_message = {
                    "text": " ",
                    "voice_settings": {
                        "speed": 1,
                        "stability": 0.5,
                        "similarity_boost": 0.8
                    },
                    "xi_api_key": api_key
                }
                await websocket.send(json.dumps(init_message))
                
                # Send text
                text_message = {
                    "text": text,
                    "try_trigger_generation": True
                }
                await websocket.send(json.dumps(text_message))
                
                # Send end marker
                await websocket.send(json.dumps({"text": ""}))
                
                # Receive and play audio
                from pydub import AudioSegment
                import io
                import pyaudio
                audio = pyaudio.PyAudio()
                stream = None
                
                async for message in websocket:
                    data = json.loads(message)
                    if "audio" in data and data["audio"]:
                        audio_bytes = base64.b64decode(data["audio"])
                        
                        # Playback
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                        pcm_data = audio_segment.get_array_of_samples()
                        pcm_float = np.array(pcm_data, dtype=np.float32) / 32768.0
                        
                        if stream is None:
                            stream = audio.open(
                                format=pyaudio.paFloat32, 
                                channels=1, 
                                rate=44100, 
                                output=True, 
                                frames_per_buffer=1024
                            )
                        
                        chunk_size = 1024
                        for i in range(0, len(pcm_float), chunk_size):
                            chunk = pcm_float[i:i + chunk_size]
                            if len(chunk) < chunk_size:
                                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                            stream.write(chunk.astype(np.float32).tobytes())
                
                if stream:
                    stream.stop_stream()
                    stream.close()
                audio.terminate()
                        
        except Exception as e:
            print(f"❌ TTS streaming error: {e}")
    
    while True:
        try:
            # Get phrase from shared queue
            phrase_data = shared_queue.get(timeout=1.0)
            
            if phrase_data is None:
                print("🛑 TTS worker received stop signal")
                break
            
            text = phrase_data['text']
            
            # Process TTS immediately
            asyncio.run(tts_stream(text))
            
        except Exception as e:
            if "Empty" not in str(e):
                print(f"❌ TTS worker error: {e}")
            continue
    
    print("✅ TTS worker stopped")

def main():
    """Main function"""
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    voice_changer = SimpleMultiProcessVoiceChanger()
    voice_changer.start_streaming()

if __name__ == "__main__":
    main() 