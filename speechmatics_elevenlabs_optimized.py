import asyncio
import base64
import json
import os
import threading
import time
import uuid
import websockets
import pyaudio
import wave
from dotenv import load_dotenv
import queue

# Load environment variables
load_dotenv()

# Configuration
EL_API_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE_ID = os.getenv("EL_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
SM_API_KEY = os.getenv("SM_API_KEY")

# Audio configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class OptimizedSpeechmaticsSTT:
    """Optimized Speechmatics STT with WebSocket connection"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or SM_API_KEY
        self.websocket = None
        self.connected = False
        
        if not self.api_key:
            raise ValueError("SM_API_KEY not found in environment variables")
    
    async def connect(self):
        """Connect to Speechmatics STT"""
        if self.connected:
            return
            
        print("🔗 Connecting to Speechmatics STT...")
        
        # Speechmatics WebSocket URL
        url = "wss://eu2.rt.speechmatics.com/v2"
        
        try:
            self.websocket = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
                ping_interval=30,
                ping_timeout=60,
                close_timeout=5
            )
            print("✅ Connected to Speechmatics STT")
            
            # Send configuration with correct format
            config = {
                "message": "StartRecognition",
                "audio_format": {
                    "type": "raw",
                    "encoding": "pcm_s16le",  # Changed from pcm_f32le to pcm_s16le
                    "sample_rate": 16000
                },
                "transcription_config": {
                    "language": "en",
                    "enable_partials": True,
                    "max_delay": 1.0,  # Reduced for lower latency
                    "max_delay_mode": "flexible",
                    "operating_point": "enhanced"
                }
            }
            
            await self.websocket.send(json.dumps(config))
            self.connected = True
            print("🚀 STT ready for transcription!")
            
        except Exception as e:
            print(f"❌ STT connection failed: {e}")
            self.connected = False
            raise
    
    async def transcribe_audio(self, audio_data):
        """Send audio data for transcription"""
        if not self.connected:
            raise RuntimeError("STT not connected. Call connect() first.")
        
        try:
            # Send audio data directly (not base64 encoded)
            await self.websocket.send(audio_data)
            
            # Wait for transcription response
            transcript = ""
            max_wait = 10  # Maximum messages to check
            
            for _ in range(max_wait):
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    
                    message_type = data.get("message")
                    
                    if message_type == "AddTranscript":
                        # Final transcript
                        transcript = data.get("metadata", {}).get("transcript", "")
                        if transcript:
                            return transcript
                    elif message_type == "AddPartialTranscript":
                        # Partial transcript
                        transcript = data.get("metadata", {}).get("transcript", "")
                        if transcript:
                            return transcript
                            
                except asyncio.TimeoutError:
                    break
                except json.JSONDecodeError:
                    continue
            
            return transcript if transcript else None
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
    
    async def close(self):
        """Close STT connection"""
        if self.connected and self.websocket:
            try:
                await self.websocket.close()
                print("🔌 STT connection closed")
            except Exception as e:
                print(f"❌ Error closing STT connection: {e}")
            finally:
                self.connected = False

class OptimizedElevenLabsTTS:
    """Optimized ElevenLabs TTS with real-time streaming"""
    
    def __init__(self, voice_id=None, api_key=None):
        self.voice_id = voice_id or EL_VOICE_ID
        self.api_key = api_key or EL_API_KEY
        self.websocket = None
        self.connected = False
        self.context_id = None
        self.audio_queue = queue.Queue()
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
    
    async def connect(self):
        """Connect to ElevenLabs TTS"""
        if self.connected:
            return
            
        print("🔗 Connecting to ElevenLabs TTS...")
        
        # Connect to multi-stream endpoint
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"
        params = {
            "model_id": "eleven_flash_v2_5",
            "output_format": "pcm_16000",
            "auto_mode": "true"
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?{query_string}"
        
        try:
            self.websocket = await websockets.connect(
                full_url,
                extra_headers={"xi-api-key": self.api_key},
                close_timeout=5
            )
            print("✅ Connected to ElevenLabs TTS")
            
            # Initialize context
            self.context_id = f"conv_{uuid.uuid4().hex[:8]}"
            context_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8
                },
                "context_id": self.context_id
            }
            
            await self.websocket.send(json.dumps(context_message))
            print(f"✅ TTS context initialized: {self.context_id}")
            
            self.connected = True
            print("🚀 TTS ready for speech synthesis!")
            
        except Exception as e:
            print(f"❌ TTS connection failed: {e}")
            self.connected = False
            raise
    
    async def speak(self, text):
        """Convert text to speech"""
        if not self.connected:
            raise RuntimeError("TTS not connected. Call connect() first.")
        
        print(f"🗣️  Speaking: '{text}'")
        
        try:
            # Send text
            message = {
                "text": text + " ",
                "context_id": self.context_id,
                "flush": True
            }
            
            await self.websocket.send(json.dumps(message))
            
            # Receive audio
            audio_chunks = []
            max_chunks = 10
            
            try:
                async with asyncio.timeout(5):
                    async for message in self.websocket:
                        try:
                            data = json.loads(message)
                            
                            if "audio" in data:
                                audio_bytes = base64.b64decode(data["audio"])
                                audio_chunks.append(audio_bytes)
                                
                                if data.get("is_final", False):
                                    break
                            
                            if len(audio_chunks) >= max_chunks:
                                break
                                
                        except json.JSONDecodeError:
                            break
                            
            except asyncio.TimeoutError:
                print("⏰ TTS timeout")
            
            return b''.join(audio_chunks)
            
        except Exception as e:
            print(f"❌ TTS failed: {e}")
            return None
    
    async def close(self):
        """Close TTS connection"""
        if self.connected and self.websocket:
            try:
                # Close context
                close_message = {
                    "context_id": self.context_id,
                    "close_context": True
                }
                await self.websocket.send(json.dumps(close_message))
                
                # Close socket
                close_socket_message = {
                    "close_socket": True
                }
                await self.websocket.send(json.dumps(close_socket_message))
                
                await self.websocket.close()
                print("🔌 TTS connection closed")
                
            except Exception as e:
                print(f"❌ Error closing TTS connection: {e}")
            finally:
                self.connected = False

class OptimizedSpeechToSpeech:
    """Optimized Speech-to-Speech pipeline with single connections"""
    
    def __init__(self):
        self.stt = OptimizedSpeechmaticsSTT()
        self.tts = OptimizedElevenLabsTTS()
        self.audio_queue = queue.Queue()
        self.player = None
        self.player_thread = None
    
    def start_audio_player(self):
        """Start audio player for output"""
        if self.player_thread is None or not self.player_thread.is_alive():
            self.player = SimpleAudioPlayer(self.audio_queue)
            self.player_thread = threading.Thread(target=self.player.run, daemon=True)
            self.player_thread.start()
            print("🎵 Audio player started")
    
    async def connect(self):
        """Connect both STT and TTS services"""
        print("🔗 Connecting to Speech-to-Speech pipeline...")
        
        # Connect both services
        await asyncio.gather(
            self.stt.connect(),
            self.tts.connect()
        )
        
        print("✅ Speech-to-Speech pipeline ready!")
    
    async def process_audio(self, audio_data):
        """Process audio through STT -> TTS pipeline"""
        try:
            # Step 1: Speech to Text
            print("🎤 Transcribing audio...")
            transcript = await self.stt.transcribe_audio(audio_data)
            
            if not transcript:
                print("❌ No transcript received")
                return None
            
            print(f"📝 Transcript: '{transcript}'")
            
            # Step 2: Text to Speech
            print("🔊 Converting to speech...")
            audio_output = await self.tts.speak(transcript)
            
            if audio_output:
                # Play the audio
                self.audio_queue.put_nowait(audio_output)
                print("🎵 Audio played")
                return audio_output
            else:
                print("❌ No audio generated")
                return None
                
        except Exception as e:
            print(f"❌ Pipeline processing failed: {e}")
            return None
    
    async def close(self):
        """Close all connections"""
        await asyncio.gather(
            self.stt.close(),
            self.tts.close()
        )
        
        if self.player:
            self.player.stop()
            if self.player_thread:
                self.player_thread.join(timeout=1)

class SimpleAudioPlayer:
    """Simple audio player for output"""
    
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
    
    def run(self):
        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    if chunk:
                        self.stream.write(chunk)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Audio player error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Failed to initialize audio player: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()
    
    def stop(self):
        self.running = False

async def main():
    """Main function demonstrating optimized Speech-to-Speech"""
    
    # Create pipeline
    pipeline = OptimizedSpeechToSpeech()
    
    try:
        # Start audio player
        pipeline.start_audio_player()
        
        # Connect to services
        await pipeline.connect()
        
        # Test with real microphone input
        print("\n🎤 Starting real-time microphone input...")
        
        # Initialize PyAudio for microphone input
        p = pyaudio.PyAudio()
        
        # Open microphone stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("🎤 Microphone active - speak now!")
        print("Press Ctrl+C to stop...")
        
        try:
            # Record audio for 5 seconds
            audio_data = b''
            for _ in range(50):  # 5 seconds at 100ms chunks
                chunk = stream.read(CHUNK_SIZE)
                audio_data += chunk
                await asyncio.sleep(0.1)  # 100ms delay
                
            print(f"🎤 Recorded {len(audio_data)} bytes of audio")
            
            # Process through pipeline
            result = await pipeline.process_audio(audio_data)
            
        except KeyboardInterrupt:
            print("\n⏹️  Stopping recording...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        
        if result:
            print("✅ Speech-to-Speech pipeline completed successfully!")
        else:
            print("❌ Pipeline failed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main()) 