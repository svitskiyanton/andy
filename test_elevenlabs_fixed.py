import asyncio
import base64
import json
import os
import threading
import time
import uuid
import websockets
import pyaudio
from dotenv import load_dotenv
import queue

# Load environment variables
load_dotenv()

# Configuration
EL_API_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE_ID = os.getenv("EL_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Audio configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Latency tracking
class LatencyTracker:
    def __init__(self):
        self.timestamps = {}
        self.latencies = {}
    
    def mark(self, event: str):
        """Mark a timestamp for an event"""
        self.timestamps[event] = time.time()
        print(f"⏱️  {event}: {self.timestamps[event]:.3f}s")
    
    def measure(self, start_event: str, end_event: str, description: str = ""):
        """Measure latency between two events"""
        if start_event in self.timestamps and end_event in self.timestamps:
            latency = (self.timestamps[end_event] - self.timestamps[start_event]) * 1000
            self.latencies[f"{start_event}->{end_event}"] = latency
            print(f"📊 {description}: {latency:.1f}ms")
            return latency
        return None
    
    def summary(self):
        """Print latency summary"""
        print("\n" + "="*50)
        print("📈 LATENCY SUMMARY")
        print("="*50)
        for key, latency in self.latencies.items():
            print(f"⏱️  {key}: {latency:.1f}ms")
        print("="*50)

# Global latency tracker
latency = LatencyTracker()

class SimpleAudioPlayer(threading.Thread):
    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
        self.daemon = True
        
    def run(self):
        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            print("🎵 Audio player started")
            
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

class OptimizedElevenLabsManager:
    """Optimized ElevenLabs manager with pre-connected WebSocket"""
    
    def __init__(self, voice_id, api_key):
        self.voice_id = voice_id
        self.api_key = api_key
        self.websocket = None
        self.connected = False
        self.context_id = None
        self.audio_queue = queue.Queue()
        self.player = None
        
    async def connect(self):
        """Connect once and keep connection alive"""
        if self.connected:
            return
            
        print("🔗 Connecting to ElevenLabs Multi-Stream...")
        latency.mark("connection_start")
        
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
            latency.mark("websocket_connected")
            print("✅ Connected to ElevenLabs Multi-Stream")
            
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
            latency.mark("context_initialized")
            print(f"✅ Context initialized: {self.context_id}")
            
            self.connected = True
            print("🚀 Ready for speech requests!")
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
    
    async def speak(self, text, description=""):
        """Send text and receive audio using existing connection"""
        if not self.connected:
            print("❌ Not connected. Call connect() first.")
            return
            
        print(f"\n🗣️  Speaking: '{text}'")
        latency.mark(f"speech_start_{description}")
        
        try:
            # Send text
            message = {
                "text": text + " ",
                "context_id": self.context_id,
                "flush": True
            }
            
            await self.websocket.send(json.dumps(message))
            latency.mark(f"text_sent_{description}")
            
            # Receive audio with shorter timeout
            first_audio_received = False
            audio_chunks_received = 0
            max_chunks = 5  # Reduced to prevent long waits
            
            try:
                async with asyncio.timeout(3):  # Reduced timeout to 3 seconds
                    async for message in self.websocket:
                        try:
                            data = json.loads(message)
                            
                            if "audio" in data:
                                if not first_audio_received:
                                    latency.mark(f"first_audio_{description}")
                                    first_audio_received = True
                                    print("🎵 First audio chunk received")
                                
                                audio_bytes = base64.b64decode(data["audio"])
                                self.audio_queue.put_nowait(audio_bytes)
                                audio_chunks_received += 1
                                print(f"🎵 Audio chunk {audio_chunks_received} received")
                                
                                if data.get("is_final", False):
                                    latency.mark(f"final_audio_{description}")
                                    print("🏁 Final audio chunk")
                                    break
                            
                            # Break after receiving enough chunks
                            if audio_chunks_received >= max_chunks:
                                latency.mark(f"max_chunks_{description}")
                                print(f"🏁 Reached max chunks ({max_chunks}), stopping")
                                break
                                
                        except json.JSONDecodeError:
                            print(f"❌ Invalid JSON: {message}")
                            break
                            
            except asyncio.TimeoutError:
                print("⏰ Timeout waiting for audio")
            except Exception as e:
                print(f"❌ Error receiving audio: {e}")
            
            # Measure latencies
            latency.measure(f"speech_start_{description}", f"text_sent_{description}", f"Text sending ({description})")
            latency.measure(f"text_sent_{description}", f"first_audio_{description}", f"First audio latency ({description})")
            if f"final_audio_{description}" in latency.timestamps:
                latency.measure(f"text_sent_{description}", f"final_audio_{description}", f"Total TTS latency ({description})")
            
        except Exception as e:
            print(f"❌ Speech failed: {e}")
    
    async def close(self):
        """Close the connection properly"""
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
                
                # Close connection
                await self.websocket.close()
                print("🔌 Connection closed")
                
            except Exception as e:
                print(f"❌ Error closing connection: {e}")
            finally:
                self.connected = False

async def test_optimized_elevenlabs():
    """Test optimized ElevenLabs with pre-connected WebSocket"""
    
    if not EL_API_KEY:
        print("❌ ELEVENLABS_API_KEY not found")
        return
    
    print("🧪 Optimized ElevenLabs Test with Pre-Connected WebSocket")
    print("=" * 70)
    print(f"Voice ID: {EL_VOICE_ID}")
    print("=" * 70)
    
    # Create audio player
    audio_queue = queue.Queue()
    player = SimpleAudioPlayer(audio_queue)
    player.start()
    
    # Create ElevenLabs manager
    manager = OptimizedElevenLabsManager(EL_VOICE_ID, EL_API_KEY)
    manager.audio_queue = audio_queue
    
    try:
        # Connect once (this is the only connection cost)
        latency.mark("test_start")
        await manager.connect()
        latency.measure("test_start", "websocket_connected", "Initial connection (one-time cost)")
        
        # Multiple speech requests (reusing connection)
        test_phrases = [
            "Hello, how are you today?",
            "The weather is beautiful outside.",
            "Thank you for using our service.",
            "Have a wonderful day ahead!"
        ]
        
        for i, phrase in enumerate(test_phrases, 1):
            await manager.speak(phrase, f"phrase_{i}")
        
        print("\n✅ All speech requests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Cleanup
        await manager.close()
        player.stop()
        await asyncio.sleep(1)
        player.join()
        
        # Print final summary
        latency.summary()
        print("\n✅ Optimized test completed!")

async def main():
    await test_optimized_elevenlabs()

if __name__ == "__main__":
    asyncio.run(main()) 