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

class DebugTimer:
    """Debug timer for tracking operation timings"""
    
    def __init__(self):
        self.start_time = time.time()
        self.events = []
    
    def mark(self, event_name, description=""):
        """Mark an event with timestamp"""
        timestamp = time.time() - self.start_time
        self.events.append({
            'time': timestamp,
            'event': event_name,
            'description': description
        })
        print(f"⏱️  [{timestamp:.3f}s] {event_name}: {description}")
    
    def measure(self, start_event, end_event, description=""):
        """Measure time between two events"""
        start_time = None
        end_time = None
        
        for event in self.events:
            if event['event'] == start_event:
                start_time = event['time']
            elif event['event'] == end_event:
                end_time = event['time']
        
        if start_time and end_time:
            duration = (end_time - start_time) * 1000  # Convert to ms
            print(f"📊 {description}: {duration:.1f}ms")
            return duration
        return None
    
    def summary(self):
        """Print timing summary"""
        print("\n" + "="*60)
        print("📈 DEBUG TIMING SUMMARY")
        print("="*60)
        
        for i, event in enumerate(self.events):
            if i > 0:
                prev_time = self.events[i-1]['time']
                interval = (event['time'] - prev_time) * 1000
                print(f"⏱️  [{event['time']:.3f}s] {event['event']} (+{interval:.1f}ms)")
            else:
                print(f"⏱️  [{event['time']:.3f}s] {event['event']}")
        
        print("="*60)

class OptimizedElevenLabsTTS:
    """Optimized ElevenLabs TTS with real-time streaming and single WebSocket connection"""
    
    def __init__(self, voice_id=None, api_key=None):
        self.voice_id = voice_id or EL_VOICE_ID
        self.api_key = api_key or EL_API_KEY
        self.websocket = None
        self.connected = False
        self.context_id = None
        self.audio_queue = queue.Queue()
        self.player = None
        self.player_thread = None
        
        # Message routing system
        self.message_queue = asyncio.Queue()
        self.receiver_task = None
        self.running = False
        
        # Debug timer
        self.timer = DebugTimer()
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
    
    def start_audio_player(self):
        """Start the audio player thread"""
        if self.player_thread is None or not self.player_thread.is_alive():
            self.player = SimpleAudioPlayer(self.audio_queue)
            self.player_thread = threading.Thread(target=self.player.run, daemon=True)
            self.player_thread.start()
            self.timer.mark("audio_player_started", "Audio player thread started")
    
    async def _message_receiver(self):
        """Single message receiver to avoid concurrency issues"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.message_queue.put(data)
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON: {message}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ Message receiver error: {e}")
        finally:
            self.running = False
    
    async def connect(self):
        """Connect once and keep connection alive"""
        if self.connected:
            return
            
        self.timer.mark("connection_start", "Starting WebSocket connection")
        
        # Connect to multi-stream endpoint with optimized settings
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"
        params = {
            "model_id": "eleven_flash_v2_5",  # Fastest model
            "output_format": "pcm_16000",
            "auto_mode": "true",  # Reduces latency
            "inactivity_timeout": "60",  # Longer timeout
            "sync_alignment": "false"  # Disable alignment for speed
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?{query_string}"
        
        try:
            self.websocket = await websockets.connect(
                full_url,
                extra_headers={"xi-api-key": self.api_key},
                close_timeout=5
            )
            self.timer.mark("websocket_connected", "WebSocket connection established")
            
            # Initialize context with optimized voice settings
            self.context_id = f"conv_{uuid.uuid4().hex[:8]}"
            context_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.3,  # Lower for faster generation
                    "similarity_boost": 0.7,  # Balanced for speed/quality
                    "speed": 1.1  # Slightly faster speech
                },
                "context_id": self.context_id
            }
            
            await self.websocket.send(json.dumps(context_message))
            self.timer.mark("context_initialized", f"Context initialized: {self.context_id}")
            
            # Start message receiver
            self.running = True
            self.receiver_task = asyncio.create_task(self._message_receiver())
            
            self.connected = True
            self.timer.mark("ready_for_speech", "Ready for real-time speech")
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            raise
    
    async def speak_multiple_realtime(self, texts):
        """Send multiple texts with optimized rate limiting"""
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        self.timer.mark("multi_speech_start", f"Starting {len(texts)} phrases with optimized rate limiting")
        
        # Send phrases with optimized flow control
        results = []
        for i, text in enumerate(texts):
            try:
                # Send phrase and wait for first audio before sending next
                audio_task = await self.speak_realtime(text, f"phrase_{i+1}")
                
                # Wait for first audio with shorter timeout
                first_audio_received = False
                start_time = time.time()
                timeout = 2  # Reduced from 3s to 2s
                
                while not first_audio_received and (time.time() - start_time) < timeout:
                    try:
                        # Check if we received first audio
                        if hasattr(audio_task, '_first_audio_received') and audio_task._first_audio_received:
                            first_audio_received = True
                            break
                        
                        # Shorter polling interval
                        await asyncio.sleep(0.05)  # 50ms instead of 100ms
                        
                    except Exception as e:
                        print(f"❌ Error waiting for first audio: {e}")
                        break
                
                if first_audio_received:
                    self.timer.mark(f"first_audio_confirmed_{i+1}", f"First audio confirmed for phrase {i+1}")
                else:
                    self.timer.mark(f"no_first_audio_{i+1}", f"No first audio for phrase {i+1}")
                
                results.append(audio_task)
                
                # Shorter delay between phrases
                if i < len(texts) - 1:  # Don't delay after last phrase
                    await asyncio.sleep(0.1)  # Reduced from 200ms to 100ms
                
            except Exception as e:
                print(f"❌ Error in phrase {i+1}: {e}")
                results.append(None)
        
        self.timer.mark("all_phrases_sent", "All phrases sent with optimized rate limiting")
        
        # Wait for all audio to complete with better timeout handling
        completed_results = []
        for i, task in enumerate(results):
            if task:
                try:
                    result = await task
                    completed_results.append(result)
                except Exception as e:
                    print(f"❌ Error waiting for phrase {i+1}: {e}")
                    completed_results.append(None)
            else:
                completed_results.append(None)
        
        self.timer.mark("multi_speech_complete", f"All phrases completed")
        return completed_results
    
    async def speak_realtime(self, text, phrase_id=""):
        """Send text immediately without waiting for completion (real-time mode)"""
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")
            
        self.timer.mark(f"speech_start_{phrase_id}", f"Starting speech: '{text[:30]}...'")
        
        try:
            # Send text immediately
            message = {
                "text": text + " ",
                "context_id": self.context_id,
                "flush": True
            }
            
            await self.websocket.send(json.dumps(message))
            self.timer.mark(f"text_sent_{phrase_id}", f"Text sent to ElevenLabs")
            
            # Start receiving audio in background (non-blocking)
            audio_task = asyncio.create_task(self._receive_audio_background(phrase_id))
            
            # Add flag to track first audio
            audio_task._first_audio_received = False
            
            return audio_task
            
        except Exception as e:
            print(f"❌ Speech failed: {e}")
            raise
    
    async def _receive_audio_background(self, phrase_id=""):
        """Receive audio in background with optimized handling"""
        audio_chunks_received = 0
        max_chunks = 15  # Increased from 10 to 15
        timeout = 8  # Increased from 5s to 8s for longer phrases
        first_audio_received = False
        last_audio_time = time.time()
        
        try:
            start_time = time.time()
            
            while audio_chunks_received < max_chunks and (time.time() - start_time) < timeout:
                try:
                    # Wait for message with shorter timeout
                    data = await asyncio.wait_for(self.message_queue.get(), timeout=0.05)  # 50ms instead of 100ms
                    
                    if "audio" in data:
                        if not first_audio_received:
                            self.timer.mark(f"first_audio_{phrase_id}", f"First audio chunk received")
                            first_audio_received = True
                            # Mark task as having received first audio
                            if hasattr(self, '_current_task'):
                                self._current_task._first_audio_received = True
                        
                        audio_bytes = base64.b64decode(data["audio"])
                        self.audio_queue.put_nowait(audio_bytes)
                        audio_chunks_received += 1
                        last_audio_time = time.time()  # Update last audio time
                        
                        if data.get("is_final", False):
                            self.timer.mark(f"audio_complete_{phrase_id}", f"Audio complete ({audio_chunks_received} chunks)")
                            break
                    
                except asyncio.TimeoutError:
                    # Check if we've been waiting too long since last audio
                    if (time.time() - last_audio_time) > 2.0:  # 2 second gap
                        self.timer.mark(f"audio_gap_timeout_{phrase_id}", f"Audio gap timeout after 2s")
                        break
                    
                    if (time.time() - start_time) >= timeout:
                        self.timer.mark(f"audio_timeout_{phrase_id}", f"Audio timeout after {timeout}s")
                        break
                    continue
                except Exception as e:
                    print(f"❌ Error processing audio: {e}")
                    break
            
            if audio_chunks_received == 0:
                self.timer.mark(f"no_audio_{phrase_id}", "No audio received")
            
            return audio_chunks_received
                        
        except Exception as e:
            print(f"❌ Error receiving audio: {e}")
            return 0
    
    async def speak_sequential_realtime(self, texts):
        """Send phrases sequentially with optimized spacing"""
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        self.timer.mark("sequential_start", f"Starting {len(texts)} phrases with optimized sequencing")
        
        results = []
        for i, text in enumerate(texts):
            try:
                # Send phrase and get background task
                audio_task = await self.speak_realtime(text, f"seq_{i+1}")
                results.append(audio_task)
                
                # Wait for first audio with optimized timeout
                first_audio_received = False
                start_time = time.time()
                timeout = 1.5  # Reduced from 3s to 1.5s
                
                while not first_audio_received and (time.time() - start_time) < timeout:
                    try:
                        # Check if we received first audio
                        if hasattr(audio_task, '_first_audio_received') and audio_task._first_audio_received:
                            first_audio_received = True
                            break
                        
                        await asyncio.sleep(0.05)  # 50ms polling
                        
                    except Exception as e:
                        print(f"❌ Error waiting for first audio: {e}")
                        break
                
                if first_audio_received:
                    self.timer.mark(f"seq_first_audio_{i+1}", f"First audio for seq {i+1}")
                
                # Shorter delay between phrases
                if i < len(texts) - 1:  # Don't delay after last phrase
                    await asyncio.sleep(0.15)  # Reduced from 300ms to 150ms
                
            except Exception as e:
                print(f"❌ Error in phrase {i+1}: {e}")
                results.append(None)
        
        self.timer.mark("sequential_sent", "All phrases sent with optimized sequencing")
        
        # Wait for all audio to complete
        completed_results = []
        for i, task in enumerate(results):
            if task:
                try:
                    result = await task
                    completed_results.append(result)
                except Exception as e:
                    print(f"❌ Error waiting for phrase {i+1}: {e}")
                    completed_results.append(None)
            else:
                completed_results.append(None)
        
        self.timer.mark("sequential_complete", "Sequential processing complete")
        return completed_results
    
    async def close(self):
        """Close the connection properly"""
        self.timer.mark("closing_start", "Starting cleanup")
        
        # Stop message receiver
        self.running = False
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass
        
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
                self.timer.mark("connection_closed", "WebSocket connection closed")
                
            except Exception as e:
                print(f"❌ Error closing connection: {e}")
            finally:
                self.connected = False
        
        # Stop audio player
        if self.player:
            self.player.stop()
            if self.player_thread:
                self.player_thread.join(timeout=1)
        
        self.timer.mark("cleanup_complete", "All cleanup completed")

class SimpleAudioPlayer:
    """Simple audio player for real-time playback"""
    
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
    """Main function demonstrating real-time TTS with debug timing"""
    
    # Create TTS instance
    tts = OptimizedElevenLabsTTS()
    
    try:
        # Start audio player
        tts.start_audio_player()
        
        # Connect once
        await tts.connect()
        
        # Test single phrase
        print("\n1️⃣ Testing single phrase (real-time)...")
        await tts.speak_realtime("Hello, this is a test of the real-time ElevenLabs TTS system.")
        
        # Test multiple phrases in real-time (no delays)
        print("\n2️⃣ Testing multiple phrases in real-time (no delays)...")
        phrases_multi = [
            "First phrase: Welcome to our real-time system.",
            "Second phrase: This demonstrates instant streaming.",
            "Third phrase: No delays between phrases.",
            "Fourth phrase: Perfect for live conversation."
        ]
        
        await tts.speak_multiple_realtime(phrases_multi)
        
        # Test sequential real-time
        print("\n3️⃣ Testing sequential real-time (minimal delays)...")
        phrases_seq = [
            "Sequential test one: This is the first sequential phrase.",
            "Sequential test two: This is the second sequential phrase.",
            "Sequential test three: This is the third sequential phrase.",
            "Sequential test four: This is the fourth sequential phrase."
        ]
        await tts.speak_sequential_realtime(phrases_seq)
        
        print("\n✅ All real-time tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await tts.close()
        
        # Print debug timing summary
        tts.timer.summary()

if __name__ == "__main__":
    asyncio.run(main()) 