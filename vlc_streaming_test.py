#!/usr/bin/env python3
"""
VLC WebSocket Streaming Test for ElevenLabs TTS
Uses VLC for professional audio playback with built-in optimization
"""

import os
import asyncio
import json
import base64
import websockets
import subprocess
import tempfile
import time
import platform
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VLCStreamingTest:
    def __init__(self):
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.MODEL_ID = "eleven_multilingual_v2"
        
        # VLC settings
        self.vlc_process = None
        self.temp_dir = tempfile.mkdtemp(prefix="vlc_stream_")
        self.output_file = os.path.join(self.temp_dir, "vlc_streaming_output.mp3")
        
        # Streaming settings
        self.buffer_size = 5  # Number of chunks to buffer before starting VLC
        self.audio_chunks = []
        self.vlc_started = False
        
        # Test text (longer Russian text)
        self.test_text = """
Так почему же вы здесь?
     - А где же мне быть? Где же мне работать, по-твоему? В школе? Что я там
буду воровать, промокашки?!  Устраиваясь на работу, ты  должен  прежде всего
задуматься: что, где и как? Что я смогу украсть? 
"""
        
        print(f"📁 Temporary directory: {self.temp_dir}")
        print(f"🖥️ Platform: {platform.system()} {platform.release()}")
    
    def find_vlc_path(self):
        """Find VLC installation path on Windows"""
        if platform.system() == "Windows":
            # Common VLC installation paths on Windows
            possible_paths = [
                r"C:\Program Files\VideoLAN\VLC\vlc.exe",
                r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
                r"C:\Program Files\VLC\vlc.exe",
                r"C:\Program Files (x86)\VLC\vlc.exe"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"✅ Found VLC at: {path}")
                    return path
            
            # Try to find in PATH
            try:
                result = subprocess.run(['where', 'vlc'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    vlc_path = result.stdout.strip().split('\n')[0]
                    print(f"✅ Found VLC in PATH: {vlc_path}")
                    return vlc_path
            except:
                pass
            
            print("❌ VLC not found in common locations or PATH")
            print("   Please install VLC from: https://www.videolan.org/vlc/")
            return None
        else:
            # On Unix-like systems, just use 'vlc'
            return 'vlc'
    
    def check_vlc_installation(self):
        """Check if VLC is installed and accessible"""
        vlc_path = self.find_vlc_path()
        if not vlc_path:
            return False
        
        try:
            # Test VLC with a simple command
            result = subprocess.run([vlc_path, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✅ VLC version: {version_line}")
                self.vlc_path = vlc_path
                return True
            else:
                print("❌ VLC not working properly")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"❌ VLC test failed: {e}")
            return False
    
    def start_vlc_streaming(self):
        """Start VLC with optimized settings for streaming audio"""
        try:
            # Create a named pipe or use a different approach for Windows
            if platform.system() == "Windows":
                # On Windows, we'll use a temporary file approach instead of stdin
                self.temp_audio_file = os.path.join(self.temp_dir, "stream_audio.mp3")
                
                # VLC command with Windows-compatible settings
                vlc_command = [
                    self.vlc_path,
                    '--intf', 'dummy',           # No GUI interface
                    '--no-video',                # Audio only
                    '--no-qt-privacy-ask',       # Skip privacy dialog
                    '--no-qt-updates-notif',     # Skip update notifications
                    '--audio-filter', 'normalizer',  # Audio optimization (simplified)
                    '--gain', '0',               # Normal volume
                    '--network-caching', '1000', # Network buffering (1 second)
                    '--live-caching', '1000',    # Live stream buffering
                    '--clock-jitter', '0',       # Minimize jitter
                    '--no-audio-time-stretch',   # Disable time stretching
                    '--audio-channels', '1',     # Mono output
                    '--audio-sample-rate', '44100', # 44.1kHz sample rate
                    self.temp_audio_file         # Play the temporary file
                ]
            else:
                # Unix/Linux approach with stdin
                vlc_command = [
                    self.vlc_path,
                    '--intf', 'dummy',
                    '--no-video',
                    '--audio-filter', 'normalizer',
                    '--gain', '0',
                    '--network-caching', '1000',
                    '--live-caching', '1000',
                    '--clock-jitter', '0',
                    '--no-audio-time-stretch',
                    '--audio-channels', '1',
                    '--audio-sample-rate', '44100',
                    'pipe:///dev/stdin'
                ]
            
            print("🎵 Starting VLC with optimized audio settings...")
            print("🔧 Audio filter: normalizer")
            print("🔧 Buffer settings: 1000ms network/live caching")
            print("🔧 Audio quality: 44.1kHz, mono")
            
            if platform.system() == "Windows":
                print(f"📁 Using temporary file: {self.temp_audio_file}")
                # On Windows, we'll write to file and let VLC monitor it
                self.vlc_process = subprocess.Popen(
                    vlc_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Unix approach with stdin
                self.vlc_process = subprocess.Popen(
                    vlc_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    bufsize=0
                )
            
            print("✅ VLC started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start VLC: {e}")
            return False
    
    def stop_vlc(self):
        """Stop VLC process"""
        if self.vlc_process:
            try:
                self.vlc_process.terminate()
                self.vlc_process.wait(timeout=5)
                print("✅ VLC stopped")
            except subprocess.TimeoutExpired:
                self.vlc_process.kill()
                print("⚠️ VLC force-killed")
            except Exception as e:
                print(f"⚠️ Error stopping VLC: {e}")
    
    def write_chunk_to_vlc(self, mp3_data):
        """Write MP3 chunk to VLC (Windows: file, Unix: stdin)"""
        try:
            if platform.system() == "Windows":
                # On Windows, append to the temporary file
                if hasattr(self, 'temp_audio_file'):
                    with open(self.temp_audio_file, 'ab') as f:
                        f.write(mp3_data)
                    return True
                else:
                    print("⚠️ Temporary audio file not created")
                    return False
            else:
                # On Unix, write to stdin
                if self.vlc_process and self.vlc_process.poll() is None:
                    self.vlc_process.stdin.write(mp3_data)
                    self.vlc_process.stdin.flush()
                    return True
                else:
                    print("⚠️ VLC process not running")
                    return False
        except Exception as e:
            print(f"❌ Error writing to VLC: {e}")
            return False
    
    async def stream_text_to_speech_vlc(self):
        """Stream text to speech using ElevenLabs WebSocket + VLC"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return
        
        # Check VLC installation
        if not self.check_vlc_installation():
            return
        
        print("🎵 Starting VLC WebSocket Streaming Test...")
        print("=" * 60)
        print("Text to stream:")
        print(self.test_text[:100] + "..." if len(self.test_text) > 100 else self.test_text)
        print("=" * 60)
        
        try:
            # WebSocket connection
            uri = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                f"?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            )
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            
            print(f"🔗 Connecting to: {uri}")
            print("⏳ Establishing WebSocket connection...")
            
            connection_start = time.time()
            async with websockets.connect(uri, extra_headers=headers) as websocket:
                connection_time = time.time() - connection_start
                print(f"✅ WebSocket connected (latency: {connection_time:.3f}s)")
                
                # Send initialization
                init_message = {
                    "text": " ",
                    "voice_settings": {
                        "speed": 1,
                        "stability": 0.5,
                        "similarity_boost": 0.8
                    },
                    "xi_api_key": self.ELEVENLABS_API_KEY
                }
                await websocket.send(json.dumps(init_message))
                print("📤 Sent initialization message")
                
                # Send text in chunks for streaming
                text_chunks = self._split_text_into_chunks(self.test_text, max_chunk_size=100)
                print(f"📝 Sending {len(text_chunks)} text chunks...")
                
                for i, chunk in enumerate(text_chunks):
                    text_message = {
                        "text": chunk,
                        "try_trigger_generation": True
                    }
                    await websocket.send(json.dumps(text_message))
                    print(f" Sent chunk {i+1}/{len(text_chunks)}: '{chunk[:30]}...'")
                    await asyncio.sleep(0.1)  # Small delay between chunks
                
                # Send end marker
                await websocket.send(json.dumps({"text": ""}))
                print(" Sent end marker")
                
                # Start VLC streaming
                print(" Starting VLC audio streaming...")
                await self._vlc_streaming_audio(websocket)
                
        except Exception as e:
            print(f"❌ WebSocket streaming error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_vlc()
    
    def _split_text_into_chunks(self, text, max_chunk_size=100):
        """Split text into chunks for streaming"""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk + " " + word) <= max_chunk_size:
                current_chunk += (" " + word) if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _vlc_streaming_audio(self, websocket):
        """Stream audio to VLC with professional buffering"""
        try:
            # Track streaming metrics
            first_audio_time = None
            total_audio_chunks = 0
            total_audio_bytes = 0
            streaming_start = time.time()
            
            # Initialize temporary file for Windows
            if platform.system() == "Windows" and hasattr(self, 'temp_audio_file'):
                # Create empty file to start with
                with open(self.temp_audio_file, 'wb') as f:
                    pass  # Create empty file
            
            # Receive and buffer audio chunks
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "audio" in data and data["audio"]:
                        # Track first audio timing
                        if first_audio_time is None:
                            first_audio_time = time.time()
                            time_to_first_audio = first_audio_time - streaming_start
                            print(f"🎵 First audio received (latency: {time_to_first_audio:.3f}s)")
                        
                        # Decode audio
                        audio_data = base64.b64decode(data["audio"])
                        total_audio_chunks += 1
                        total_audio_bytes += len(audio_data)
                        
                        # Add to buffer
                        self.audio_chunks.append(audio_data)
                        
                        # Start VLC when buffer is ready
                        if not self.vlc_started and len(self.audio_chunks) >= self.buffer_size:
                            if self.start_vlc_streaming():
                                self.vlc_started = True
                                print(f"🎵 VLC started (buffer: {len(self.audio_chunks)} chunks)")
                                
                                # Send all buffered chunks to VLC
                                for chunk in self.audio_chunks:
                                    self.write_chunk_to_vlc(chunk)
                                self.audio_chunks.clear()  # Clear buffer after sending
                        
                        # Send chunk to VLC if already started
                        elif self.vlc_started:
                            self.write_chunk_to_vlc(audio_data)
                        
                        print(f"🎵 Buffer: {len(self.audio_chunks)} chunks, VLC: {'Running' if self.vlc_started else 'Waiting'}")
                    
                    elif "audio" in data and data["audio"] is None:
                        print("📡 End of stream signal received")
                    
                    if data.get("isFinal"):
                        print("✅ Stream completed")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
                    break
            
            # Wait for VLC to finish playing
            if self.vlc_started:
                print("⏳ Waiting for VLC to finish playing...")
                await asyncio.sleep(3)  # Give VLC more time to process remaining audio
            
            # Calculate streaming metrics
            streaming_end = time.time()
            total_streaming_time = streaming_end - streaming_start
            
            print("\n" + "=" * 60)
            print("🎵 VLC STREAMING METRICS:")
            print("=" * 60)
            print(f"Total streaming time: {total_streaming_time:.3f}s")
            if first_audio_time:
                print(f"Time to first audio: {time_to_first_audio:.3f}s")
            print(f"Total audio chunks: {total_audio_chunks}")
            print(f"Total audio bytes: {total_audio_bytes}")
            if total_audio_chunks > 0:
                print(f"Average chunk size: {total_audio_bytes/total_audio_chunks:.0f} bytes")
            print(f"Streaming rate: {total_audio_bytes/total_streaming_time:.0f} bytes/s")
            print(f"Buffer size used: {self.buffer_size} chunks")
            print(f"VLC audio filters: normalizer")
            print(f"Platform: {platform.system()}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ VLC streaming error: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main test function"""
    test = VLCStreamingTest()
    await test.stream_text_to_speech_vlc()

if __name__ == "__main__":
    asyncio.run(main())