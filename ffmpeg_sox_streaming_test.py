#!/usr/bin/env python3
"""
FFmpeg + SoX Real-Time Streaming Test for ElevenLabs TTS
Uses FFmpeg for MP3 decoding and SoX for real-time audio playback
Cross-platform solution with minimal latency
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

class FFmpegSoXStreamingTest:
    def __init__(self):
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.MODEL_ID = "eleven_multilingual_v2"
        
        # FFmpeg + SoX settings
        self.ffmpeg_process = None
        self.sox_process = None
        self.temp_dir = tempfile.mkdtemp(prefix="ffmpeg_sox_stream_")
        
        # Streaming settings
        self.buffer_size = 3  # Number of chunks to buffer before starting playback
        self.audio_chunks = []
        self.playback_started = False
        
        # Test text (Russian text for testing)
        self.test_text = """
Так почему же вы здесь?
     - А где же мне быть? Где же мне работать, по-твоему? В школе? Что я там
буду воровать, промокашки?!  Устраиваясь на работу, ты  должен  прежде всего
задуматься: что, где и как? Что я смогу украсть? Где  я смогу украсть? И как
я  смогу украсть?.. Ты  понял?  Вот и хорошо. Все будет нормально.  К вечеру
бабки появятся.
"""
        
        print(f"📁 Temporary directory: {self.temp_dir}")
        print(f"🖥️ Platform: {platform.system()} {platform.release()}")
    
    def check_dependencies(self):
        """Check if FFmpeg and SoX are installed and accessible"""
        dependencies_ok = True
        
        # Check FFmpeg (required on Windows)
        if platform.system() == "Windows":
            try:
                result = subprocess.run(['ffmpeg', '-version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    print(f"✅ FFmpeg found: {version_line}")
                else:
                    print("❌ FFmpeg not working properly")
                    dependencies_ok = False
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"❌ FFmpeg not found: {e}")
                print("   Please install FFmpeg from: https://ffmpeg.org/download.html")
                dependencies_ok = False
        
        # Check SoX
        try:
            result = subprocess.run(['sox', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✅ SoX found: {version_line}")
            else:
                print("❌ SoX not working properly")
                dependencies_ok = False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"❌ SoX not found: {e}")
            print("   Please install SoX from: https://sox.sourceforge.net/")
            dependencies_ok = False
        
        return dependencies_ok
    
    def start_ffmpeg_sox_streaming(self):
        """Start FFmpeg for real-time audio streaming with direct audio output"""
        try:
            if platform.system() == "Windows":
                # Use FFmpeg's DirectShow output on Windows - much more reliable
                ffmpeg_command = [
                    'ffmpeg',
                    '-hide_banner',
                    '-loglevel', 'error',
                    '-f', 'mp3',           # Input format is MP3
                    '-i', 'pipe:0',        # Read from stdin
                    '-f', 'wav',           # Output format is WAV
                    '-ar', '44100',        # Sample rate 44.1kHz
                    '-ac', '1',            # Mono audio
                    '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Audio normalization
                    '-acodec', 'pcm_s16le', # PCM audio codec
                    '-f', 'wav',           # Force WAV format
                    'pipe:1'               # Output to stdout
                ]
                
                # SoX command: read WAV from stdin, play to default device
                sox_command = [
                    'sox',
                    '-t', 'wav',           # Input type is WAV
                    '-',                   # Read from stdin
                    '-d'                   # Play to default audio device
                ]
                
                print("🎵 Starting FFmpeg + SoX real-time streaming...")
                print("🔧 FFmpeg: MP3 → WAV decoding with normalization")
                print("🔧 SoX: Real-time playback")
                
                # Start FFmpeg process
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0  # Unbuffered for real-time streaming
                )
                
                # Start SoX process, connected to FFmpeg's output
                self.sox_process = subprocess.Popen(
                    sox_command,
                    stdin=self.ffmpeg_process.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Close FFmpeg's stdout in our process to avoid deadlock
                self.ffmpeg_process.stdout.close()
                
            else:
                # Unix/Linux approach - use SoX directly with MP3
                sox_command = [
                    'sox',
                    '-t', 'mp3',           # Input type is MP3
                    '-',                   # Read from stdin
                    '-d',                  # Play to default audio device
                    'norm',                # Normalize audio levels
                    'compand', '0.3,1', '6:-70,-60,-20', '-5', '-90', '0.2'  # Compression
                ]
                
                print("🎵 Starting SoX real-time streaming...")
                print("🔧 SoX: MP3 → Direct playback with normalization & compression")
                
                # Start SoX process directly
                self.sox_process = subprocess.Popen(
                    sox_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    bufsize=0  # Unbuffered for real-time streaming
                )
                
                # Set FFmpeg process to None since we're not using it
                self.ffmpeg_process = None
            
            print("✅ Audio pipeline started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start audio pipeline: {e}")
            return False
    
    def stop_ffmpeg_sox(self):
        """Stop FFmpeg and SoX processes"""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
                print("✅ FFmpeg stopped")
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
                print("⚠️ FFmpeg force-killed")
            except Exception as e:
                print(f"⚠️ Error stopping FFmpeg: {e}")
        
        if self.sox_process:
            try:
                self.sox_process.terminate()
                self.sox_process.wait(timeout=5)
                print("✅ SoX stopped")
            except subprocess.TimeoutExpired:
                self.sox_process.kill()
                print("⚠️ SoX force-killed")
            except Exception as e:
                print(f"⚠️ Error stopping SoX: {e}")
    
    def write_chunk_to_ffmpeg(self, mp3_data):
        """Write MP3 chunk to the audio pipeline for real-time streaming"""
        try:
            if platform.system() == "Windows":
                # On Windows, write to FFmpeg
                if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                    self.ffmpeg_process.stdin.write(mp3_data)
                    self.ffmpeg_process.stdin.flush()
                    return True
                else:
                    print("⚠️ FFmpeg process not running")
                    return False
            else:
                # On Unix, write to SoX
                if self.sox_process and self.sox_process.poll() is None:
                    self.sox_process.stdin.write(mp3_data)
                    self.sox_process.stdin.flush()
                    return True
                else:
                    print("⚠️ SoX process not running")
                    return False
        except Exception as e:
            print(f"❌ Error writing to audio pipeline: {e}")
            return False
    
    async def stream_text_to_speech_ffmpeg_sox(self):
        """Stream text to speech using ElevenLabs WebSocket + FFmpeg + SoX"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return
        
        # Check dependencies
        if not self.check_dependencies():
            return
        
        print("🎵 Starting FFmpeg + SoX WebSocket Streaming Test...")
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
                    print(f"📤 Sent chunk {i+1}/{len(text_chunks)}: '{chunk[:30]}...'")
                    await asyncio.sleep(0.1)  # Small delay between chunks
                
                # Send end marker
                await websocket.send(json.dumps({"text": ""}))
                print("📤 Sent end marker")
                
                # Start FFmpeg + SoX streaming
                print("🎵 Starting audio streaming...")
                await self._sox_streaming_audio(websocket)
                
        except Exception as e:
            print(f"❌ WebSocket streaming error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_ffmpeg_sox()
    
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
    
    async def _sox_streaming_audio(self, websocket):
        """Stream audio to SoX with real-time playback"""
        try:
            # Track streaming metrics
            first_audio_time = None
            total_audio_chunks = 0
            total_audio_bytes = 0
            streaming_start = time.time()
            
            # Receive and stream audio chunks
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
                        
                        # Start FFmpeg + SoX when buffer is ready
                        if not self.playback_started and len(self.audio_chunks) >= self.buffer_size:
                            if self.start_ffmpeg_sox_streaming():
                                self.playback_started = True
                                print(f"🎵 Audio pipeline started (buffer: {len(self.audio_chunks)} chunks)")
                                
                                # Send all buffered chunks to the pipeline
                                for chunk in self.audio_chunks:
                                    self.write_chunk_to_ffmpeg(chunk)
                                self.audio_chunks.clear()  # Clear buffer after sending
                        
                        # Send chunk to the pipeline if already started
                        elif self.playback_started:
                            self.write_chunk_to_ffmpeg(audio_data)
                        
                        print(f"🎵 Buffer: {len(self.audio_chunks)} chunks, Playback: {'Running' if self.playback_started else 'Waiting'}")
                    
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
            
            # Wait for audio to finish playing
            if self.playback_started:
                print("⏳ Waiting for audio to finish playing...")
                await asyncio.sleep(2)  # Give SoX time to process remaining audio
            
            # Calculate streaming metrics
            streaming_end = time.time()
            total_streaming_time = streaming_end - streaming_start
            
            print("\n" + "=" * 60)
            print("🎵 SOX STREAMING METRICS:")
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
            print(f"Audio processing: FFmpeg (MP3 → WAV with normalization) + SoX (Real-time playback)")
            print(f"Platform: {platform.system()}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ SoX streaming error: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main test function"""
    test = FFmpegSoXStreamingTest()
    await test.stream_text_to_speech_ffmpeg_sox()

if __name__ == "__main__":
    asyncio.run(main()) 