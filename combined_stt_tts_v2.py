#!/usr/bin/env python3
"""
Combined Speechmatics STT + ElevenLabs TTS - Version 2
Simple and reliable parallel execution
"""

import asyncio
import base64
import json
import os
import sys
import time
import threading
import queue
import uuid
import websockets
import pyaudio
from dotenv import load_dotenv
from typing import AsyncGenerator

# Import Speechmatics models
from speechmatics.models import (
    AudioSettings,
    TranscriptionConfig,
)

# Load environment variables
load_dotenv()

# Configuration
SPEECHMATICS_API_KEY = os.getenv('SPEECHMATICS_API_KEY')
EL_API_KEY = os.getenv("ELEVENLABS_API_KEY")
EL_VOICE_ID = os.getenv("EL_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

if not SPEECHMATICS_API_KEY:
    print("ERROR: SPEECHMATICS_API_KEY not found in .env file")
    sys.exit(1)

if not EL_API_KEY:
    print("ERROR: ELEVENLABS_API_KEY not found in .env file")
    sys.exit(1)

# WebSocket URLs
SPEECHMATICS_URL = "wss://eu2.rt.speechmatics.com/v2"

# Audio configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Speechmatics audio chunking
CHUNK_DURATION_MS = 100
BYTES_PER_SAMPLE = 2
SAMPLES_PER_CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)
CHUNK_BYTES = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE

# Global state for transcript management
final_transcript = ""
current_partial = ""
audio_chunks_sent = 0
last_display_text = ""
last_tts_text = ""  # Track last text sent to TTS
last_final_transcript = ""  # Track last final transcript sent to TTS
pending_tts_text = ""  # Accumulate text for TTS
last_tts_time = 0  # Track when last TTS was sent
tts_sent_length = 0  # Track how much text we've already sent to TTS

# Sentence-based chunking buffer
text_buffer = ""  # Buffer for accumulating text
last_buffer_update = 0  # Track when buffer was last updated

# Sequential TTS queue
tts_queue = []  # Queue of chunks to send in order
tts_processing = False  # Flag to prevent concurrent processing

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

# Global debug timer
debug_timer = DebugTimer()

class TTSAudioPlayer:
    """Dedicated audio player for TTS output"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = True
        self.thread = None
    
    def start(self):
        """Start the audio player thread"""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        debug_timer.mark("tts_player_started", "TTS Audio player started")
    
    def _run(self):
        """Audio player thread function"""
        try:
            print(f"🔊 TTS Player: Initializing with format={FORMAT}, channels={CHANNELS}, rate={RATE}, chunk_size={CHUNK_SIZE}")
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            print(f"🔊 TTS Player: Stream initialized successfully")
            
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    if chunk:
                        print(f"🔊 TTS Player: Playing chunk: {len(chunk)} bytes")
                        print(f"🔊 TTS Player: Chunk sample: {chunk[:20].hex()}")
                        self.stream.write(chunk)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ TTS Audio player error: {e}")
                    print(f"❌ TTS Audio player error type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    break
                    
        except Exception as e:
            print(f"❌ Failed to initialize TTS audio player: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()
    
    def play(self, audio_data):
        """Add audio data to the queue"""
        self.audio_queue.put(audio_data)
    
    def stop(self):
        """Stop the audio player"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

class ElevenLabsTTS:
    """Simple ElevenLabs TTS client"""
    
    def __init__(self, audio_player):
        self.voice_id = EL_VOICE_ID
        self.api_key = EL_API_KEY
        self.websocket = None
        self.audio_player = audio_player
        self.connected = False
        self.message_queue = asyncio.Queue()
        self.receiver_task = None
        self.running = False
    
    async def connect(self):
        """Connect to ElevenLabs"""
        try:
            # Connect with proper PCM output format
            url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"
            params = {
                "model_id": "eleven_flash_v2_5",
                "output_format": "pcm_16000",  # Force PCM output
                "auto_mode": "true",
                "inactivity_timeout": "60",
                "sync_alignment": "false"
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{url}?{query_string}"
            
            self.websocket = await websockets.connect(
                full_url,
                extra_headers={"xi-api-key": self.api_key}
            )
            debug_timer.mark("tts_connected", "TTS connected to ElevenLabs")
            
            # Initialize context with optimized settings
            self.context_id = f"conv_{uuid.uuid4().hex[:8]}"
            init_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.3,
                    "similarity_boost": 0.7,
                    "speed": 1.1
                },
                "context_id": self.context_id
            }
            
            await self.websocket.send(json.dumps(init_message))
            self.connected = True
            debug_timer.mark("tts_initialized", "TTS context initialized")
            
            # Start message receiver
            self.running = True
            self.receiver_task = asyncio.create_task(self._message_receiver())
            
        except Exception as e:
            print(f"❌ TTS: Failed to connect: {e}")
            raise
    
    async def _message_receiver(self):
        """Background task to receive and route WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    print(f"🔊 TTS Receiver: Message keys: {list(data.keys())}")
                    if "audio" in data:
                        print(f"🔊 TTS Receiver: Audio message length: {len(data['audio'])} chars")
                    await self.message_queue.put(data)
                except Exception as e:
                    print(f"❌ TTS: Message receiver error: {e}")
        except Exception as e:
            print(f"❌ TTS: Message receiver error: {e}")
        finally:
            self.running = False
    
    async def speak(self, text):
        """Speak a text phrase"""
        if not self.connected:
            return
        
        debug_timer.mark("tts_speak_start", f"TTS speaking: {text[:50]}...")
        
        # Send text
        text_message = {"text": text, "xi_api_key": self.api_key}
        await self.websocket.send(json.dumps(text_message))
        debug_timer.mark("tts_text_sent", "TTS text sent to ElevenLabs")
        
        # Receive audio using message queue
        try:
            audio_chunks_received = 0
            max_chunks = 15
            timeout = 8
            first_audio_received = False
            last_audio_time = time.time()
            start_time = time.time()
            
            while audio_chunks_received < max_chunks and (time.time() - start_time) < timeout:
                try:
                    data = await asyncio.wait_for(self.message_queue.get(), timeout=0.05)
                    
                    if "audio" in data:
                        audio_bytes = base64.b64decode(data["audio"])
                        
                        if not first_audio_received:
                            debug_timer.mark("tts_first_audio", "TTS first audio chunk received")
                            first_audio_received = True
                        
                        # Send raw audio bytes to player (no processing)
                        self.audio_player.play(audio_bytes)
                        audio_chunks_received += 1
                        last_audio_time = time.time()
                        
                        if data.get("is_final", False):
                            print(f"🔊 TTS: Audio complete ({audio_chunks_received} chunks)")
                            break
                    
                except asyncio.TimeoutError:
                    if (time.time() - last_audio_time) > 2.0:
                        print(f"🔊 TTS: Audio gap timeout after 2s")
                        break
                    
                    if (time.time() - start_time) >= timeout:
                        print(f"🔊 TTS: Audio timeout after {timeout}s")
                        break
                    continue
                except Exception as e:
                    print(f"❌ TTS: Error processing audio: {e}")
                    break
            
            if audio_chunks_received == 0:
                print(f"🔊 TTS: No audio received")
                    
        except Exception as e:
            print(f"❌ TTS: Error receiving audio: {e}")
    
    async def close(self):
        """Close the connection"""
        self.running = False
        
        if self.receiver_task:
            self.receiver_task.cancel()
            try:
                await self.receiver_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()

def print_transcript():
    """Print current transcript"""
    global final_transcript, current_partial, last_display_text
    display_text = final_transcript + current_partial
    
    if display_text != last_display_text:
        print(f"🎤 STT: {display_text}")
        last_display_text = display_text

async def send_to_tts_if_new(tts, full_text):
    """Sentence-based chunking with timeout fallback"""
    global text_buffer, last_buffer_update, last_tts_text, tts_sent_length
    
    # Update buffer with new text
    new_text = full_text[tts_sent_length:].strip()
    if new_text:
        # Ensure proper spacing between words
        if text_buffer and not text_buffer.endswith(' '):
            text_buffer += ' '
        text_buffer += new_text + " "
        last_buffer_update = time.time()
        debug_timer.mark("buffer_updated", f"Buffer updated: {len(text_buffer)} chars")
    
    # Process buffer for complete sentences
    await process_text_buffer(tts)
    
    # Update tracking
    tts_sent_length = len(full_text)
    last_tts_text = full_text

async def process_text_buffer(tts):
    """Process text buffer using sentence-based chunking with sequential queue"""
    global text_buffer, last_buffer_update, last_tts_time, tts_queue, tts_processing
    
    if not text_buffer.strip():
        return
    
    # Split on sentence endings (., !, ?) for more natural breaks
    sentences = []
    current_sentence = ""
    
    for char in text_buffer:
        current_sentence += char
        if char in '.!?':
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add any remaining text as incomplete sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # Add complete sentences to queue (all except the last one)
    sentences_to_queue = sentences[:-1] if len(sentences) > 1 else []
    
    for sentence in sentences_to_queue:
        if sentence and len(sentence.split()) >= 4:  # At least 4 words for sentences
            clean_sentence = sentence.lstrip('.,!?:; ')
            if clean_sentence and len(clean_sentence.split()) >= 4:  # Double check after cleaning
                tts_queue.append(clean_sentence)
                debug_timer.mark("tts_queued", f"Queued sentence ({len(clean_sentence.split())} words): {clean_sentence[:50]}...")
    
    # Update buffer to contain only the last sentence
    text_buffer = sentences[-1] if sentences else ""
    
    # If we have a very long incomplete sentence, add it to queue too
    if text_buffer and len(text_buffer.split()) >= 8:  # If buffer has 8+ words
        clean_buffer = text_buffer.lstrip('.,!?:; ')
        if clean_buffer and len(clean_buffer.split()) >= 8:
            tts_queue.append(clean_buffer)
            debug_timer.mark("tts_queued", f"Queued long sentence ({len(clean_buffer.split())} words): {clean_buffer[:50]}...")
            text_buffer = ""  # Clear buffer after queuing
    
    # Timeout fallback: if buffer has been waiting too long, add it to queue
    if text_buffer and (time.time() - last_buffer_update) > 3.0:  # 3.0 second timeout (longer)
        if len(text_buffer.split()) >= 6:  # At least 6 words for timeout
            clean_buffer = text_buffer.lstrip('.,!?:; ')
            if clean_buffer and len(clean_buffer.split()) >= 6:  # Double check after cleaning
                tts_queue.append(clean_buffer)
                debug_timer.mark("tts_queued", f"Queued timeout sentence ({len(clean_buffer.split())} words): {clean_buffer[:50]}...")
                text_buffer = ""  # Clear buffer after queuing
    
    # Process queue if not already processing
    if tts_queue and not tts_processing:
        await process_tts_queue(tts)

async def process_tts_queue(tts):
    """Process TTS queue sequentially"""
    global tts_queue, tts_processing, last_tts_time
    
    if tts_processing:
        return
    
    tts_processing = True
    
    try:
        while tts_queue:
            chunk = tts_queue.pop(0)  # Get first chunk from queue
            debug_timer.mark("tts_sending", f"Sending queued chunk ({len(chunk.split())} words): {chunk[:50]}...")
            await tts.speak(chunk)
            last_tts_time = time.time()
            debug_timer.mark("tts_sent", f"Queued chunk sent to TTS ({len(chunk)} chars)")
            
            # Small delay between chunks for natural flow
            await asyncio.sleep(0.1)
    
    finally:
        tts_processing = False

async def mic_stream_generator() -> AsyncGenerator[bytes, None]:
    """Generate audio chunks from microphone"""
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=SAMPLES_PER_CHUNK
        )
        
        print("🎤 STT: Microphone started. Speak now...")
        
        while True:
            try:
                audio_data = stream.read(SAMPLES_PER_CHUNK, exception_on_overflow=False)
                
                if len(audio_data) % BYTES_PER_SAMPLE != 0:
                    padding_needed = BYTES_PER_SAMPLE - (len(audio_data) % BYTES_PER_SAMPLE)
                    audio_data += b'\x00' * padding_needed
                
                yield audio_data
                await asyncio.sleep(CHUNK_DURATION_MS / 1000)
                
            except Exception as e:
                print(f"❌ STT: Microphone error: {e}")
                break
                
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

async def speechmatics_handler(websocket, tts=None):
    """Handle Speechmatics messages"""
    global final_transcript, current_partial
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("message")
                
                if message_type == "RecognitionStarted":
                    session_id = data.get("id", "unknown")
                    debug_timer.mark("stt_recognition_started", f"STT recognition started (ID: {session_id})")
                    
                elif message_type == "AddTranscript":
                    transcript_segment = data.get("metadata", {}).get("transcript", "")
                    final_transcript += transcript_segment
                    current_partial = ""
                    print_transcript()
                    
                    # Send to TTS if we have accumulated substantial text
                    if tts:
                        # Get the full accumulated text since last TTS
                        full_text = final_transcript + current_partial
                        if len(full_text.strip().split()) >= 4:  # At least 4 words
                            await send_to_tts_if_new(tts, full_text)
                    
                elif message_type == "AddPartialTranscript":
                    full_partial = data.get("metadata", {}).get("transcript", "")
                    
                    if full_partial.startswith(final_transcript):
                        new_partial = full_partial[len(final_transcript):]
                        if new_partial != current_partial:
                            current_partial = new_partial
                            print_transcript()
                    else:
                        if full_partial != current_partial:
                            current_partial = full_partial
                            print_transcript()
                    
                elif message_type == "EndOfTranscript":
                    print("\n🎤 STT: End of transcript")
                    break
                    
                elif message_type == "Error":
                    error_type = data.get("type", "unknown")
                    error_reason = data.get("reason", "No reason provided")
                    print(f"\n❌ STT: Error - {error_type}: {error_reason}")
                    break
                    
            except json.JSONDecodeError as e:
                print(f"\n❌ STT: Failed to parse message: {e}")
                break
            except Exception as e:
                print(f"\n❌ STT: Message handling error: {e}")
                break
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n❌ STT: Connection closed: {e}")
    except Exception as e:
        print(f"\n❌ STT: Unexpected error: {e}")

async def speechmatics_sender(websocket):
    """Send audio to Speechmatics"""
    global audio_chunks_sent
    
    # Configure audio settings
    audio_settings = AudioSettings(
        encoding="pcm_s16le",
        sample_rate=RATE
    )
    
    # Configure transcription settings
    transcription_config = TranscriptionConfig(
        language="ru",
        max_delay=1,
        max_delay_mode="flexible",
        accuracy="enhanced"
    )
    
    # Start recognition
    start_message = {
        "message": "StartRecognition",
        "audio_format": {
            "type": "raw",
            "encoding": audio_settings.encoding,
            "sample_rate": audio_settings.sample_rate
        },
        "transcription_config": transcription_config.asdict()
    }
    
    try:
        await websocket.send(json.dumps(start_message))
        print("🎤 STT: Recognition started")
        
        # Send audio chunks
        audio_generator = mic_stream_generator()
        async for audio_chunk in audio_generator:
            try:
                await websocket.send(audio_chunk)
                audio_chunks_sent += 1
            except websockets.exceptions.ConnectionClosed:
                print("❌ STT: Connection closed while sending audio")
                break
            except Exception as e:
                print(f"❌ STT: Failed to send audio: {e}")
                break
        
        # End stream
        end_message = {
            "message": "EndOfStream",
            "last_seq_no": audio_chunks_sent
        }
        await websocket.send(json.dumps(end_message))
        print(f"\n🎤 STT: End of stream sent ({audio_chunks_sent} chunks)")
        
    except Exception as e:
        print(f"❌ STT: Sender error: {e}")

async def tts_monitor_task(tts):
    """Monitor TTS system"""
    try:
        debug_timer.mark("tts_monitor_started", "TTS monitor started")
        
        # Keep the task alive to handle TTS requests
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"❌ TTS: Monitor failed: {e}")

async def flush_pending_tts(tts):
    """Flush any remaining text in buffer and queue"""
    global text_buffer, tts_queue
    
    # Add any remaining buffer text to queue
    if text_buffer.strip():
        clean_buffer = text_buffer.lstrip('.,!?:; ')
        if clean_buffer and len(clean_buffer.split()) >= 4:  # At least 4 words for flush
            tts_queue.append(clean_buffer)
            debug_timer.mark("tts_flush", f"Flushing buffer to queue: {clean_buffer[:50]}...")
            text_buffer = ""  # Clear buffer
    
    # Process any remaining queue items
    if tts_queue:
        await process_tts_queue(tts)
        debug_timer.mark("tts_flushed", "Queue flushed")

async def main():
    """Main function"""
    print("🚀 Starting Combined STT + TTS System v2")
    print("📋 STT: Real-time transcription")
    print("🔊 TTS: Parallel audio output")
    
    # Create TTS audio player
    tts_player = TTSAudioPlayer()
    tts_player.start()
    
    # Create TTS client
    tts = ElevenLabsTTS(tts_player)
    
    try:
        # Connect TTS
        await tts.connect()
        
        # Start TTS monitor task
        tts_task = asyncio.create_task(tts_monitor_task(tts))
        
        # Connect to Speechmatics
        debug_timer.mark("stt_connecting", "Connecting to Speechmatics")
        
        async with websockets.connect(
            SPEECHMATICS_URL,
            extra_headers={"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"},
            ping_interval=30,
            ping_timeout=60
        ) as websocket:
            debug_timer.mark("stt_connected", "STT connected to Speechmatics")
            
            # Run STT tasks
            receiver_task = asyncio.create_task(speechmatics_handler(websocket, tts))
            sender_task = asyncio.create_task(speechmatics_sender(websocket))
            
            # Wait for all tasks
            done, pending = await asyncio.wait(
                [receiver_task, sender_task, tts_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cleanup
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ STT: Connection failed (Status: {e.status_code})")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"❌ STT: Connection closed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        await flush_pending_tts(tts)
        await tts.close()
        tts_player.stop()
        debug_timer.mark("system_shutdown", "System shutdown complete")
        
        # Print debug summary
        debug_timer.summary()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 