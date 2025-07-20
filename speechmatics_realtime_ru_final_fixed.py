#!/usr/bin/env python3
"""
Real-time Speechmatics transcription for Russian language - FINAL FIXED VERSION
Requirements:
- max_delay = 1 second
- max_delay_mode = flexible  
- accuracy = enhanced
- Russian language
"""

import asyncio
import json
import os
import sys
import time
import shutil
from typing import AsyncGenerator
import pyaudio
import websockets
from dotenv import load_dotenv

# Import Speechmatics models for better type safety and validation
from speechmatics.models import (
    AudioSettings,
    TranscriptionConfig,
)

# Load environment variables
load_dotenv()

# Configuration
SPEECHMATICS_API_KEY = os.getenv('SPEECHMATICS_API_KEY')
if not SPEECHMATICS_API_KEY:
    print("ERROR: SPEECHMATICS_API_KEY not found in .env file")
    sys.exit(1)

# WebSocket URL for Speechmatics Real-Time SaaS (correct from official docs)
CONNECTION_URL = "wss://eu2.rt.speechmatics.com/v2"

# Microphone stream configuration
CHUNK_SIZE = 1024 * 2  # 2048 samples
FORMAT = pyaudio.paInt16  # Corresponds to pcm_s16le
CHANNELS = 1
SAMPLE_RATE = 16000  # 16kHz for optimal performance

# Audio chunking parameters
CHUNK_DURATION_MS = 100  # 100ms chunks for real-time feel
BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
CHUNK_BYTES = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE

# Global state for transcript management
final_transcript = ""
current_partial = ""
audio_chunks_sent = 0  # Track sequence numbers for EndOfStream
last_display_text = ""  # Track the last displayed text to avoid repetition

def print_current_transcript():
    """Print the current transcript only if it has changed"""
    global final_transcript, current_partial, last_display_text
    display_text = final_transcript + current_partial
    
    # Only print if the text has actually changed
    if display_text != last_display_text:
        # Clear the line and print new text
        print(f"\r{display_text}", end="", flush=True)
        last_display_text = display_text

async def mic_stream_generator() -> AsyncGenerator[bytes, None]:
    """
    Asynchronous generator that yields audio chunks from the microphone
    with proper real-time timing to prevent server buffer overflow
    """
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=SAMPLES_PER_CHUNK
        )
        
        print("INFO: Microphone stream started. Speak now...")
        
        while True:
            try:
                # Read audio chunk
                audio_data = stream.read(SAMPLES_PER_CHUNK, exception_on_overflow=False)
                
                # Ensure chunk size is correct (multiple of bytes per sample)
                if len(audio_data) % BYTES_PER_SAMPLE != 0:
                    # Pad with zeros if needed
                    padding_needed = BYTES_PER_SAMPLE - (len(audio_data) % BYTES_PER_SAMPLE)
                    audio_data += b'\x00' * padding_needed
                
                yield audio_data
                
                # Sleep to maintain real-time rate (100ms chunk every 100ms)
                await asyncio.sleep(CHUNK_DURATION_MS / 1000.0)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nERROR: Audio capture error: {e}")
                break
                
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

async def receive_handler(websocket):
    """
    Handle incoming messages from the Speechmatics server
    Manages transcript volatility and error handling
    """
    global final_transcript, current_partial
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("message")
                
                if message_type == "RecognitionStarted":
                    session_id = data.get("id", "unknown")
                    print(f"\nINFO: Recognition started with session ID: {session_id}")
                    
                elif message_type == "AddTranscript":
                    # Final transcript - this contains the complete transcript for this segment
                    transcript_segment = data.get("metadata", {}).get("transcript", "")
                    final_transcript += transcript_segment
                    current_partial = ""  # Clear partial as it's now finalized
                    print_current_transcript()
                    
                elif message_type == "AddPartialTranscript":
                    # Partial transcript - this contains the complete transcript so far
                    # We need to extract only the new partial part
                    full_partial = data.get("metadata", {}).get("transcript", "")
                    
                    # Calculate the new partial part by removing what's already finalized
                    if full_partial.startswith(final_transcript):
                        new_partial = full_partial[len(final_transcript):]
                        # Only update if the partial has actually changed
                        if new_partial != current_partial:
                            current_partial = new_partial
                            print_current_transcript()
                    else:
                        # Fallback: use the full partial if we can't determine the new part
                        if full_partial != current_partial:
                            current_partial = full_partial
                            print_current_transcript()
                    
                elif message_type == "AudioAdded":
                    # Server confirmed audio chunk received
                    seq_no = data.get("seq_no", 0)
                    # Optional: track sequence numbers for debugging
                    
                elif message_type == "EndOfTranscript":
                    print("\nINFO: End of transcript received.")
                    break
                    
                elif message_type == "Error":
                    error_type = data.get("type", "unknown")
                    error_reason = data.get("reason", "No reason provided")
                    print(f"\nERROR: Server error - Type: {error_type}, Reason: {error_reason}")
                    
                    # Handle specific error types from official documentation
                    if error_type in ["quota_exceeded", "job_error", "internal_error"]:
                        print("INFO: Retryable error detected. Consider retrying in 5-10 seconds.")
                    elif error_type in ["buffer_error", "data_error"]:
                        print("WARNING: Audio being sent too quickly. Consider reducing chunk rate.")
                    elif error_type.startswith("invalid_"):
                        print("ERROR: Configuration error. Check your settings.")
                    elif error_type == "not_authorised":
                        print("ERROR: Authentication failed. Check your API key.")
                    elif error_type == "insufficient_funds":
                        print("ERROR: Insufficient credits in your account.")
                    elif error_type == "not_allowed":
                        print("ERROR: Action not allowed with your current permissions.")
                    elif error_type == "timelimit_exceeded":
                        print("ERROR: Usage quota exceeded for your contract.")
                    elif error_type == "idle_timeout":
                        print("ERROR: Session timed out due to inactivity (1 hour limit).")
                    elif error_type == "session_timeout":
                        print("ERROR: Session reached maximum duration (48 hours).")
                    
                    break
                    
                elif message_type == "Warning":
                    warning_type = data.get("type", "unknown")
                    warning_reason = data.get("reason", "No reason provided")
                    print(f"\nWARNING: {warning_type} - {warning_reason}")
                    
                elif message_type == "Info":
                    info_type = data.get("type", "unknown")
                    info_reason = data.get("reason", "No reason provided")
                    print(f"\nINFO: {info_type} - {info_reason}")
                    
            except json.JSONDecodeError as e:
                print(f"\nERROR: Failed to parse server message: {e}")
                break
            except Exception as e:
                print(f"\nERROR: Unexpected error in message handling: {e}")
                break
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\nERROR: WebSocket connection closed: {e}")
    except Exception as e:
        print(f"\nERROR: Unexpected error in receive handler: {e}")

async def send_handler(websocket):
    """
    Configure the session and send audio data from the microphone
    Follows the strict message protocol sequence using Speechmatics models
    """
    global audio_chunks_sent
    
    # Create configuration objects using Speechmatics models with correct parameters
    audio_settings = AudioSettings(
        encoding="pcm_s16le",
        sample_rate=SAMPLE_RATE,
        chunk_size=CHUNK_BYTES
    )
    
    # Create transcription config with optimal settings for your requirements
    transcription_config = TranscriptionConfig(
        language="ru",                    # Russian language
        max_delay=1.0,                    # Ultra-low latency
        max_delay_mode="flexible",        # Better entity formatting
        operating_point="enhanced",       # Highest accuracy model
        enable_partials=True              # Critical for <500ms perceived latency
    )
    
    # Create the StartRecognition message using Speechmatics models
    start_recognition_message = {
        "message": "StartRecognition",
        "audio_format": {
            "type": "raw",  # Add type manually since AudioSettings doesn't have it
            "encoding": audio_settings.encoding,
            "sample_rate": audio_settings.sample_rate
        },
        "transcription_config": transcription_config.asdict()
    }
    
    try:
        # Send the configuration to start the session
        await websocket.send(json.dumps(start_recognition_message))
        print("INFO: StartRecognition message sent. Waiting for confirmation...")
        
        # Stream audio from the microphone generator
        audio_generator = mic_stream_generator()
        async for audio_chunk in audio_generator:
            try:
                await websocket.send(audio_chunk)
                audio_chunks_sent += 1  # Track sequence numbers
            except websockets.exceptions.ConnectionClosed:
                print("ERROR: Connection closed by server while sending audio.")
                break
            except Exception as e:
                print(f"ERROR: Failed to send audio chunk: {e}")
                break
        
        # Send EndOfStream with correct sequence number (required by official docs)
        end_of_stream_message = {
            "message": "EndOfStream",
            "last_seq_no": audio_chunks_sent  # Must match actual number of AddAudio messages
        }
        await websocket.send(json.dumps(end_of_stream_message))
        print(f"\nINFO: EndOfStream message sent with last_seq_no: {audio_chunks_sent}")
        
    except Exception as e:
        print(f"ERROR: Unexpected error in send handler: {e}")

async def main():
    """
    Main entry point for the real-time transcription client
    Establishes WebSocket connection and runs sender/receiver concurrently
    """
    print("INFO: Connecting to Speechmatics Real-Time API...")
    
    try:
        # Establish WebSocket connection with authentication
        async with websockets.connect(
            CONNECTION_URL,
            extra_headers={"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"},
            ping_interval=30,  # Send ping every 30 seconds (within 20-60s range)
            ping_timeout=60    # Wait 60 seconds for pong response
        ) as websocket:
            print("INFO: WebSocket connection established successfully.")
            
            # Run the sender and receiver coroutines concurrently
            receiver_task = asyncio.create_task(receive_handler(websocket))
            sender_task = asyncio.create_task(send_handler(websocket))
            
            # Wait for both tasks to complete
            done, pending = await asyncio.wait(
                [receiver_task, sender_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Clean up pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"ERROR: Connection failed. Status code: {e.status_code}")
        if e.status_code == 401:
            print("REASON: Authentication failed. Check your SPEECHMATICS_API_KEY in .env file.")
        elif e.status_code == 404:
            print("REASON: Invalid endpoint URL.")
        elif e.status_code == 405:
            print("REASON: Invalid request method.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"ERROR: Connection closed unexpectedly: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nINFO: Process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Fatal error: {e}")
        sys.exit(1) 