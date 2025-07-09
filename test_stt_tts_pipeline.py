#!/usr/bin/env python3
"""
Test STT→TTS Pipeline
Tests Google Speech-to-Text and ElevenLabs Text-to-Speech integration
"""

import os
import asyncio
import json
import base64
import websockets
import wave
import pyaudio
import numpy as np
from dotenv import load_dotenv
from google.cloud import speech
import time

# Load environment variables
load_dotenv()

class STTTTSTester:
    def __init__(self):
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
        self.MODEL_ID = "eleven_multilingual_v2"
        self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Initialize Google client
        self.speech_client = None
        if self.GOOGLE_APPLICATION_CREDENTIALS:
            self.speech_client = speech.SpeechClient()
    
    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        print(f"🎤 Recording {duration} seconds of audio...")
        print("   Speak in Russian for best results!")
        
        audio = pyaudio.PyAudio()
        
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        frames = []
        for i in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        audio_data = b''.join(frames)
        print(f"✅ Recorded {len(audio_data)} bytes")
        
        return audio_data
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Google Speech-to-Text"""
        if not self.speech_client:
            print("❌ Google Cloud Speech client not available")
            return None
        
        try:
            print("🔍 Transcribing audio...")
            
            audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="ru-RU",  # Russian
                enable_automatic_punctuation=True,
                model="latest_long"
            )
            
            response = self.speech_client.recognize(config=config, audio=audio)
            
            transcribed_text = ""
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript + " "
            
            text = transcribed_text.strip()
            if text:
                print(f"📝 Transcribed: '{text}'")
                return text
            else:
                print("🔇 No speech detected")
                return None
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    async def text_to_speech(self, text):
        """Convert text to speech using ElevenLabs WebSocket (enhanced debug version)"""
        if not self.ELEVENLABS_API_KEY:
            print("❌ ElevenLabs API key not found")
            return None

        try:
            print("🎵 Converting text to speech...")
            print(f"Text to send: {text!r}")

            uri = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{self.VOICE_ID}/stream-input"
                f"?model_id={self.MODEL_ID}&optimize_streaming_latency=4"
            )
            headers = {
                "xi-api-key": self.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            print(f"WebSocket URI: {uri}")
            print(f"Headers: {headers}")

            async with websockets.connect(uri, extra_headers=headers) as websocket:
                # 1. Send initialization message
                init_message = {
                    "text": " ",
                    "voice_settings": {
                        "speed": 1,
                        "stability": 0.5,
                        "similarity_boost": 0.8
                    },
                    "xi_api_key": self.ELEVENLABS_API_KEY
                }
                print(f"Sending init message: {init_message}")
                await websocket.send(json.dumps(init_message))

                # 2. Send text message
                text_message = {
                    "text": text,
                    "try_trigger_generation": True
                }
                print(f"Sending text message: {text_message}")
                await websocket.send(json.dumps(text_message))

                # 3. Send end marker
                print("Sending end marker: {'text': ''}")
                await websocket.send(json.dumps({"text": ""}))

                # Receive audio
                audio_chunks = []
                debug_responses = []
                idx = 0
                async for message in websocket:
                    print(f"Raw message [{idx}]: {message!r}")
                    try:
                        data = json.loads(message)
                    except Exception as e:
                        print(f"JSON decode error: {e}")
                        continue
                    debug_responses.append(data)
                    print(f"Decoded message [{idx}]: {data}")
                    idx += 1

                    if "audio" in data and data["audio"]:
                        audio_data = base64.b64decode(data["audio"])
                        audio_chunks.append(audio_data)
                    elif "audio" in data and data["audio"] is None:
                        print("Info: Received final message with 'audio': None (end of stream).")

                    if data.get("isFinal"):
                        print("Received isFinal=True, breaking loop.")
                        break

                if audio_chunks:
                    combined_audio = b''.join(audio_chunks)
                    print(f"✅ Generated {len(combined_audio)} bytes of audio")
                    return combined_audio
                else:
                    print("❌ No audio received from ElevenLabs.")
                    print(f"Full ElevenLabs WebSocket responses:")
                    for idx, resp in enumerate(debug_responses):
                        print(f"  [{idx}] {resp}")
                    return None

        except Exception as e:
            print(f"❌ Text-to-speech error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_audio(self, audio_data, filename):
        """Save audio data to file (WAV or MP3)"""
        try:
            from pydub import AudioSegment
            import io
            
            # Determine file format from extension
            file_ext = filename.lower().split('.')[-1]
            
            if file_ext == 'mp3':
                # Save as MP3 (ElevenLabs returns MP3 by default)
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                audio_segment.export(filename, format="mp3")
                print(f"💾 MP3 audio saved to: {filename}")
            else:
                # Save as WAV (decode MP3 first)
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                audio_segment.export(filename, format="wav")
                print(f"💾 WAV audio saved to: {filename}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return False
    
    def play_audio(self, audio_data):
        """Play audio through speakers, decoding MP3 to PCM if needed"""
        try:
            print("🔊 Playing audio...")
            from pydub import AudioSegment
            import io

            # Decode MP3 to PCM using pydub
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            raw_pcm = audio_segment.raw_data
            sample_width = audio_segment.sample_width
            frame_rate = audio_segment.frame_rate
            channels = audio_segment.channels

            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=audio.get_format_from_width(sample_width),
                channels=channels,
                rate=frame_rate,
                output=True
            )

            chunk_size = 1024 * 2
            for i in range(0, len(raw_pcm), chunk_size):
                chunk = raw_pcm[i:i + chunk_size]
                stream.write(chunk)

            stream.stop_stream()
            stream.close()
            audio.terminate()
            print("✅ Audio playback complete")
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
    
    async def run_test(self):
        """Run the complete STT→TTS test"""
        print("🧪 STT→TTS Pipeline Test")
        print("=" * 30)
        
        # Check prerequisites
        if not self.ELEVENLABS_API_KEY:
            print("❌ ELEVENLABS_API_KEY not found")
            return False
        
        if not self.GOOGLE_APPLICATION_CREDENTIALS:
            print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
            print("   Run: python setup_google_cloud.py")
            return False
        
        print("✅ Prerequisites check passed")
        
        # Step 1: Record audio
        audio_data = self.record_audio(duration=5)
        
        # Step 2: Transcribe audio
        text = self.transcribe_audio(audio_data)
        if not text:
            print("❌ No text transcribed, cannot continue")
            return False
        
        # Step 3: Convert text to speech
        tts_audio = await self.text_to_speech(text)
        if not tts_audio:
            print("❌ Text-to-speech failed")
            return False
        
        # Step 4: Save and play result
        timestamp = int(time.time())
        wav_filename = f"stt_tts_test_{timestamp}.wav"
        mp3_filename = f"stt_tts_test_{timestamp}.mp3"
        
        # Save both formats
        self.save_audio(tts_audio, wav_filename)
        self.save_audio(tts_audio, mp3_filename)
        
        # Play the audio
        self.play_audio(tts_audio)
        
        print("\n🎉 STT→TTS pipeline test completed successfully!")
        return True

async def main():
    """Main test function"""
    tester = STTTTSTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main()) 