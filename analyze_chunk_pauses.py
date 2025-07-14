#!/usr/bin/env python3
"""
Simple Audio Chunk Pause Analyzer
=================================

Analyzes an audio chunk to detect pauses and help debug why real-time pipeline
didn't split it into smaller chunks.
"""

import os
import sys
import numpy as np
import pyaudio
import webrtcvad
from pydub import AudioSegment
import matplotlib.pyplot as plt
from pathlib import Path

class ChunkPauseAnalyzer:
    def __init__(self):
        # Audio settings (same as realtime pipeline)
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 2048
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        # VAD settings (same as realtime pipeline)
        self.VAD_MODE = 2
        self.VAD_SAMPLE_RATE = 16000
        self.VAD_FRAME_DURATION = 0.03
        
        # Pause detection settings (same as realtime pipeline)
        self.SILENCE_THRESHOLD = 0.005
        self.SILENCE_DURATION = 0.3
        self.MIN_PHRASE_DURATION = 0.8
        self.MAX_PHRASE_DURATION = 6.0
        
        # Initialize VAD
        try:
            import webrtcvad_wheels as webrtcvad
            print("✓ Using webrtcvad-wheels for VAD")
        except ImportError:
            try:
                import webrtcvad
                print("✓ Using webrtcvad for VAD")
            except ImportError:
                print("✗ No VAD library found. Please install webrtcvad-wheels")
                print("  pip install webrtcvad-wheels")
                sys.exit(1)
        
        self.vad = webrtcvad.Vad(self.VAD_MODE)
    
    def analyze_chunk(self, audio_file_path):
        """Analyze pauses in an audio chunk"""
        print(f"🔍 Analyzing chunk: {audio_file_path}")
        print("=" * 60)
        
        # Load audio file
        try:
            if audio_file_path.endswith('.mp3'):
                audio = AudioSegment.from_mp3(audio_file_path)
            elif audio_file_path.endswith('.wav'):
                audio = AudioSegment.from_wav(audio_file_path)
            else:
                print(f"❌ Unsupported file format: {audio_file_path}")
                return
            
            # Convert to numpy array
            audio_array = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                audio_array = audio_array.reshape((-1, 2))[:, 0]  # Take left channel
            
            duration = len(audio_array) / audio.frame_rate
            print(f"📊 Audio duration: {duration:.2f}s")
            print(f"📊 Sample rate: {audio.frame_rate}Hz")
            print(f"📊 Channels: {audio.channels}")
            print(f"📊 Total samples: {len(audio_array)}")
            
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            return
        
        # Analyze in small chunks (same as realtime pipeline)
        chunk_samples = int(self.SAMPLE_RATE * 0.1)  # 100ms chunks for analysis
        silence_periods = []
        speech_periods = []
        current_silence_start = None
        current_speech_start = None
        
        print(f"\n🔍 Analyzing in {chunk_samples} sample chunks...")
        
        for i in range(0, len(audio_array), chunk_samples):
            chunk = audio_array[i:i + chunk_samples]
            if len(chunk) < chunk_samples // 2:  # Skip incomplete chunks
                break
            
            # Calculate audio level
            audio_level = np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768.0
            
            # Detect speech using VAD
            is_speech = self.detect_speech_in_chunk(chunk)
            
            current_time = i / self.SAMPLE_RATE
            
            # Check for silence
            if audio_level < self.SILENCE_THRESHOLD:
                if current_silence_start is None:
                    current_silence_start = current_time
                if current_speech_start is not None:
                    # End speech period
                    speech_periods.append((current_speech_start, current_time))
                    current_speech_start = None
            else:
                if current_silence_start is not None:
                    # End silence period
                    silence_periods.append((current_silence_start, current_time))
                    current_silence_start = None
                if current_speech_start is None:
                    current_speech_start = current_time
        
        # Handle final periods
        if current_silence_start is not None:
            silence_periods.append((current_silence_start, duration))
        if current_speech_start is not None:
            speech_periods.append((current_speech_start, duration))
        
        # Analyze results
        print(f"\n📊 Analysis Results:")
        print(f"   Silence periods found: {len(silence_periods)}")
        print(f"   Speech periods found: {len(speech_periods)}")
        
        # Check for significant pauses
        significant_pauses = []
        for start, end in silence_periods:
            pause_duration = end - start
            if pause_duration >= self.SILENCE_DURATION:
                significant_pauses.append((start, end, pause_duration))
        
        print(f"\n🔇 Significant pauses (≥{self.SILENCE_DURATION}s):")
        if significant_pauses:
            for i, (start, end, duration) in enumerate(significant_pauses, 1):
                print(f"   Pause {i}: {start:.2f}s - {end:.2f}s ({duration:.2f}s)")
        else:
            print("   ❌ No significant pauses found!")
            print(f"   This explains why the chunk wasn't split.")
        
        # Check if any pause would have triggered splitting
        print(f"\n🎯 Pipeline Settings Check:")
        print(f"   Silence threshold: {self.SILENCE_THRESHOLD}")
        print(f"   Required silence duration: {self.SILENCE_DURATION}s")
        print(f"   Min phrase duration: {self.MIN_PHRASE_DURATION}s")
        print(f"   Max phrase duration: {self.MAX_PHRASE_DURATION}s")
        
        # Show all silence periods for debugging
        print(f"\n🔍 All silence periods:")
        for i, (start, end) in enumerate(silence_periods, 1):
            duration = end - start
            print(f"   {i}: {start:.2f}s - {end:.2f}s ({duration:.2f}s)")
        
        # Show all speech periods
        print(f"\n🗣️ All speech periods:")
        for i, (start, end) in enumerate(speech_periods, 1):
            duration = end - start
            print(f"   {i}: {start:.2f}s - {end:.2f}s ({duration:.2f}s)")
    
    def detect_speech_in_chunk(self, audio_chunk):
        """Detect speech in audio chunk using WebRTC VAD (same as realtime pipeline)"""
        try:
            # Convert to 16kHz for VAD
            audio_16k = self._resample_audio(audio_chunk, self.VAD_SAMPLE_RATE)
            
            # Convert to int16
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            
            # Frame size for VAD
            frame_size = int(self.VAD_SAMPLE_RATE * self.VAD_FRAME_DURATION)
            
            speech_detected = False
            
            # Check each frame
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                if len(frame) == frame_size:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.VAD_SAMPLE_RATE)
                    if is_speech:
                        speech_detected = True
                        break
            
            return speech_detected
            
        except Exception as e:
            print(f"❌ VAD error: {e}")
            return True  # Default to speech if VAD fails
    
    def _resample_audio(self, audio_data, target_sample_rate):
        """Simple resampling by interpolation (same as realtime pipeline)"""
        if target_sample_rate == self.SAMPLE_RATE:
            return audio_data
        
        # Simple linear interpolation resampling
        original_length = len(audio_data)
        target_length = int(original_length * target_sample_rate / self.SAMPLE_RATE)
        
        indices = np.linspace(0, original_length - 1, target_length)
        return np.interp(indices, np.arange(original_length), audio_data)

def main():
    """Main function"""
    print("🔍 Audio Chunk Pause Analyzer")
    print("=" * 50)
    print("This script analyzes an audio chunk to detect pauses and help debug")
    print("why the real-time pipeline didn't split it into smaller chunks.")
    print()
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_chunk_pauses.py <audio_file>")
        print("Example: python analyze_chunk_pauses.py sts_chunk_001_20250715_000637.mp3")
        return
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return
    
    # Create analyzer and analyze the chunk
    analyzer = ChunkPauseAnalyzer()
    analyzer.analyze_chunk(audio_file)

if __name__ == "__main__":
    main() 