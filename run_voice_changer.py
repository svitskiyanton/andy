#!/usr/bin/env python3
"""
Voice Changer Launcher
Choose between standard, advanced, and ultra-low latency versions
"""

import sys
import os

def main():
    print("🎤 ElevenLabs Real-Time Voice Changer")
    print("=" * 50)
    print()
    print("Choose your version:")
    print("1. Standard Version (Recommended for English)")
    print("   - Single API worker")
    print("   - 0.8-second audio buffers")
    print("   - Good balance of latency and quality")
    print()
    print("2. Advanced Version (Best for English)")
    print("   - Multiple API workers (parallel processing)")
    print("   - 1-second audio buffers")
    print("   - Lower latency, higher resource usage")
    print()
    print("3. STT→TTS Version (Best for Russian/Multi-language)")
    print("   - Google Speech-to-Text + ElevenLabs TTS")
    print("   - 2-second audio buffers")
    print("   - Better quality for non-English languages")
    print()
    print("4. Ultra-Low Latency Version (English only)")
    print("   - Single API worker with maximum optimizations")
    print("   - 0.5-second audio buffers")
    print("   - Fastest processing, minimal latency")
    print()
    print("5. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting Standard Voice Changer...")
                os.system("python real_time_voice_changer.py")
                break
            elif choice == "2":
                print("\n🚀 Starting Advanced Voice Changer...")
                os.system("python real_time_voice_changer_advanced.py")
                break
            elif choice == "3":
                print("\n🚀 Starting STT→TTS Voice Changer...")
                print("   (Make sure you have Google Cloud credentials set up)")
                os.system("python realtime_stt_tts_voice_changer.py")
                break
            elif choice == "4":
                print("\n🚀 Starting Ultra-Low Latency Voice Changer...")
                os.system("python real_time_voice_changer_ultra.py")
                break
            elif choice == "5":
                print("👋 Goodbye!")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 