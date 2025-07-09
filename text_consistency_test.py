#!/usr/bin/env python3
"""
Text Consistency Test - Test version that simulates text streaming without ElevenLabs
Shows input/output text consistency and buffering behavior
"""

import asyncio
import time
import os
from dotenv import load_dotenv

# Import the progressive streaming class
from progressive_streaming_test import ProgressiveWebSocketStreamingTest

# Load environment variables
load_dotenv()

class TextConsistencyTest:
    def __init__(self, text_file_path="large_text.txt"):
        self.text_file_path = text_file_path
        self.chunk_size = 100  # Characters per chunk
        self.delay_between_chunks = 0.1  # Seconds between chunks
        
        # Test metrics
        self.total_chunks = 0
        self.processed_chunks = 0
        self.start_time = None
        
        # Text tracking
        self.input_text = ""
        self.output_text = ""
        self.chunk_history = []
        
        # Check if text file exists
        if not os.path.exists(self.text_file_path):
            self._create_sample_text_file()
    
    def _create_sample_text_file(self):
        """Create a sample large text file if it doesn't exist"""
        sample_text = """
Так почему же вы здесь?
     - А где же мне быть? Где же мне работать, по-твоему? В школе? Что я там
буду воровать, промокашки?!  Устраиваясь на работу, ты  должен  прежде всего
задуматься: что, где и как? Что я смогу украсть? Где  я смогу украсть? И как
я  смогу украсть?.. Ты  понял?  Вот и хорошо. Все будет нормально.  К вечеру
бабки появятся.

Это продолжение большого текста для тестирования потоковой передачи. Мы будем
отправлять этот текст небольшими частями, чтобы проверить, как работает наша
система прогрессивного стриминга. Каждая часть будет обрабатываться отдельно,
что позволит нам получить плавное воспроизведение аудио в реальном времени.

Продолжаем тестирование с еще большим количеством текста. Это поможет нам
убедиться, что система работает стабильно даже с длинными текстами. Мы также
проверим, как система справляется с различными типами предложений и пунктуации.

Еще больше текста для тестирования. Этот текст содержит различные символы,
цифры 123, и специальные символы: !@#$%^&*(). Мы хотим убедиться, что наша
система корректно обрабатывает все типы символов и знаков препинания.

Финальная часть нашего тестового текста. К этому моменту мы должны были
отправить достаточно текста, чтобы протестировать всю систему потоковой
передачи. Надеемся, что аудио воспроизводится плавно и без прерываний.
        """
        
        with open(self.text_file_path, 'w', encoding='utf-8') as f:
            f.write(sample_text.strip())
        
        print(f"📝 Created sample text file: {self.text_file_path}")
    
    def _read_text_file(self):
        """Read the text file and return its content"""
        try:
            with open(self.text_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content.strip()
        except Exception as e:
            print(f"❌ Error reading text file: {e}")
            return None
    
    def _split_text_into_chunks(self, text):
        """Split text into chunks for streaming (same logic as original)"""
        chunks = []
        current_chunk = ""
        words = text.split()
        
        for word in words:
            if len(current_chunk + " " + word) <= self.chunk_size:
                current_chunk += (" " + word) if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_text_into_chunks_preserve_whitespace(self, text):
        """Split text into chunks while preserving whitespace"""
        chunks = []
        current_chunk = ""
        
        # Split by words but preserve original spacing
        import re
        parts = re.split(r'(\s+)', text)
        
        for part in parts:
            if len(current_chunk + part) <= self.chunk_size:
                current_chunk += part
            else:
                if current_chunk.strip():  # Only add non-empty chunks
                    chunks.append(current_chunk)
                current_chunk = part
        
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        return chunks
    
    def _show_progress(self, current, total, chunk_text):
        """Show streaming progress in console"""
        progress = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r📤 [{bar}] {progress:.1f}% ({current}/{total}) | '{chunk_text[:30]}...'", end='', flush=True)
    
    async def test_text_consistency(self):
        """Test text consistency without sending to ElevenLabs"""
        # Read text file
        text_content = self._read_text_file()
        if not text_content:
            return
        
        # Store original input text
        self.input_text = text_content
        
        # Split into chunks (preserving whitespace)
        text_chunks = self._split_text_into_chunks_preserve_whitespace(text_content)
        self.total_chunks = len(text_chunks)
        
        print("🧪 Starting Text Consistency Test...")
        print("=" * 80)
        print(f"📁 Text file: {self.text_file_path}")
        print(f"📊 Total text length: {len(text_content):,} characters")
        print(f"📦 Total chunks: {self.total_chunks}")
        print(f"⏱️  Simulated time: {self.total_chunks * self.delay_between_chunks:.1f} seconds")
        print("=" * 80)
        
        # Simulate the text streaming process
        print("\n📤 Simulating text chunk streaming...")
        self.start_time = time.time()
        
        # Process each chunk (simulate what the original system does)
        for i, chunk in enumerate(text_chunks):
            # Store chunk history
            self.chunk_history.append({
                'index': i + 1,
                'chunk': chunk,
                'length': len(chunk),
                'timestamp': time.time() - self.start_time
            })
            
            # Show progress
            self._show_progress(i + 1, self.total_chunks, chunk)
            
            # Simulate processing delay
            await asyncio.sleep(self.delay_between_chunks)
        
        # Reconstruct output text from chunks (preserve original spacing)
        self.output_text = "".join([chunk['chunk'] for chunk in self.chunk_history])
        
        # Calculate final metrics
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("🧪 TEXT CONSISTENCY TEST RESULTS:")
        print("=" * 80)
        
        # Input vs Output comparison
        print("📊 TEXT COMPARISON:")
        print(f"Input text length:  {len(self.input_text):,} characters")
        print(f"Output text length: {len(self.output_text):,} characters")
        print(f"Text match: {'✅ YES' if self.input_text == self.output_text else '❌ NO'}")
        
        if self.input_text != self.output_text:
            print("\n⚠️  TEXT DIFFERENCES DETECTED:")
            print("Input text:")
            print(f"'{self.input_text[:100]}...'")
            print("\nOutput text:")
            print(f"'{self.output_text[:100]}...'")
        
        # Chunk analysis
        print(f"\n📦 CHUNK ANALYSIS:")
        print(f"Total chunks processed: {len(self.chunk_history)}")
        print(f"Average chunk size: {sum(chunk['length'] for chunk in self.chunk_history) / len(self.chunk_history):.1f} characters")
        print(f"Largest chunk: {max(chunk['length'] for chunk in self.chunk_history)} characters")
        print(f"Smallest chunk: {min(chunk['length'] for chunk in self.chunk_history)} characters")
        
        # Timing analysis
        print(f"\n⏱️  TIMING ANALYSIS:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per chunk: {total_time / len(self.chunk_history):.3f} seconds")
        print(f"Processing rate: {len(self.chunk_history) / total_time:.1f} chunks/second")
        
        # Show first few chunks
        print(f"\n📋 FIRST 5 CHUNKS:")
        for i, chunk_info in enumerate(self.chunk_history[:5]):
            print(f"Chunk {chunk_info['index']}: '{chunk_info['chunk'][:50]}...' ({chunk_info['length']} chars)")
        
        # Show last few chunks
        if len(self.chunk_history) > 5:
            print(f"\n📋 LAST 5 CHUNKS:")
            for chunk_info in self.chunk_history[-5:]:
                print(f"Chunk {chunk_info['index']}: '{chunk_info['chunk'][:50]}...' ({chunk_info['length']} chars)")
        
        print("=" * 80)
        
        # Final verdict
        if self.input_text == self.output_text:
            print("✅ CONSISTENCY TEST PASSED: Input and output text match perfectly!")
        else:
            print("❌ CONSISTENCY TEST FAILED: Input and output text do not match!")
        
        return self.input_text == self.output_text

async def main():
    """Main function"""
    # You can specify a different text file path here
    text_file = "large_text.txt"  # Change this to your text file
    
    tester = TextConsistencyTest(text_file)
    success = await tester.test_text_consistency()
    
    if success:
        print("\n🎉 Text streaming consistency verified! Your system is ready for ElevenLabs.")
    else:
        print("\n⚠️  Text streaming consistency issues detected. Check the chunking logic.")

if __name__ == "__main__":
    asyncio.run(main()) 