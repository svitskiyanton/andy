#!/usr/bin/env python3
"""
Text Streamer - External script to stream large text files to progressive streaming test
Shows console progress while streaming text chunks
"""

import asyncio
import time
import os
from dotenv import load_dotenv
import re # Added for text cleaning

# Import the progressive streaming class
from progressive_streaming_test import ProgressiveWebSocketStreamingTest

# Load environment variables
load_dotenv()

class TextStreamer:
    def __init__(self, text_file_path="large_text.txt"):
        self.text_file_path = text_file_path
        self.chunk_size = 50  # Smaller chunks for better TTS handling
        self.delay_between_chunks = 0.1  # Seconds between chunks
        
        # Streaming metrics
        self.total_chunks = 0
        self.sent_chunks = 0
        self.start_time = None
        
        # Check if text file exists
        if not os.path.exists(self.text_file_path):
            self._create_sample_text_file()
    
    def _create_sample_text_file(self):
        """Create a sample large text file with various gaps and formatting issues"""
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
    
    def _detect_text_gaps(self, text):
        """Detect and analyze gaps in text"""
        gaps = []
        
        # Find large whitespace gaps (3+ spaces or newlines)
        import re
        gap_patterns = [
            (r'\n\s*\n', 'Double newlines'),
            (r' {3,}', 'Multiple spaces'),
            (r'\t+', 'Tabs'),
            (r'\n\s{2,}', 'Indented lines'),
            (r'\.{3,}', 'Ellipsis sequences'),
            (r'!{2,}', 'Multiple exclamation marks'),
            (r'\?{2,}', 'Multiple question marks'),
        ]
        
        for pattern, description in gap_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                gaps.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'type': description,
                    'length': len(match.group())
                })
        
        return sorted(gaps, key=lambda x: x['start'])
    
    def _improve_number_pronunciation(self, text):
        """Improve number pronunciation for Russian TTS"""
        import re
        
        # Step 1: Add spaces around numbers for better pronunciation
        # This helps TTS break up long number sequences
        text = re.sub(r'(\d+)([а-яё])', r'\1 \2', text)  # Number followed by Russian letter
        text = re.sub(r'([а-яё])(\d+)', r'\1 \2', text)  # Russian letter followed by number
        
        # Step 2: Format large numbers with spaces for better pronunciation
        # Add spaces every 3 digits from the right for thousands
        def format_large_numbers(match):
            number = match.group(1)
            if len(number) > 3:
                # Add spaces every 3 digits from the right
                formatted = ''
                for i, digit in enumerate(reversed(number)):
                    if i > 0 and i % 3 == 0:
                        formatted = ' ' + formatted
                    formatted = digit + formatted
                return formatted
            return number
        
        text = re.sub(r'(\d{4,})', format_large_numbers, text)
        
        # Step 3: Improve specific number patterns
        # Phone numbers
        text = re.sub(r'(\d{1})(\d{3})(\d{3})(\d{2})(\d{2})', r'\1 \2 \3 \4 \5', text)  # +7 123 456 78 90
        text = re.sub(r'(\d{3})(\d{3})(\d{2})(\d{2})', r'\1 \2 \3 \4', text)  # 123 456 78 90
        
        # Years
        text = re.sub(r'(\d{4})', r'\1', text)  # Keep years as is but ensure spacing
        
        # Step 4: Add "тысяча" for better pronunciation of thousands
        def add_thousand_words(match):
            number = match.group(1)
            if len(number) >= 4:
                # For numbers 1000 and above, add "тысяча" context
                return f"{number} (тысяча)"
            return number
        
        # Only apply to standalone numbers (not in phone numbers, years, etc.)
        text = re.sub(r'\b(\d{4,})\b', add_thousand_words, text)
        
        return text
    
    def _clean_text_advanced(self, text):
        """Advanced text cleaning that handles gaps and formatting issues"""
        original_length = len(text)
        
        # Step 1: Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Step 2: Handle paragraph breaks (double newlines)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Step 3: Normalize indentation (preserve single spaces, normalize multiple)
        text = re.sub(r' {2,}', ' ', text)
        
        # Step 4: Clean up punctuation sequences
        text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipsis
        text = re.sub(r'!{2,}', '!', text)     # Normalize exclamation marks
        text = re.sub(r'\?{2,}', '?', text)    # Normalize question marks
        text = re.sub(r'-{2,}', '-', text)     # Normalize hyphens
        text = re.sub(r'—+', '—', text)        # Normalize em dashes
        
        # Step 5: Remove excessive tabs
        text = re.sub(r'\t+', ' ', text)
        
        # Step 6: Clean up leading/trailing whitespace on lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Preserve intentional indentation (single space at start)
            if line.startswith(' '):
                line = ' ' + line.lstrip()
            else:
                line = line.strip()
            if line:  # Only add non-empty lines
                cleaned_lines.append(line)
        
        # Step 7: Join lines with appropriate spacing
        text = '\n'.join(cleaned_lines)
        
        # Step 8: Improve number pronunciation
        text = self._improve_number_pronunciation(text)
        
        # Step 9: Final whitespace normalization
        text = text.strip()
        
        return text, original_length - len(text)
    
    def _show_number_improvements(self, original_text, cleaned_text):
        """Show how numbers were improved for pronunciation"""
        import re
        
        # Find all numbers in original text
        original_numbers = re.findall(r'\d+', original_text)
        cleaned_numbers = re.findall(r'\d+', cleaned_text)
        
        if original_numbers:
            print("🔢 Number Processing Analysis:")
            print("   Original numbers found:", original_numbers)
            print("   Cleaned numbers found:", cleaned_numbers)
            
            # Show specific improvements
            improvements = []
            for num in original_numbers:
                if len(num) >= 4:
                    # Show how large numbers were formatted
                    formatted = re.search(rf'\b{num}\b', cleaned_text)
                    if formatted:
                        context = cleaned_text[max(0, formatted.start()-20):formatted.end()+20]
                        improvements.append(f"   {num} → {context.strip()}")
            
            if improvements:
                print("   Number improvements:")
                for imp in improvements[:5]:  # Show first 5 improvements
                    print(imp)
                if len(improvements) > 5:
                    print(f"   ... and {len(improvements) - 5} more improvements")
    
    def _split_text_into_chunks_improved(self, text, max_chunk_size=100):
        """Improved chunking that handles gaps and maintains flow"""
        chunks = []
        current_chunk = ""
        current_length = 0
        
        # Split by words but preserve the original spacing
        import re
        # Split by word boundaries but keep the separators
        parts = re.split(r'(\s+)', text)
        
        for i, part in enumerate(parts):
            # Check if this part would exceed the limit
            if current_length + len(part) > max_chunk_size and current_chunk:
                # Special handling for sentence endings
                if part.strip() and not current_chunk.endswith(('.', '!', '?', ':')):
                    # Try to find a better break point within the current chunk
                    better_break = self._find_better_break_point(current_chunk, max_chunk_size)
                    if better_break > 0:
                        # Split at the better break point
                        first_part = current_chunk[:better_break]
                        second_part = current_chunk[better_break:] + part
                        chunks.append(first_part)
                        current_chunk = second_part
                        current_length = len(second_part)
                        continue
                
                # Save current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = part
                current_length = len(part)
            else:
                # Add to current chunk
                current_chunk += part
                current_length += len(part)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _find_better_break_point(self, text, max_size):
        """Find a better break point within text (prefer sentence endings)"""
        # Look for sentence endings in reverse order
        for i in range(len(text) - 1, max(0, len(text) - 50), -1):
            if text[i] in '.!?:' and i < max_size:
                return i + 1
        
        # Look for commas or other natural breaks
        for i in range(len(text) - 1, max(0, len(text) - 30), -1):
            if text[i] in ',;' and i < max_size:
                return i + 1
        
        return 0  # No better break point found
    
    def _show_progress(self, current, total, chunk_text):
        """Show streaming progress in console"""
        progress = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r📤 [{bar}] {progress:.1f}% ({current}/{total}) | '{chunk_text[:30]}...'", end='', flush=True)
    
    async def stream_text_to_progressive(self):
        """Stream text from file to progressive streaming test with improved gap handling"""
        # Read text file
        text_content = self._read_text_file()
        if not text_content:
            return
        
        # Detect gaps in original text
        gaps = self._detect_text_gaps(text_content)
        
        # Advanced text cleaning
        cleaned_text, chars_removed = self._clean_text_advanced(text_content)
        
        # Debug: Show text processing
        print("🔍 DEBUG: Text Processing Analysis")
        print("=" * 60)
        print(f"📄 Original text length: {len(text_content):,} characters")
        print(f"🧹 Cleaned text length: {len(cleaned_text):,} characters")
        print(f"📊 Characters removed: {chars_removed}")
        
        # Show gap analysis
        if gaps:
            print(f"🔍 Gap Analysis:")
            print(f"   Found {len(gaps)} potential gaps:")
            for i, gap in enumerate(gaps[:5]):  # Show first 5 gaps
                print(f"   Gap {i+1}: {gap['type']} at position {gap['start']}-{gap['end']} (length: {gap['length']})")
            if len(gaps) > 5:
                print(f"   ... and {len(gaps) - 5} more gaps")
        else:
            print("✅ No significant gaps detected")
        
        # Show number improvements
        self._show_number_improvements(text_content, cleaned_text)
        
        # Show text differences
        if chars_removed > 0:
            print("⚠️ Text was modified during cleaning!")
            print("Original preview:", text_content[:100] + "...")
            print("Cleaned preview:", cleaned_text[:100] + "...")
        else:
            print("✅ Text cleaning: No changes")
        print("=" * 60)

        # Split into chunks with improved method
        text_chunks = self._split_text_into_chunks_improved(cleaned_text, max_chunk_size=100)
        self.total_chunks = len(text_chunks)
        
        # Debug: Show chunking analysis
        print("🔍 DEBUG: Chunking Analysis")
        print("=" * 60)
        print(f"📦 Total chunks created: {self.total_chunks}")
        print(f"⚙️ Chunk size limit: 100 characters")
        
        # Analyze chunk sizes
        chunk_sizes = [len(chunk) for chunk in text_chunks]
        print(f"📊 Chunk size stats:")
        print(f"   Smallest: {min(chunk_sizes)} characters")
        print(f"   Largest: {max(chunk_sizes)} characters")
        print(f"   Average: {sum(chunk_sizes) / len(chunk_sizes):.1f} characters")
        
        # Show first few chunks
        print(f"📋 First 5 chunks:")
        for i, chunk in enumerate(text_chunks[:5]):
            print(f"   Chunk {i+1}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}' ({len(chunk)} chars)")
        
        # Show last few chunks
        if len(text_chunks) > 5:
            print(f"📋 Last 5 chunks:")
            for i, chunk in enumerate(text_chunks[-5:]):
                actual_index = len(text_chunks) - 5 + i
                print(f"   Chunk {actual_index+1}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}' ({len(chunk)} chars)")
        
        # Verify total characters
        total_chars_in_chunks = sum(len(chunk) for chunk in text_chunks)
        print(f"🔢 Character verification:")
        print(f"   Original text: {len(cleaned_text):,} characters")
        print(f"   In chunks: {total_chars_in_chunks:,} characters")
        print(f"   Difference: {len(cleaned_text) - total_chars_in_chunks} characters")
        
        if total_chars_in_chunks != len(cleaned_text):
            print("⚠️ WARNING: Character count mismatch!")
            print("   This indicates text loss during chunking!")
        else:
            print("✅ Character count: Perfect match")
        print("=" * 60)
        
        print("🎵 Starting Text Streamer for Progressive Streaming Test...")
        print("=" * 80)
        print(f"📁 Text file: {self.text_file_path}")
        print(f"📊 Total text length: {len(cleaned_text):,} characters")
        print(f"📦 Total chunks: {self.total_chunks}")
        print(f"⏱️  Estimated time: {self.total_chunks * self.delay_between_chunks:.1f} seconds")
        print("=" * 80)
        
        # Create progressive streaming instance
        progressive_test = ProgressiveWebSocketStreamingTest()
        
        # Replace the test text with our file content
        progressive_test.test_text = cleaned_text
        
        # Debug: Track what we're sending
        print("🔍 DEBUG: Sending to Progressive Streaming")
        print("=" * 60)
        print(f"📤 Text being sent: '{cleaned_text[:100]}{'...' if len(cleaned_text) > 100 else ''}'")
        print(f"📦 Number of chunks: {self.total_chunks}")
        print("=" * 60)
        
        # Start the progressive streaming
        print("\n🎵 Starting progressive streaming with file content...")
        self.start_time = time.time()
        
        # Run the progressive streaming test
        await progressive_test.stream_text_to_speech_progressive()
        
        # Calculate final metrics
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\n📊 Streaming completed in {total_time:.2f} seconds")
            print(f"📦 Processed {self.total_chunks} text chunks")
            
            # Final debug summary
            print("\n🔍 DEBUG: Final Summary")
            print("=" * 60)
            print(f"📄 Original text: {len(text_content):,} characters")
            print(f"🧹 Cleaned text: {len(cleaned_text):,} characters")
            print(f"📦 Chunks created: {self.total_chunks}")
            print(f"⏱️  Total time: {total_time:.2f} seconds")
            print(f"⚡ Processing rate: {self.total_chunks/total_time:.1f} chunks/second")
            print("=" * 60)
            
            # Verify the saved MP3 file
            self._verify_saved_mp3()

    def _create_gap_test_file(self):
        """Create a test file with various gaps and formatting issues"""
        gap_test_text = """
Это текст с различными пробелами и форматированием.

     - Диалог с отступом и множественными пробелами:    вот    так    много    пробелов!

Параграф с двойными переносами строк...


И еще один параграф.

Текст с табуляцией:	вот	так	табуляция.

Множественные знаки препинания!!!???... И еще больше!!!???

Длинное предложение без естественных точек разрыва которое может вызвать проблемы при разбивке на части для потоковой передачи текста в речь

Финальная часть с различными символами: !@#$%^&*() и цифрами 1234567890.

Теперь добавим числа для тестирования произношения:
- Простой номер: 123
- Большое число: 1234567
- Год: 2024
- Телефон: 89123456789
- Цена: 1500рублей
- Количество: 1000штук
- Код: ABC123DEF
- Номер карты: 1234567890123456
- Процент: 25%
- Температура: -15градусов
        """
        
        test_file = "gap_test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(gap_test_text)
        
        print(f"📝 Created gap test file: {test_file}")
        return test_file

    def _verify_saved_mp3(self):
        """Verify the saved MP3 file"""
        try:
            from pydub import AudioSegment
            import os
            
            output_file = "progressive_streaming_output.mp3"
            if not os.path.exists(output_file):
                print("❌ MP3 file not found for verification")
                return False
            
            # Load and analyze the audio file
            audio = AudioSegment.from_file(output_file, format="mp3")
            file_size = os.path.getsize(output_file)
            duration = len(audio) / 1000.0
            
            print("\n🔍 MP3 File Verification")
            print("=" * 60)
            print(f"📁 File: {output_file}")
            print(f"📊 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            print(f"🎵 Sample rate: {audio.frame_rate} Hz")
            print(f"🔊 Channels: {audio.channels}")
            print(f"🔊 Average amplitude: {audio.dBFS:.1f} dB")
            
            # Quality checks
            if audio.dBFS < -50:
                print("⚠️  WARNING: Audio appears to be very quiet or silent")
            elif audio.dBFS < -30:
                print("⚠️  WARNING: Audio is quite quiet")
            else:
                print("✅ Audio volume appears normal")
            
            if audio.max_possible_amplitude > 0.95:
                print("⚠️  WARNING: Audio may be clipping")
            else:
                print("✅ No clipping detected")
            
            # Calculate bitrate
            bitrate = (file_size * 8) / duration
            print(f"📊 Calculated bitrate: {bitrate / 1000:.0f} kbps")
            
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"❌ Error verifying MP3 file: {e}")
            return False

async def main():
    """Main function"""
    import sys
    
    # Check if user wants to test gap handling
    if len(sys.argv) > 1 and sys.argv[1] == "--gap-test":
        print("🧪 Testing gap handling with problematic text...")
        streamer = TextStreamer("gap_test.txt")
        # Create the gap test file
        streamer._create_gap_test_file()
    else:
        # Use default text file
        text_file = "large_text.txt"  # Change this to your text file
        streamer = TextStreamer(text_file)
    
    await streamer.stream_text_to_progressive()

if __name__ == "__main__":
    asyncio.run(main()) 