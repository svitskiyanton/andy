#!/usr/bin/env python3
"""
Text Preprocessor - Breaks long sentences into shorter chunks for better TTS processing
Maintains continuous flow to prevent audio gaps
"""

import re

class TextPreprocessor:
    def __init__(self):
        # Maximum sentence length (characters)
        self.max_sentence_length = 150  # Shorter for better TTS
        
        # Natural break points (in order of preference)
        self.break_points = [
            r'\.\s+',      # Period followed by space
            r'\?\s+',      # Question mark followed by space
            r'!\s+',       # Exclamation mark followed by space
            r'\.\.\.\s+',  # Ellipsis followed by space
            r';\s+',       # Semicolon followed by space
            r',\s+',       # Comma followed by space
            r'—\s+',       # Em dash followed by space
            r'-\s+',       # Hyphen followed by space
            r':\s+',       # Colon followed by space
        ]
    
    def preprocess_text(self, text):
        """Preprocess text to break long sentences into shorter chunks"""
        print("🔧 Preprocessing text for better TTS processing...")
        
        # Split into paragraphs first
        paragraphs = text.split('\n')
        processed_paragraphs = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Process each paragraph
                processed_paragraph = self._process_paragraph(paragraph.strip())
                processed_paragraphs.append(processed_paragraph)
            else:
                # Keep empty lines
                processed_paragraphs.append('')
        
        # Join paragraphs back together with single spaces (not newlines)
        processed_text = ' '.join(processed_paragraphs)
        
        # Clean up multiple spaces
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        # Count sentences
        original_sentences = len(re.findall(r'[.!?]+', text))
        processed_sentences = len(re.findall(r'[.!?]+', processed_text))
        
        print(f"📊 Preprocessing results:")
        print(f"   Original sentences: {original_sentences}")
        print(f"   Processed sentences: {processed_sentences}")
        print(f"   Text length: {len(text):,} → {len(processed_text):,} characters")
        
        return processed_text
    
    def _process_paragraph(self, paragraph):
        """Process a single paragraph"""
        # If paragraph is already short enough, return as is
        if len(paragraph) <= self.max_sentence_length:
            return paragraph
        
        # Try to break at natural points
        for break_pattern in self.break_points:
            if re.search(break_pattern, paragraph):
                return self._break_at_pattern(paragraph, break_pattern)
        
        # If no natural breaks, force break at word boundary
        return self._force_break(paragraph)
    
    def _break_at_pattern(self, text, pattern):
        """Break text at specific pattern"""
        parts = re.split(pattern, text)
        result_parts = []
        current_part = ""
        
        for i, part in enumerate(parts):
            if i == 0:
                current_part = part
            else:
                # Add the break character back
                break_char = re.search(pattern, text).group()
                test_part = current_part + break_char + part
                
                if len(test_part) <= self.max_sentence_length:
                    current_part = test_part
                else:
                    if current_part:
                        result_parts.append(current_part)
                    current_part = part
        
        if current_part:
            result_parts.append(current_part)
        
        # Join with spaces instead of newlines to maintain flow
        return ' '.join(result_parts)
    
    def _force_break(self, text):
        """Force break at word boundary"""
        words = text.split()
        result_parts = []
        current_part = ""
        
        for word in words:
            test_part = current_part + (" " + word) if current_part else word
            
            if len(test_part) <= self.max_sentence_length:
                current_part = test_part
            else:
                if current_part:
                    result_parts.append(current_part)
                current_part = word
        
        if current_part:
            result_parts.append(current_part)
        
        # Join with spaces instead of newlines to maintain flow
        return ' '.join(result_parts)

def preprocess_file(input_file, output_file=None):
    """Preprocess a text file"""
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    processed_text = preprocessor.preprocess_text(text)
    
    # Write output file
    if output_file is None:
        output_file = input_file.replace('.txt', '_processed.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    print(f"✅ Processed text saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    # Example usage
    input_file = "large_text.txt"
    output_file = preprocess_file(input_file)
    
    print(f"\n📝 Original text preview:")
    with open(input_file, 'r', encoding='utf-8') as f:
        original = f.read()
    print(original[:200] + "...")
    
    print(f"\n🔧 Processed text preview:")
    with open(output_file, 'r', encoding='utf-8') as f:
        processed = f.read()
    print(processed[:200] + "...") 