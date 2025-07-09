#!/usr/bin/env python3
"""
Test script to demonstrate the chunking issue where spaces are lost
"""

def split_text_into_chunks_broken(text, max_chunk_size=100):
    """Original broken chunking function"""
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

def split_text_into_chunks_fixed(text, max_chunk_size=100):
    """Fixed chunking function that preserves original spacing"""
    chunks = []
    current_chunk = ""
    current_length = 0
    
    # Split by words but preserve the original spacing
    import re
    # Split by word boundaries but keep the separators
    parts = re.split(r'(\s+)', text)
    
    for part in parts:
        # If adding this part would exceed the limit
        if current_length + len(part) > max_chunk_size and current_chunk:
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

def test_chunking():
    """Test both chunking methods"""
    # Test text with various spacing
    test_text = "Начинало темнеть, когда бригадир сказал в последний раз:\n     - Мое окончательное слово - тысяча шестьсот. Причем сейчас, вот здесь, наличными... О"
    
    print("🔍 TESTING CHUNKING METHODS")
    print("=" * 60)
    print(f"📄 Original text: '{test_text}'")
    print(f"📊 Original length: {len(test_text)} characters")
    print()
    
    # Test broken method
    print("❌ BROKEN METHOD (word-based):")
    broken_chunks = split_text_into_chunks_broken(test_text, 50)
    print(f"📦 Chunks created: {len(broken_chunks)}")
    for i, chunk in enumerate(broken_chunks):
        print(f"   Chunk {i+1}: '{chunk}' ({len(chunk)} chars)")
    
    broken_total = sum(len(chunk) for chunk in broken_chunks)
    print(f"📊 Total in chunks: {broken_total} characters")
    print(f"📊 Characters lost: {len(test_text) - broken_total}")
    print()
    
    # Test fixed method
    print("✅ FIXED METHOD (preserves words AND spacing):")
    fixed_chunks = split_text_into_chunks_fixed(test_text, 50)
    print(f"📦 Chunks created: {len(fixed_chunks)}")
    for i, chunk in enumerate(fixed_chunks):
        print(f"   Chunk {i+1}: '{chunk}' ({len(chunk)} chars)")
    
    fixed_total = sum(len(chunk) for chunk in fixed_chunks)
    print(f"📊 Total in chunks: {fixed_total} characters")
    print(f"📊 Characters lost: {len(test_text) - fixed_total}")
    print()
    
    # Verify reconstruction
    print("🔍 RECONSTRUCTION TEST:")
    broken_reconstructed = "".join(broken_chunks)
    fixed_reconstructed = "".join(fixed_chunks)
    
    print(f"❌ Broken method reconstruction: '{broken_reconstructed}'")
    print(f"✅ Fixed method reconstruction: '{fixed_reconstructed}'")
    print(f"📄 Original text: '{test_text}'")
    print()
    
    print(f"❌ Broken matches original: {broken_reconstructed == test_text}")
    print(f"✅ Fixed matches original: {fixed_reconstructed == test_text}")
    
    # Show what was lost in broken method
    if broken_reconstructed != test_text:
        print("\n🔍 WHAT WAS LOST IN BROKEN METHOD:")
        print("Original spacing and formatting:")
        print(repr(test_text))
        print("Broken reconstruction:")
        print(repr(broken_reconstructed))

if __name__ == "__main__":
    test_chunking() 