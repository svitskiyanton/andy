#!/usr/bin/env python3
"""
ElevenLabs Pro Configuration
Defines all Pro-specific settings and optimizations
"""

import os
from typing import Dict, Any

class ProConfig:
    """Configuration class for ElevenLabs Pro features"""
    
    # ============================================================================
    # PRO SUBSCRIPTION LIMITS
    # ============================================================================
    MAX_CONCURRENT_REQUESTS = 10  # Pro tier allows 10 concurrent requests
    MONTHLY_CREDITS = 500_000     # 500k credits per month
    ADDITIONAL_CREDIT_COST = 0.12  # $0.12 per 1000 additional credits
    
    # ============================================================================
    # AUDIO QUALITY SETTINGS (Pro Features)
    # ============================================================================
    AUDIO_QUALITY_OPTIONS = {
        "standard": {
            "name": "Standard (128 kbps)",
            "bitrate": "128k",
            "format": "mp3_44100",
            "description": "Standard quality, faster processing"
        },
        "high": {
            "name": "High Quality (192 kbps)",
            "bitrate": "192k", 
            "format": "mp3_44100_192",
            "description": "Pro: High quality audio"
        },
        "pcm": {
            "name": "PCM (44.1kHz)",
            "bitrate": "1411k",
            "format": "pcm_44100",
            "description": "Pro: Uncompressed PCM audio"
        }
    }
    
    # ============================================================================
    # MODEL SETTINGS (Pro Features)
    # ============================================================================
    MODEL_OPTIONS = {
        "turbo": {
            "id": "eleven_turbo_v2",
            "name": "Turbo v2",
            "description": "Pro: Fastest generation, good quality",
            "best_for": "Real-time applications"
        },
        "multilingual": {
            "id": "eleven_multilingual_v2", 
            "name": "Multilingual v2",
            "description": "Best for multiple languages",
            "best_for": "International content"
        },
        "monolingual": {
            "id": "eleven_monolingual_v1",
            "name": "Monolingual v1",
            "description": "Highest quality for English",
            "best_for": "English content only"
        }
    }
    
    # ============================================================================
    # VOICE SETTINGS (Pro Features)
    # ============================================================================
    VOICE_SETTINGS = {
        "balanced": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "description": "Balanced between stability and similarity"
        },
        "stable": {
            "stability": 0.8,
            "similarity_boost": 0.5,
            "style": 0.0,
            "use_speaker_boost": True,
            "description": "More stable, less similar to original"
        },
        "similar": {
            "stability": 0.3,
            "similarity_boost": 0.9,
            "style": 0.0,
            "use_speaker_boost": True,
            "description": "More similar to original, less stable"
        },
        "creative": {
            "stability": 0.4,
            "similarity_boost": 0.6,
            "style": 0.7,
            "use_speaker_boost": True,
            "description": "Creative style with moderate stability"
        }
    }
    
    # ============================================================================
    # STREAMING OPTIMIZATION (Pro Features)
    # ============================================================================
    STREAMING_OPTIONS = {
        "ultra_fast": {
            "optimize_streaming_latency": "0",
            "description": "Ultra-low latency, may affect quality",
            "best_for": "Real-time voice changing"
        },
        "fast": {
            "optimize_streaming_latency": "4", 
            "description": "Fast with good quality balance",
            "best_for": "Most real-time applications"
        },
        "balanced": {
            "optimize_streaming_latency": "8",
            "description": "Balanced latency and quality",
            "best_for": "General use"
        }
    }
    
    # ============================================================================
    # DEFAULT PRO CONFIGURATION
    # ============================================================================
    DEFAULT_CONFIG = {
        # Audio settings
        "audio_quality": "high",  # Use 192 kbps
        "model": "turbo",         # Use Turbo for speed
        "voice_settings": "balanced",
        "streaming": "fast",
        
        # Performance settings
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "request_timeout": 30,    # seconds
        "retry_attempts": 3,
        
        # Buffer settings
        "input_buffer_size": 100,
        "output_buffer_size": 200,
        "text_buffer_size": 50,
        
        # Audio processing
        "chunk_size": 4096,       # 4KB chunks
        "sample_rate": 16000,     # Google STT requirement
        "output_sample_rate": 44100,  # Pro: 44.1kHz output
        
        # Language settings
        "language_code": "ru-RU",  # Russian
        "enable_automatic_punctuation": True,
        "enable_word_time_offsets": False,
        
        # Monitoring
        "enable_performance_monitoring": True,
        "log_latency_stats": True,
        "save_test_audio": True
    }
    
    @classmethod
    def get_audio_quality_config(cls, quality: str = "high") -> Dict[str, Any]:
        """Get audio quality configuration"""
        return cls.AUDIO_QUALITY_OPTIONS.get(quality, cls.AUDIO_QUALITY_OPTIONS["high"])
    
    @classmethod
    def get_model_config(cls, model: str = "turbo") -> Dict[str, Any]:
        """Get model configuration"""
        return cls.MODEL_OPTIONS.get(model, cls.MODEL_OPTIONS["turbo"])
    
    @classmethod
    def get_voice_settings(cls, preset: str = "balanced") -> Dict[str, Any]:
        """Get voice settings configuration"""
        return cls.VOICE_SETTINGS.get(preset, cls.VOICE_SETTINGS["balanced"])
    
    @classmethod
    def get_streaming_config(cls, option: str = "fast") -> Dict[str, Any]:
        """Get streaming configuration"""
        return cls.STREAMING_OPTIONS.get(option, cls.STREAMING_OPTIONS["fast"])
    
    @classmethod
    def get_websocket_url(cls, voice_id: str, config: Dict[str, Any] = None) -> str:
        """Generate WebSocket URL with Pro optimizations"""
        if config is None:
            config = cls.DEFAULT_CONFIG
        
        base_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        
        # Get configurations
        audio_quality = cls.get_audio_quality_config(config.get("audio_quality", "high"))
        model = cls.get_model_config(config.get("model", "turbo"))
        streaming = cls.get_streaming_config(config.get("streaming", "fast"))
        
        # Build parameters
        params = {
            "model_id": model["id"],
            "optimize_streaming_latency": streaming["optimize_streaming_latency"],
            "output_format": audio_quality["format"]
        }
        
        # Add audio quality if specified
        if audio_quality["bitrate"] != "128k":
            params["audio_quality"] = audio_quality["bitrate"]
        
        # Build URL
        param_str = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{param_str}"
    
    @classmethod
    def get_rest_api_config(cls, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get REST API configuration for Pro features"""
        if config is None:
            config = cls.DEFAULT_CONFIG
        
        audio_quality = cls.get_audio_quality_config(config.get("audio_quality", "high"))
        model = cls.get_model_config(config.get("model", "turbo"))
        voice_settings = cls.get_voice_settings(config.get("voice_settings", "balanced"))
        
        return {
            "model_id": model["id"],
            "voice_settings": voice_settings,
            "output_format": audio_quality["format"]
        }
    
    @classmethod
    def print_pro_features(cls):
        """Print all available Pro features"""
        print("🎉 ElevenLabs Pro Features Available:")
        print("=" * 50)
        
        print("\n📊 Subscription Limits:")
        print(f"   • Concurrent Requests: {cls.MAX_CONCURRENT_REQUESTS}")
        print(f"   • Monthly Credits: {cls.MONTHLY_CREDITS:,}")
        print(f"   • Additional Credits: ${cls.ADDITIONAL_CREDIT_COST}/1000")
        
        print("\n🎧 Audio Quality Options:")
        for key, config in cls.AUDIO_QUALITY_OPTIONS.items():
            print(f"   • {config['name']}: {config['description']}")
        
        print("\n⚡ Model Options:")
        for key, config in cls.MODEL_OPTIONS.items():
            print(f"   • {config['name']}: {config['description']}")
            print(f"     Best for: {config['best_for']}")
        
        print("\n🎭 Voice Settings Presets:")
        for key, config in cls.VOICE_SETTINGS.items():
            print(f"   • {key.title()}: {config['description']}")
        
        print("\n🌊 Streaming Optimization:")
        for key, config in cls.STREAMING_OPTIONS.items():
            print(f"   • {key.replace('_', ' ').title()}: {config['description']}")
            print(f"     Best for: {config['best_for']}")
        
        print("=" * 50)

# Environment-specific configurations
class DevelopmentConfig(ProConfig):
    """Development configuration with debugging enabled"""
    DEFAULT_CONFIG = {
        **ProConfig.DEFAULT_CONFIG,
        "enable_performance_monitoring": True,
        "log_latency_stats": True,
        "save_test_audio": True,
        "debug_mode": True
    }

class ProductionConfig(ProConfig):
    """Production configuration optimized for performance"""
    DEFAULT_CONFIG = {
        **ProConfig.DEFAULT_CONFIG,
        "enable_performance_monitoring": False,
        "log_latency_stats": False,
        "save_test_audio": False,
        "debug_mode": False,
        "audio_quality": "high",
        "model": "turbo",
        "streaming": "ultra_fast"
    }

# Usage example
if __name__ == "__main__":
    # Print all Pro features
    ProConfig.print_pro_features()
    
    # Example configurations
    print("\n🔧 Example Configurations:")
    
    # Development config
    dev_config = DevelopmentConfig.DEFAULT_CONFIG
    print(f"\nDevelopment:")
    print(f"   Audio Quality: {dev_config['audio_quality']}")
    print(f"   Model: {dev_config['model']}")
    print(f"   Debug Mode: {dev_config['debug_mode']}")
    
    # Production config
    prod_config = ProductionConfig.DEFAULT_CONFIG
    print(f"\nProduction:")
    print(f"   Audio Quality: {prod_config['audio_quality']}")
    print(f"   Model: {prod_config['model']}")
    print(f"   Debug Mode: {prod_config['debug_mode']}")
    
    # Example WebSocket URL
    voice_id = "21m00Tcm4TlvDq8ikWAM"
    ws_url = ProConfig.get_websocket_url(voice_id, dev_config)
    print(f"\n🌐 Example WebSocket URL:")
    print(f"   {ws_url}") 