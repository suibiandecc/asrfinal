"""配置文件"""
import os

# Whisper 模型配置
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # tiny, base, small, medium, large-v2, large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # cuda 或 cpu
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # float16, int8, float32

# 服务器配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# 音频配置
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# 支持的音频格式
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"]