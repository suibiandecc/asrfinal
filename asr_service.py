"""ASR 服务封装"""
import io
import os
import tempfile
import logging
from typing import Optional, Generator, AsyncGenerator
from faster_whisper import WhisperModel
import numpy as np

# 设置 Hugging Face 镜像（国内加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    SAMPLE_RATE
)

logger = logging.getLogger(__name__)


class ASRService:
    """语音识别服务"""
    
    _instance: Optional["ASRService"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        logger.info(f"正在加载 Whisper 模型: {WHISPER_MODEL_SIZE}")
        self.model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        self._initialized = True
        logger.info("Whisper 模型加载完成")
    
    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码 (如 "zh", "en")，None 为自动检测
        
        Returns:
            包含转录结果的字典
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # 合并所有片段
        text = "".join(segment.text for segment in segments)
        
        return {
            "text": text.strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration
        }
    
    def transcribe_bytes(
        self, 
        audio_bytes: bytes, 
        language: Optional[str] = None
    ) -> dict:
        """
        转录音频字节数据
        
        Args:
            audio_bytes: 音频字节数据
            language: 语言代码
        
        Returns:
            转录结果字典
        """
        # 使用临时文件处理
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            return self.transcribe_file(tmp_path, language)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def transcribe_stream(
        self, 
        audio_stream: Generator[bytes, None, None],
        language: Optional[str] = None
    ) -> Generator[dict, None, None]:
        """
        流式转录音频
        
        Args:
            audio_stream: 音频字节流生成器
            language: 语言代码
        
        Yields:
            转录片段结果
        """
        # 收集音频数据
        audio_data = b"".join(audio_stream)
        
        # 转录
        segments, info = self.model.transcribe(
            io.BytesIO(audio_data),
            language=language,
            beam_size=5,
            vad_filter=True
        )
        
        for segment in segments:
            yield {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "confidence": getattr(segment, 'avg_logprob', None)
            }
    
    def transcribe_numpy(
        self,
        audio_array: np.ndarray,
        language: Optional[str] = None
    ) -> Generator[dict, None, None]:
        """
        转录 numpy 音频数组
        
        Args:
            audio_array: 音频数据数组 (float32, 16kHz)
            language: 语言代码
        
        Yields:
            转录片段结果
        """
        segments, info = self.model.transcribe(
            audio_array,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        
        for segment in segments:
            yield {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            }


# 全局单例
asr_service = ASRService()