"""FastAPI 实时语音识别服务"""
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import websockets
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from asr_service import asr_service
from config import HOST, PORT, SUPPORTED_AUDIO_FORMATS, SAMPLE_RATE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="实时语音识别 API",
    description="基于 Faster Whisper 的实时语音识别服务，支持实时流式识别和文件上传识别",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 响应模型 ============
class TranscribeResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float


class StreamSegment(BaseModel):
    text: str
    start: float
    end: float
    is_final: bool = False


# ============ API 端点 ============

@app.get("/")
async def root():
    """返回测试页面"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return {"message": "ASR 服务已启动，请访问 /docs 查看 API 文档"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model": "faster-whisper"}


@app.post("/transcribe/file", response_model=TranscribeResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None)
):
    """
    上传音频文件进行识别
    
    - **file**: 音频文件 (支持 wav, mp3, m4a, flac, ogg, webm)
    - **language**: 语言代码 (如 zh, en)，不指定则自动检测
    """
    # 检查文件格式
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {file_ext}。支持格式: {SUPPORTED_AUDIO_FORMATS}"
        )
    
    # 读取文件内容
    content = await file.read()
    
    # 保存临时文件
    temp_path = Path(f"temp_{file.filename}")
    try:
        temp_path.write_bytes(content)
        
        # 转录
        result = asr_service.transcribe_file(str(temp_path), language)
        
        return TranscribeResponse(
            text=result["text"],
            language=result["language"],
            language_probability=result["language_probability"],
            duration=result["duration"]
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/transcribe/bytes")
async def transcribe_bytes(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None)
):
    """
    上传音频字节流进行识别（适用于前端录音）
    """
    content = await file.read()
    result = asr_service.transcribe_bytes(content, language)
    return JSONResponse(content=result)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket 实时语音识别端点
    
    客户端发送格式:
    - 配置消息: {"type": "config", "language": "zh"}
    - 音频数据: 二进制音频帧 (16kHz, 16bit, mono PCM)
    
    服务端返回格式:
    - {"type": "segment", "text": "...", "start": 0.0, "end": 1.0}
    - {"type": "final", "text": "完整文本"}
    """
    await websocket.accept()
    logger.info("WebSocket 连接已建立")
    
    audio_buffer = []
    language = None
    sample_rate = SAMPLE_RATE
    
    try:
        while True:
            # 接收消息
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # 接收音频数据
                    audio_data = message["bytes"]
                    audio_buffer.append(audio_data)
                    
                    # 当缓冲区累积到一定量时进行识别
                    total_bytes = sum(len(chunk) for chunk in audio_buffer)
                    # 约 3 秒的音频数据 (16000 * 2 * 3)
                    if total_bytes >= sample_rate * 2 * 3:
                        await process_audio_buffer(websocket, audio_buffer, language)
                        audio_buffer = []
                        
                elif "text" in message:
                    # 接收配置消息
                    try:
                        config = json.loads(message["text"])
                        if config.get("type") == "config":
                            language = config.get("language")
                            sample_rate = config.get("sample_rate", SAMPLE_RATE)
                            logger.info(f"配置更新: language={language}, sample_rate={sample_rate}")
                            await websocket.send_json({
                                "type": "config_ack",
                                "language": language,
                                "sample_rate": sample_rate
                            })
                    except json.JSONDecodeError:
                        logger.warning(f"无效的 JSON 消息: {message['text']}")
                        
            elif message["type"] == "websocket.disconnect":
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        await websocket.close()
    
    # 处理剩余的音频数据
    if audio_buffer:
        await process_audio_buffer(websocket, audio_buffer, language, is_final=True)


async def process_audio_buffer(
    websocket: WebSocket, 
    audio_buffer: list, 
    language: Optional[str],
    is_final: bool = False
):
    """处理音频缓冲区并发送识别结果"""
    try:
        # 合并音频数据
        audio_bytes = b"".join(audio_buffer)
        
        # 转换为 numpy 数组 (假设 16bit PCM)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        # 转录
        segments = list(asr_service.transcribe_numpy(audio_array, language))
        
        # 发送结果
        for segment in segments:
            await websocket.send_json({
                "type": "segment",
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
                "is_final": is_final
            })
            
        if is_final:
            full_text = "".join(s["text"] for s in segments)
            await websocket.send_json({
                "type": "final",
                "text": full_text
            })
            
    except Exception as e:
        logger.error(f"处理音频错误: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


# ============ 静态文件服务 ============
from fastapi.staticfiles import StaticFiles

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)