# 实时语音识别系统

基于 Faster Whisper + FastAPI 实现的实时语音识别系统，支持文件上传识别和 WebSocket 实时流式识别。

## 功能特性

- ✅ **文件上传识别** - 支持 WAV、MP3、M4A、FLAC、OGG、WEBM 等格式
- ✅ **实时流式识别** - 通过 WebSocket 实现低延迟语音转写
- ✅ **多语言支持** - 支持中文、英文、日文、韩文等多种语言自动检测
- ✅ **GPU 加速** - 支持 CUDA 加速，显著提升识别效率
- ✅ **友好界面** - 提供 Web 前端测试页面

## 技术栈

- **框架**: FastAPI + Uvicorn
- **语音识别**: Faster Whisper
- **实时通信**: WebSocket
- **GPU 加速**: CUDA 12.x
- **前端**: HTML5 + JavaScript (Web Audio API)

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.x（可选，用于 GPU 加速）
- NVIDIA GPU（可选）

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动服务

```bash
# 默认配置（CPU模式）
python main.py

# 使用 GPU（需要安装 CUDA）
# 确保 config.py 中配置了 device=cuda
python main.py
```

### 访问服务

- **测试页面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## API 接口

### 文件上传识别

```bash
curl -X POST "http://localhost:8000/transcribe/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.wav" \
  -F "language=zh"
```

### WebSocket 实时识别

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/transcribe');
ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'config', language: 'zh' }));
    // 发送音频数据...
};
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.text);
};
```

## 项目结构

```
ASRf/
├── config.py          # 配置文件
├── asr_service.py     # ASR 服务封装（单例模式）
├── main.py            # FastAPI 主程序
├── requirements.txt   # 依赖文件
├── README.md          # 项目说明
├── .gitignore         # Git 忽略配置
└── static/
    └── index.html     # 前端测试页面
```

## 配置说明

在 `config.py` 中可以修改以下配置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `WHISPER_MODEL_SIZE` | `small` | 模型大小：tiny/base/small/medium/large-v3 |
| `WHISPER_DEVICE` | `cuda` | 计算设备：cuda/cpu |
| `WHISPER_COMPUTE_TYPE` | `float16` | 计算精度：float16/int8/float32 |
| `PORT` | `8000` | 服务端口 |

## 性能对比

| 模型 | 设备 | 10秒音频耗时 | 内存占用 |
|------|------|-------------|---------|
| small | CPU | ~8.5秒 | ~2GB |
| small | GPU | ~1.2秒 | ~4GB |
| base | CPU | ~12.3秒 | ~3GB |
| base | GPU | ~1.8秒 | ~5GB |

## 使用示例

### 实时录音识别

1. 访问 http://localhost:8000
2. 选择识别语言（或自动检测）
3. 点击"开始录音"按钮
4. 开始说话，实时显示识别结果
5. 点击"停止录音"结束

### 文件上传识别

1. 访问 http://localhost:8000
2. 选择识别语言
3. 点击"选择音频文件"，选择本地音频文件
4. 点击"开始识别"
5. 查看识别结果

## 注意事项

1. **首次运行**：首次启动会自动下载模型文件，可能需要几分钟
2. **CUDA 加速**：需要安装 CUDA 12.x 才能使用 GPU
3. **网络环境**：国内用户建议配置 HF_ENDPOINT 环境变量
4. **模型选择**：模型越大准确率越高，但速度越慢，内存占用越大

## 许可证

MIT License

## 参考文献

- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI Whisper](https://openai.com/research/whisper)