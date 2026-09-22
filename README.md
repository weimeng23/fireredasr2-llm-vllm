# FireRedASR2-LLM · vLLM · WebRTC VAD

基于 vLLM 0.29.0 的语音转写服务，支持直接推理，以及通过 WebRTC VAD 对长音频分段转写。单个 Docker 镜像提供两种运行模式，默认使用单 GPU、BF16。

| `SERVICE_MODE` | 功能 |
|---|---|
| `vllm`（默认） | 直接调用 vLLM，适合短音频或已完成切片的音频 |
| `all` | 网关 + vLLM，支持 VAD、长音频分段和结果拼接 |

## 环境与模型

- Linux、Docker Engine（启用 BuildKit）、NVIDIA Container Toolkit。
- NVIDIA GPU，以及兼容基础镜像 CUDA 13.0 的驱动。
- 下载完整的 [FireRedASR2-LLM-vllm 模型](https://huggingface.co/allendou/FireRedASR2-LLM-vllm)，约 33.5 GB，包含全部九个权重分片、配置和 tokenizer 文件。可使用[固定快照](https://huggingface.co/allendou/FireRedASR2-LLM-vllm/tree/24078c33d69cafe365e343af5b1894548879707d)。

模型通过本地目录只读挂载，服务以离线模式加载。请使用上述转换版模型，不要直接挂载 FireRedTeam 原始权重。

## 构建与启动

```bash
docker build -t fireredasr2-vllm:0.29.0 .
```

下面以网关模式启动。将 `/your/model/path` 替换为完整模型目录，`your-key` 替换为自己的 API 密钥：

```bash
docker run -d \
  --name fireredasr2-vllm \
  --restart unless-stopped --stop-timeout 35 \
  --gpus 'device=0' --shm-size 8g \
  -p 127.0.0.1:8000:8000 \
  -v /your/model/path:/models/fireredasr2:ro \
  -v fireredasr2-cache:/root/.cache \
  -e SERVICE_MODE=all \
  -e ASR_API_KEY=your-key \
  fireredasr2-vllm:0.29.0
```

仅运行 vLLM 时，将 `SERVICE_MODE=all` 改为 `SERVICE_MODE=vllm`。两种模式均通过容器端口 `8000` 提供服务；网关模式的 vLLM 后端仅监听容器内部 `127.0.0.1:8001`。

需要从其他机器访问时，将端口映射改为 `-p 8000:8000`。切换模式前，先停止并删除同名容器。

## 转写接口

### 网关模式

`POST /v1/audio/transcriptions` 使用 multipart 表单，音频必须为 **16 kHz、单声道、有符号 16 位小端 PCM**。网关不做重采样或声道转换。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `file` | 必填 | 上传音频文件 |
| `audio_format` | `wav` | `wav`：PCM16 WAV；`pcm`：无文件头的 PCM16LE |
| `model` | `fireredasr2-llm` | 模型名称 |
| `response_format` | `json` | `json`、`text` 或 `verbose_json` |

裸 PCM 的采样率、声道和编码由调用方保证。WAV 根据文件头校验参数，不合规音频返回 400，不支持的 `audio_format` 返回 422。接口接收完整文件，不支持 `stream=true`。

```bash
curl --fail-with-body --max-time 7200 \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -H 'Authorization: Bearer your-key' \
  -F 'file=@meeting.wav' \
  -F 'response_format=verbose_json'
```

上传裸 PCM 时，将文件改为 `meeting.pcm`，并添加 `-F 'audio_format=pcm'`。

网关默认跳过静音，将语音切为最长 25 秒的片段，逐段识别后按顺序拼接；多个文件可并发处理。设置 `VAD_ENABLED=0` 可保留全部音频，仍按最长片段时长切分。纯静音返回空文本，任一片段识别失败则返回错误。

`json` 返回 `{"text":"转写内容"}`；`text` 返回纯文本；`verbose_json` 额外返回音频时长、片段、请求 ID、处理耗时和 RTF。片段时间戳表示切片边界，不是字级对齐。

网关固定使用 `temperature=0`、`repetition_penalty=1.0`，输出长度通过 `MAX_COMPLETION_TOKENS` 设置。

### 直连 vLLM 模式

使用 vLLM 原生接口，无需传入 `audio_format`：

```bash
curl --fail-with-body --max-time 180 \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -H 'Authorization: Bearer your-key' \
  -F 'model=fireredasr2-llm' \
  -F 'file=@sample.wav' \
  -F 'response_format=json' \
  -F 'temperature=0' \
  -F 'repetition_penalty=1.0' \
  -F 'max_completion_tokens=512'
```

## 常用配置

通过 `docker run -e KEY=value` 设置，或复制 [.env.example](.env.example) 为 `.env`，使用 `--env-file .env` 加载。其余参数见该文件。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `ASR_API_KEY` | 空 | 对外 API 密钥，空值表示不启用鉴权 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | vLLM 显存预算比例 |
| `MAX_MODEL_LEN` | `4096` | 单次请求的总上下文 token 上限 |
| `MAX_NUM_SEQS` | `8` | vLLM 每轮调度的序列上限 |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | vLLM 每轮处理的 token 预算 |
| `MAX_NUM_QUEUED_REQS` | `64` | vLLM 等待和运行的请求总数上限 |
| `VAD_ENABLED` | `1` | 网关是否启用 VAD |
| `VAD_MODE` | `1` | 检测模式 0–3，数值越大越倾向判为非语音 |
| `CHUNK_SECONDS` | `25` | 网关切片最长秒数，范围为大于 0 且不超过 30 |
| `MAX_ACTIVE_JOBS` | `8` | 网关活动文件任务数，超限返回 429 |
| `BACKEND_CONCURRENCY` | `8` | 网关同时提交的片段识别数 |
| `MAX_COMPLETION_TOKENS` | `512` | 网关每段识别的输出 token 上限 |
| `MAX_UPLOAD_MB` | `256` | 网关单文件大小上限，单位 MiB |
| `MAX_AUDIO_SECONDS` | `3600` | 网关单文件时长上限，单位秒 |

## 客户端示例

[标准库客户端](examples/client.py) 无需额外安装依赖：

```bash
export ASR_API_KEY=your-key
python3 examples/client.py sample.wav
python3 examples/client.py meeting.wav --format verbose_json
python3 examples/client.py meeting.pcm --audio-format pcm --format verbose_json
```

默认连接 `http://127.0.0.1:8000`，可通过 `--url` 指定服务地址。`verbose_json` 和裸 PCM 示例用于网关模式。

也可以使用 [OpenAI SDK 示例](examples/openai_client.py)，安装 `openai` 后运行。通过 `ASR_BASE_URL` 指定服务地址，裸 PCM 使用 `--audio-format pcm`。

## 服务管理

```bash
curl --fail-with-body http://127.0.0.1:8000/health
docker logs -f --tail 100 fireredasr2-vllm
docker exec -it fireredasr2-vllm bash
docker stop -t 35 fireredasr2-vllm
docker rm fireredasr2-vllm
```

两种模式均提供 `/health`；网关另提供 `/healthz`，同时检查后端状态。

需要手动启动时，在启动示例中移除 `-d` 和 `--restart unless-stopped`，加入 `--rm -it`，并在镜像名后追加 `bash`。进入容器后执行 `python3 scripts/entrypoint.py all` 或 `python3 scripts/entrypoint.py vllm`。已运行的容器无需重复启动服务。
