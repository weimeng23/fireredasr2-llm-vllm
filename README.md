# FireRedASR2-LLM · vLLM 0.29.0 · WebRTC VAD

一个 Dockerfile、一个镜像，两种运行模式；两种模式都只需启动一个普通 Docker 容器，无需 Compose。默认使用单 GPU、BF16，面向 RTX PRO 5000 Blackwell 48GB / 72GB。未在该 GPU 上实测容量或吞吐。

| 模式 | 容器内进程 | 对外接口 |
|---|---|---|
| `vllm`（默认） | vLLM | 容器 `8000`：原生转写接口 |
| `all` | WebRTC VAD 网关 + vLLM | 容器 `8000`：网关；vLLM 只监听 `127.0.0.1:8001` |

## 模型与环境

模型由使用者下载，项目不会自动下载：

- [allendou/FireRedASR2-LLM-vllm](https://huggingface.co/allendou/FireRedASR2-LLM-vllm)
- [固定快照 24078c33d69cafe365e343af5b1894548879707d](https://huggingface.co/allendou/FireRedASR2-LLM-vllm/tree/24078c33d69cafe365e343af5b1894548879707d)

完整模型约 33.5GB，需下载所有九个 safetensors 分片、配置和 tokenizer 等文件，不能只下载语言模型部分。不要将原始 FireRedTeam 权重目录直接替代这个转换版。运行时只读挂载并开启 Hugging Face 离线模式。

宿主机需要 Linux、Docker Engine、NVIDIA Container Toolkit 和支持该 GPU / CUDA 的 NVIDIA 驱动。默认镜像 `vllm/vllm-openai:v0.29.0` 使用 CUDA 13.0；CUDA 13.x 的驱动基线为 R580，具体 GPU 和内核仍需目标机验证。无需在宿主机另装相同版本的 CUDA Toolkit。

网关使用 CPU 版 `webrtcvad-wheels==2.0.14`（导入名 `webrtcvad`），不需要 GPU，也不用额外下载 VAD 模型。网关接收已转换好的 PCM16 或 WAV，使用 Python 标准库读取 WAV，不再解码其他音频格式或自动重采样。网关的 Python 依赖安装在 `/opt/gateway-venv`，与 vLLM 的依赖隔离。

网关仅接受 **16 kHz、单声道、有符号 16 位小端 PCM**：通过可选的 multipart 字段 `audio_format` 选择 `wav`（默认，有 WAV 文件头）或 `pcm`（无文件头的原始采样数据）。WAV 会校验实际文件头，错误采样率、声道数、位深、压缩编码或损坏的数据返回 400；不支持的字段值返回 422。裸 PCM 没有元数据，服务端只能校验非空、完整的双字节采样和按约定计算的时长，实际采样率、声道数、位深及字节序必须由调用方保证。该限制只作用于网关，直连 vLLM 的接口保持原样。

## 代码结构

- `Dockerfile`：vLLM 0.29.0 与网关依赖，默认通过 CMD 启动服务。
- `scripts/entrypoint.py`：选择运行模式，管理双服务的启动、等待和退出。
- `scripts/serve.py`：校验本地模型、构造 vLLM 启动参数。
- `gateway/app.py`：HTTP 接口、并发限制和后端请求。
- `gateway/audio.py`：PCM/WAV 校验、WebRTC VAD 和切片。
- `scripts/healthcheck.py`：容器健康检查。
- `.env.example`：可选配置；`examples/`：客户端请求示例，不打进镜像。

没有 Docker Compose、宿主机启动封装，也不安装 tini。

## 构建一次

在项目根目录执行：

```bash
docker build -t fireredasr2-vllm:0.29.0 .
```

构建用 BuildKit 的缓存挂载把 uv 下载的 wheel 挡在镜像层之外，需要 Docker 23 或更高版本；更老的版本需显式设置 `DOCKER_BUILDKIT=1`。镜像不安装任何 apt 包，网关读取 PCM/WAV 不依赖系统 FFmpeg。

镜像把 Ubuntu 主源改写为腾讯云镜像，供进入容器后临时安装工具使用；构建过程本身不装任何 apt 包。改回官方源或换成其他镜像时覆盖 `APT_MIRROR`：

```bash
docker build \
  --build-arg APT_MIRROR=http://archive.ubuntu.com/ubuntu/ \
  -t fireredasr2-vllm:0.29.0 .
```

该参数只改写 Ubuntu 主源。基础镜像里还有 deadsnakes PPA 和 NVIDIA CUDA 源，`apt-get update` 仍会访问它们。

基础镜像标签与 vLLM 音频扩展安装要求共用 `VLLM_VERSION`。网关依赖由 `requirements.txt` 独立管理，安装到单独的虚拟环境中。手动修改 `FROM` 时，需确保基础镜像中的 vLLM 版本与该参数一致。

0.29.0 是撰写时 vLLM 的最新发行版。将来升级需覆盖 `VLLM_VERSION`，但要先确认目标版本仍支持 FireRedASR2，并自行验证驱动与 GPU：

```bash
docker build --build-arg VLLM_VERSION=<版本号> -t fireredasr2-vllm:<版本号> .
```

需要私有仓库、镜像加速器或 `-cu129` 这类变体镜像时，请直接修改 Dockerfile 的 `FROM` 行。

镜像构建需要下载软件包，但不下载模型。以下运行命令中的 `/your/model/path` 必须替换为完整模型目录绝对路径。

## 模式一：仅运行 vLLM

```bash
docker run -d \
  --name fireredasr2-vllm \
  --restart unless-stopped --stop-timeout 35 \
  --gpus 'device=0' --shm-size 8g \
  -p 127.0.0.1:8000:8000 \
  -v /your/model/path:/models/fireredasr2:ro \
  -v fireredasr2-cache:/root/.cache \
  -e ASR_API_KEY=your-key \
  fireredasr2-vllm:0.29.0
```

默认 `SERVICE_MODE=vllm`。这里不启动网关、不运行 VAD；已有短音频或上游已完成切片时使用。

```bash
curl --fail-with-body http://127.0.0.1:8000/health
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

## 模式二：同一个容器内运行网关 + vLLM

若已存在同名容器，先执行 `docker stop fireredasr2-vllm` 和 `docker rm fireredasr2-vllm`，再执行下列命令。这是另一种启动方式，不需要同时运行模式一。

```bash
docker run -d \
  --name fireredasr2-vllm \
  --restart unless-stopped --stop-timeout 35 \
  --gpus 'device=0' --shm-size 8g \
  -p 127.0.0.1:8080:8000 \
  -v /your/model/path:/models/fireredasr2:ro \
  -v fireredasr2-cache:/root/.cache \
  -e ASR_API_KEY=your-key \
  -e SERVICE_MODE=all \
  -e VAD_MODE=1 \
  -e VAD_FRAME_MS=20 \
  -e VAD_SILENCE_MS=500 \
  -e VAD_PADDING_MS=200 \
  fireredasr2-vllm:0.29.0
```

通过 `-e SERVICE_MODE=all` 选择双服务模式。两种模式内部对外端口始终是 8000，此处选择映射到宿主机 8080。后端 8001 仅在容器内部使用。示例只绑定宿主机 loopback；需要从其他机器访问时，将 `-p 127.0.0.1:8080:8000` 改为 `-p 8080:8000`。

```bash
curl --fail-with-body http://127.0.0.1:8080/healthz
curl --fail-with-body --max-time 7200 \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  -H 'Authorization: Bearer your-key' \
  -F 'model=fireredasr2-llm' \
  -F 'file=@meeting.wav' \
  -F 'audio_format=wav' \
  -F 'response_format=verbose_json'
```

上传裸 PCM 时，使用同一个网关接口：

```bash
curl --fail-with-body --max-time 7200 \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  -H 'Authorization: Bearer your-key' \
  -F 'model=fireredasr2-llm' \
  -F 'file=@meeting.pcm;type=application/octet-stream' \
  -F 'audio_format=pcm' \
  -F 'response_format=verbose_json'
```

## 进入容器后手动启动

在宿主机执行下面的命令，镜像后面直接写 `bash`：

```bash
docker run --rm -it \
  --name fireredasr2-manual \
  --gpus 'device=0' --shm-size 8g --stop-timeout 35 \
  -p 127.0.0.1:8000:8000 \
  -v /your/model/path:/models/fireredasr2:ro \
  -v fireredasr2-cache:/root/.cache \
  -e ASR_API_KEY=your-key \
  fireredasr2-vllm:0.29.0 bash
```

容器工作目录为 `/opt/fireredasr2-vllm`。进入后选择一个启动命令，勿同时执行：

```bash
# 仅 vLLM
python3 scripts/entrypoint.py vllm

# 或：网关 + vLLM
python3 scripts/entrypoint.py all
```

可先 `export GPU_MEMORY_UTILIZATION=0.90` 等，再启动。服务在前台运行，按 `Ctrl+C` 停止后返回 Bash，可以调整参数再启动。手动调试时先停止服务再 `exit`；如果希望服务接替 Bash 运行，使用 `exec python3 scripts/entrypoint.py all`，服务退出时容器也会退出。

本模式没有启动服务前，健康检查不会通过；这是预期行为。若默认模式的容器已经启动，使用 `docker exec -it fireredasr2-vllm bash` 进入查看，不要重复启动占用相同端口的服务。

## CMD 与自定义启动命令

Dockerfile 清除基础镜像的 ENTRYPOINT，仅保留默认命令：

```dockerfile
ENTRYPOINT []
CMD ["python3", "/opt/fireredasr2-vllm/scripts/entrypoint.py"]
```

基础镜像只提供 `python3`，没有 `python` 这个命令，容器内的所有命令都需要写 `python3`。

默认命令读取 `SERVICE_MODE`，未设置时为 `vllm`。镜像名后面可直接写完整命令，例如 `bash` 或 `python3 /opt/fireredasr2-vllm/scripts/entrypoint.py all`。显式传给 Python 脚本的模式优先于环境变量。

注意：新版不能在镜像名后面只写 `all`；那会被当作可执行程序。使用 `-e SERVICE_MODE=all`，或写完整 Python 命令。

也可以进入容器后直接调用 vLLM，自行设置完整参数（与默认启动器二选一）：

```bash
vllm serve /models/fireredasr2 \
  --served-model-name fireredasr2-llm \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --max-num-queued-reqs 64 \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"audio":1}' \
  --api-key your-key
```

直接运行 `vllm serve` 时，项目的 `GPU_MEMORY_UTILIZATION`、`ASR_API_KEY` 等环境变量不会自动转换成命令行参数，需自行传入；不会启动网关和 VAD。

## 网关行为与接口

网关处理流程：完整文件上传 → 按 `audio_format` 校验并提取 16 kHz 单声道 PCM16 → WebRTC VAD → 语音分段 → 逐段包装为 WAV 并调用内部 vLLM → 按原始顺序拼接文本。`audio_format` 只用于网关解析输入，不透传给 vLLM；不根据文件扩展名或 MIME 类型自动转换。

音频校验、VAD 和切片在内存中进行，PCM 每秒占 32,000 字节；例如 10 分钟音频、8 个任务的 PCM 合计约 146.5 MiB。实际峰值还包括上传缓存、解析过程和临时副本，FastAPI 的 `UploadFile` 解析大文件时可能使用临时磁盘。`MAX_ACTIVE_JOBS` 在上传与表单解析后才检查，不能限制同时上传的总内存。切片在发送前才生成，同时驻留的切片数受 `BACKEND_CONCURRENCY` 限制。

- 使用 10 / 20 / 30ms 帧判断语音，默认 20ms。每份录音独立创建 VAD 实例，避免并发请求污染状态。
- 默认连续非语音达到 500ms 后断句；短停顿合并，不按单帧判断直接剪音频。
- 每段语音前后保留 200ms 缓冲；重叠缓冲区合并，避免重复识别同一段音频。
- 末尾不足一帧只在 VAD 判断时补零，切出的音频不补长、不丢尾部；检测到的短语音不因最小时长而被删除。
- 连续语音超过 `CHUNK_SECONDS=25` 时按采样点硬切，不插入重叠；没有自然停顿时仍可能切断词句。
- 判为非语音的区间跳过 ASR。纯非语音文件返回空文本；误判可能漏掉弱语音，需用业务录音验收。`VAD_MODE=3` 更激进，不代表更准确。
- 若输入已分好句或希望保留所有音频，可设 `VAD_ENABLED=0`；此时仍按最长 25 秒切片，且不会跳过任何区间。

`verbose_json` 为网关自定义扩展：包含 duration、segments、request_id、elapsed_seconds、rtf、vad。每个 segment 包含 start/end/text/skipped_silence；`skipped_silence` 为兼容旧接口保留的字段名，现在表示被 WebRTC 判为非语音。时间是原始录音中的切片边界，不是字级对齐；开启 VAD 时 `timestamp_type=vad_chunk_boundaries`。

也支持 `response_format=json` / `text`。`stream=true` 返回 400；裸 PCM 同样必须完整上传，不是实时音频流。没有多文件 batch 入口。网关固定传 `temperature=0`、`repetition_penalty=1.0`，输出 token 上限来自 `MAX_COMPLETION_TOKENS` 环境变量，不透传调用方的采样参数。直连 vLLM 时可自行设置请求参数。

## 配置

使用 `docker run -e KEY=value` 设置参数，或复制 `.env.example` 为 `.env`，通过 `docker run --env-file .env ...` 传入。模型挂载、GPU 和端口仍由 Docker 命令参数指定。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `SERVICE_MODE` | `vllm` | 默认启动命令选择 `vllm` / `all` |
| `MODEL_PATH` | `/models/fireredasr2` | 容器内模型路径 |
| `DTYPE` | `bfloat16` | vLLM 精度 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | vLLM 显存预算 |
| `MAX_MODEL_LEN` | `4096` | 输入音频 token、文本提示及输出的总上下文上限，不是秒数 |
| `MAX_NUM_SEQS` | `8` | 每轮调度序列上限，不是实测性能上限 |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | 每轮 token 预算 |
| `MAX_NUM_QUEUED_REQS` | `64` | vLLM 等待和运行请求总数限制 |
| `ENFORCE_EAGER` | `0` | 设为 1 可关闭 CUDA Graph 进行排障 |
| `MM_PROCESSOR_CACHE_GB` | `0` | 当前关闭多模态预处理缓存，后续可压测调优 |
| `ASR_API_KEY` | 空 | 对外 API 密钥，两种模式均可用 |
| `STARTUP_TIMEOUT_SECONDS` | `900` | 双服务模式等待后端就绪超时 |
| `SHUTDOWN_TIMEOUT_SECONDS` | `25` | 双服务模式退出清理宽限期；Docker stop timeout 需更大 |
| `VAD_ENABLED` | `1` | 仅网关：是否进行 WebRTC 检测 |
| `VAD_MODE` | `1` | 仅网关：0–3，越高越激进 |
| `VAD_FRAME_MS` | `20` | 仅网关：10、20 或 30 |
| `VAD_SILENCE_MS` | `500` | 仅网关：断句所需连续非语音时长 |
| `VAD_PADDING_MS` | `200` | 仅网关：语音首尾缓冲 |
| `CHUNK_SECONDS` | `25` | 仅网关：最长切片，必须大于 0 且不超过 30 秒 |
| `MAX_ACTIVE_JOBS` | `8` | 仅网关：活动转写任务上限，超出返回 429 |
| `BACKEND_CONCURRENCY` | `8` | 仅网关：跨请求的后端调用并发上限 |
| `MAX_COMPLETION_TOKENS` | `512` | 仅网关：每段输出上限，需验证高语速样本是否截断 |
| `MAX_UPLOAD_MB` | `256` | 仅网关：单文件上限 MiB，另留 1 MiB multipart 预算；上传整体驻留内存 |
| `MAX_AUDIO_SECONDS` | `3600` | 仅网关：最长文件时长 |
| `MEDIA_TIMEOUT_SECONDS` | `180` | 仅网关：VAD 检测超时 |
| `BACKEND_TIMEOUT_SECONDS` | `120` | 仅网关：每段 ASR 调用超时 |

单份文件中的片段顺序识别，多份文件可并发。任何 ASR 片段失败均返回错误，不将部分结果伪装成完整转写。`/health` 在两种模式均可用，网关额外提供 `/healthz`，且会检查后端健康。网关使用一个 worker，以保证进程内全局并发限制生效。

25 秒是项目的切片策略，不是模型官方硬上限。原始模型卡写的是最多 40 秒，本转换版预处理配置的窗口为 30 秒；升级或改变该参数仍需实测。

## 进程管理与运维

默认由 CMD 直接启动 Python。仅 vLLM 模式通过 exec 将进程替换为 vLLM；双服务模式由 `entrypoint.py` 作为主进程，先启动 vLLM，后端健康后才启动网关。后端启动超时、任一服务退出都会清理两个进程组并以非零状态退出，配合 Docker 重启策略恢复。双服务模式收到 SIGTERM / SIGINT 时，会向两个进程组发送 SIGTERM，等待宽限期后清理未退出的进程，并回收直接子进程。Docker 的 stop timeout 必须大于内部清理宽限期。

这里没有实现通用 init 的孤儿进程回收功能；需要时可自行在 `docker run` 中加 `--init`，默认启动不依赖它。

```bash
docker logs -f --tail 100 fireredasr2-vllm
docker stop -t 35 fireredasr2-vllm
docker rm fireredasr2-vllm
```

健康检查不能证明 ASR 推理不会挂起；Docker 的 unhealthy 状态本身不会触发 restart。上线需配合真实语音探针与外部监控。双服务模式下后端仅监听 loopback，可另设 `BACKEND_API_KEY`，网关会自动携带它访问内部服务；公开鉴权使用 `ASR_API_KEY`。日志中启动命令会遮盖后端密钥。

客户端示例：

```bash
export ASR_API_KEY=your-key
python3 examples/client.py sample.wav
python3 examples/client.py meeting.wav --url http://127.0.0.1:8080 --audio-format wav --format verbose_json
python3 examples/client.py meeting.pcm --url http://127.0.0.1:8080 --audio-format pcm --format verbose_json
```

`examples/openai_client.py` 同样支持 `--audio-format pcm` / `wav`，通过 SDK 的 `extra_body` 发送该字段。省略时网关按 WAV 解析；直连 vLLM 时无需此字段。

## 验证范围

仓库当前不包含自动化测试。`pytest.ini` 与 `requirements-dev.txt` 保留，补充测试后可直接在 `tests/` 下运行：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

当前工作环境没有 Docker 和 NVIDIA GPU，未执行镜像构建或真实模型推理。不能将本地测试结果等同于 PRO 5000 上的识别效果、性能或显存验收。

参考：[vLLM 0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)、[FireRedASR2 适配](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/fireredasr2.py)、[WebRTC Python 封装](https://github.com/wiseman/py-webrtcvad)、[webrtcvad-wheels](https://pypi.org/project/webrtcvad-wheels/)。
