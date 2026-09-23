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

仅运行 vLLM 时，将 `SERVICE_MODE=all` 改为 `SERVICE_MODE=vllm`。两种模式默认通过容器端口 `8000` 提供服务；网关模式的 vLLM 后端仅监听容器内部 `127.0.0.1:8001`。

需要从其他机器访问时，将端口映射改为 `-p 8000:8000`。切换模式前，先停止并删除同名容器。

平台只允许配置启动命令（CMD）时，可直接指定运行模式和容器监听端口。例如网关监听 `12345`：

```bash
python3 /opt/fireredasr2-vllm/scripts/entrypoint.py all --port 12345
```

如果平台将命令和参数分开填写，命令填 `python3`，参数依次填 `/opt/fireredasr2-vllm/scripts/entrypoint.py`、`all`、`--port`、`12345`。已有完整默认 CMD 的情况下，只需在末尾追加 `all --port 12345`。健康检查自动使用所选端口，平台的服务端口配置也应指向 `12345`。仅运行 vLLM 时将 `all` 改为 `vllm`；`all` 模式下 `8001` 保留给内部后端，不能作为网关端口。

也可以通过 CMD 的 `--env-file` 加载配置。例如镜像附带的模板：

```bash
python3 /opt/fireredasr2-vllm/scripts/entrypoint.py \
  --env-file /opt/fireredasr2-vllm/.env.example all --port 12345
```

自定义文件必须在容器内可读，可由平台挂载，或在构建镜像时复制非敏感配置；宿主机上的文件不会自动可见。配置可写 `SERVICE_MODE=all`、`SERVICE_PORT=12345` 及其他参数，这样 CMD 只需指定 `--env-file /容器内路径/.env`。每行使用 `KEY=VALUE`，值不加引号，注释独立成行；不执行 shell 或展开变量。配置优先级为：CMD 的模式及端口参数 > 配置文件 > 已有环境变量 > 代码默认值。API 密钥等敏感配置应通过平台运行时挂载的文件提供，不应写入镜像。如果由平台环境变量注入密钥，应从配置文件删除对应项，避免被文件中的值覆盖。

## 转写接口

### 网关模式

`POST /v1/audio/transcriptions` 使用 multipart 表单，音频必须为 **16 kHz、单声道、有符号 16 位小端 PCM**。网关不做重采样或声道转换。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `file` | 必填 | 单个文件；上传多条音频时重复此字段 |
| `uttid` | 自动生成 | 可重复提交，与文件一一对应；传入时必须非空且批内唯一 |
| `audio_format` | `wav` | 整批共用：`wav` 为 PCM16 WAV，`pcm` 为无文件头的 PCM16LE |
| `model` | `fireredasr2-llm` | 模型名称 |
| `response_format` | `json` | `json` 或 `verbose_json`，均返回统一的 JSON 结构 |

裸 PCM 的采样率、声道和编码由调用方保证。WAV 根据文件头校验参数，不合规音频的错误码为 400，不支持的 `audio_format` 返回 HTTP 422。接口接收完整文件，不支持 `stream=true`。

```bash
curl --fail-with-body --max-time 7200 \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -H 'Authorization: Bearer your-key' \
  -F 'file=@meeting.wav' \
  -F 'uttid=meeting_001' \
  -F 'response_format=verbose_json'
```

上传裸 PCM 时，将文件改为 `meeting.pcm`，并添加 `-F 'audio_format=pcm'`。

批量请求使用相同接口，重复提交 `file`：

```bash
curl --fail-with-body --max-time 7200 \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -H 'Authorization: Bearer your-key' \
  -F 'file=@a.wav' \
  -F 'uttid=audio_001' \
  -F 'file=@b.wav' \
  -F 'uttid=audio_002' \
  -F 'response_format=json'
```

单条和 batch 返回相同结构：成功项放在 `results`，失败项放在 `errors`，两个数组始终存在，各自按上传顺序排列。通过 `uttid` 匹配音频；省略时服务端生成 ID，同时返回从 0 开始的原始上传 `index` 和 `filename`。以下为部分失败示例：

```json
{
  "request_id": "...",
  "results": [{"uttid": "audio_001", "index": 0, "filename": "a.wav", "status_code": 200, "text": "转写内容"}],
  "errors": [{"uttid": "audio_002", "index": 1, "filename": "b.wav", "status_code": 400, "error": "Audio file is empty"}]
}
```

单条和 batch 处理完成后均返回 HTTP 200，**必须检查 `errors`，200 不代表全部文件成功**。全部成功时 `errors` 为空，全部失败时 `results` 为空。文件失败不影响同批其他文件；请求级错误（鉴权、参数、文件数、总大小或容量不足）返回非 200 状态和 `{"detail": "..."}`。

默认每批最多 32 个文件，所有文件合计不超过 256 MiB。每个网关 worker 默认最多接纳 256 条尚未完成的音频，包括等待、预处理和识别中的音频；剩余容量不足以接纳整个批次时，请求返回 429，不会只接纳其中一部分。每条音频完成后立即释放自己的容量，不必等待同批其他音频。

网关默认使用 1 个 Uvicorn worker，可通过 `GATEWAY_WORKERS` 设置为多个，共用对外端口和同一个 vLLM 后端。每个 worker 有独立的队列、预处理池和片段调度器，`QUEUE_CAPACITY`、`PREPROCESS_CONCURRENCY` 和 `BACKEND_CONCURRENCY` 均按每个 worker 计算，容量不共享。

每个 worker 默认将预处理交给最多 4 个子进程执行，子进程按需启动并复用，最多同时对 4 条音频读取和预处理；音频校验、PCM 提取、VAD 和切片规划在子进程中执行。请求取消时，已运行的预处理结束后才释放名额。

每个 worker 默认最多同时执行 8 个片段识别请求。已完成预处理的音频轮流提交片段；同一条长音频也可以并发识别多个片段，完成后按原顺序拼接。HTTP batch 与 GPU 推理 batch 独立，GPU 调度由 vLLM 负责。客户端断开连接后，网关清理该请求的待处理任务和后端调用。

`QUEUE_CAPACITY` 在上传解析后按音频条数限制接纳容量，不是全局内存字节限制；部署时需结合音频长度、上传大小和可用内存配置。

网关默认跳过静音，将语音切为最长 30 秒的片段，片段并发识别后按原顺序拼接；多个文件可并发处理。设置 `VAD_ENABLED=0` 可保留全部音频，仍按最长片段时长切分。纯静音返回空文本，任一片段识别失败则该文件返回错误。

`json` 的成功项包含转写文本及音频标识；`verbose_json` 在每个成功项中额外返回音频时长、片段、请求 ID、处理耗时和 RTF。片段时间戳表示切片边界，不是字级对齐。网关不提供纯文本响应。

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
| `GATEWAY_WORKERS` | `1` | 网关 worker 进程数，正整数，仅 all 模式生效 |
| `VAD_ENABLED` | `1` | 网关是否启用 VAD |
| `VAD_MODE` | `1` | 检测模式 0–3，数值越大越倾向判为非语音 |
| `CHUNK_SECONDS` | `30` | 网关切片最长秒数，范围为大于 0 且不超过 30 |
| `QUEUE_CAPACITY` | `256` | 每个 worker 已接纳但未完成的音频总数，容量不足时整个新请求返回 429 |
| `PREPROCESS_CONCURRENCY` | `4` | 每个 worker 的预处理进程池最大进程数，同时限制读取和预处理的音频数 |
| `BACKEND_CONCURRENCY` | `8` | 每个 worker 同时提交的片段识别数 |
| `MAX_COMPLETION_TOKENS` | `512` | 网关每段识别的输出 token 上限 |
| `MAX_UPLOAD_MB` | `256` | 网关单次请求所有文件合计大小上限，单位 MiB |
| `MAX_BATCH_FILES` | `32` | 网关单次请求的文件数量上限 |
| `MAX_AUDIO_SECONDS` | `3600` | 网关单文件时长上限，单位秒 |

升级旧配置时，删除 `MAX_ACTIVE_JOBS`，改用 `QUEUE_CAPACITY` 和 `PREPROCESS_CONCURRENCY`。保留旧变量会导致网关启动报错，避免旧限制被静默忽略。

## 客户端示例

[标准库客户端](examples/client.py) 无需额外安装依赖：

```bash
export ASR_API_KEY=your-key
python3 examples/client.py sample.wav
python3 examples/client.py meeting.wav --format verbose_json
python3 examples/client.py meeting.pcm --audio-format pcm --format verbose_json
python3 examples/client.py a.wav b.wav --uttid audio_001 --uttid audio_002 --format verbose_json
```

默认连接 `http://127.0.0.1:8000`，可通过 `--url` 指定服务地址。`verbose_json`、裸 PCM、多文件和 `uttid` 用于网关模式。单条或 batch 返回中有错误时，客户端打印完整响应并以非零状态退出。

单文件也可以使用 [OpenAI SDK 示例](examples/openai_client.py)，安装 `openai` 后运行。通过 `ASR_BASE_URL` 指定服务地址，裸 PCM 使用 `--audio-format pcm`，音频 ID 使用 `--uttid`。该示例读取原始 JSON 响应以支持网关的统一结构；直连 vLLM 仍返回原生格式。批量上传使用上述标准库客户端或 curl。

## 接口压测

[benchmark.py](examples/benchmark.py) 无需额外安装依赖，支持单条、batch 和多档并发压测。在项目根目录执行：

```bash
python3 examples/benchmark.py \
  --url http://127.0.0.1:12345/v1/audio/transcriptions \
  --manifest examples/example_manifest.jsonl \
  --concurrency 1 4 8 16 32 \
  --request-batch-size 1 2 \
  --requests 100 \
  --output benchmark.json
```

按[示例清单](examples/example_manifest.jsonl)准备音频，相对路径以清单目录为基准；也可用 `--audio a.wav b.wav` 替代 `--manifest`。鉴权读取 `ASR_API_KEY`。

上述命令测试 5 档 HTTP 并发 × 2 档 batch，每组 100 个请求，另有默认 3 次预热。报告统计成功率、音频吞吐及 P50/P95/P99 延迟，部分失败也计入错误。更多参数见 `--help`。

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
