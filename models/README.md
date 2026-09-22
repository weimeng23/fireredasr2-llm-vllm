# 模型由使用者下载

模型地址：[allendou/FireRedASR2-LLM-vllm](https://huggingface.co/allendou/FireRedASR2-LLM-vllm)

固定快照：[24078c33d69cafe365e343af5b1894548879707d](https://huggingface.co/allendou/FireRedASR2-LLM-vllm/tree/24078c33d69cafe365e343af5b1894548879707d)

将完整内容下载到任意本地目录，再通过 `docker run --mount` 只读挂载到容器的 `/models/fireredasr2`。模型约 33.5 GB，包含九个 safetensors 分片和 tokenizer、config、preprocessor、CMVN 等文件。不要只下载语言模型部分，也不要用 Git LFS 指针文件替代真正权重。

本项目和镜像构建都不下载模型；vLLM 容器设置了 Hugging Face 离线模式。原始 FireRedTeam 权重目录与本文转换版的加载路径不同。
