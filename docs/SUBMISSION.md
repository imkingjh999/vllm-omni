# 提交信息总览

## 队伍信息
- 队伍名:升龙拳
- 参赛报名名:升龙拳
- GitHub:imkingjh999
- Fork 地址:https://github.com/imkingjh999/vllm-omni
- 分支:minicpm-challenge
- 提交面 commit 链(基于官方 minicpm-challenge,各项见 docs/OPTIMIZATION.md):
  - `de9d347` plumb-fast → `cab8279` prep-fast → `d2630f6` TF-prefill bypass → `1c2ad5f` Code2Wav caches → `2b43522` submission docs → `0bc5323` assistant-prompt bypass fix → `751865b` docs 追补 → `0f21d91` stage runners 收尾包(默认开) → `5a6e8c7` stage1 FIA pad-to-bucket(默认开,代码面 HEAD)

## 提交定位
基于官方 minicpm-challenge 分支的**推理性能优化**。全部优化以**代码默认值**形式生效:
官方评测用外部 deploy yaml 启动(perf yaml 会被覆盖),因此所有优化必须从代码默认值激活,
不依赖任何环境变量或自定义 yaml。不修改模型权重。

## 精度结果(全部按官方口径实测)

| Benchmark | 指标 | 实测值 | 官方基线 | 准入阈值 | 状态 |
|-----------|------|--------|----------|----------|------|
| Seed-TTS (zh, 全量 2020) | WER | **1.21%** | 1.414% | ≤1.56% | 通过 |
| Seed-TTS | SIM (wavlm-base-plus) | 0.8382 | 0.709 | ≥0.689 | 通过 |
| Daily-Omni (全量 1197, 官方 file:// recipe) | Accuracy | **77.94%** | 79.5% | ≥77.5% | 通过 |
| Video-MME (均衡集) | Accuracy | 67.78% | 69.0% | ≥67.0% | 通过 |

注:WER 1.21% 优于官方基线与正常路径(1.39%),来源见 OPTIMIZATION.md 第 1 条
(TF-prefill 消除了 thinker AR 复打文本的抄写错误);HEAD `5a6e8c7`(FIA pad-to-bucket)
全量 2020 复测仍为 1.21%,与关闭该优化的读数无差。
SIM 为同栈复测读数(hidden 同源,余量 0.87 vs 0.689)。

## 性能结果(seed-tts 32 prompts, c1, --disable-shuffle;命令见 docs/REPRODUCE.md 5.1)

zh 口径(HEAD `5a6e8c7` 两轮):

| 指标 | 实测值 | 官方基线 | 倍数 |
|------|--------|----------|------|
| audio_rtf median | **0.1545 / 0.1558** | 0.4423 | 2.85× |
| audio_rtf mean | 0.1552 / 0.1553 | — | — |
| audio_rtf p99 | 0.1705 / 0.1718 | — | — |
| ttft median | **59.2 / 59.6 ms** | 333.27 ms | 5.6× |
| audio_ttfp median | **156.6 / 156.7 ms** | 986.47 ms | 6.3× |

en 口径(HEAD `5a6e8c7`):median RTF **0.1567**,TTFP 158.5 ms(对照关掉 5a6e8c7
的 0.1685/0.1708,单此一项 −0.013)。
链上各阶段 zh median RTF:0.2511(d2630f6 前)→ 0.1708(d2630f6)→ 0.1658(0f21d91)
→ **0.1545-0.1558**(5a6e8c7)。

## Demo
- 视频:TODO(待录制,B 站链接)
- 启动方式:见 demo/README.md
