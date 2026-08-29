# 提交信息总览

## 队伍信息
- 队伍名:升龙拳
- 参赛报名名:升龙拳
- GitHub:imkingjh999
- Fork 地址:https://github.com/imkingjh999/vllm-omni
- 分支:minicpm-challenge
- 提交面 commit 链(基于官方 minicpm-challenge,各项见 docs/OPTIMIZATION.md):
  - `de9d347` plumb-fast → `cab8279` prep-fast → `d2630f6` TF-prefill bypass → `1c2ad5f` Code2Wav caches → `2b43522` submission docs → `0bc5323` assistant-prompt bypass fix(代码面 HEAD；本 docs 追补 commit 紧随其后)

## 提交定位
基于官方 minicpm-challenge 分支的**推理性能优化**。全部优化以**代码默认值**形式生效:
官方评测用外部 deploy yaml 启动(perf yaml 会被覆盖),因此所有优化必须从代码默认值激活,
不依赖任何环境变量或自定义 yaml。不修改模型权重。

## 精度结果(全部按官方口径实测)

| Benchmark | 指标 | 实测值 | 官方基线 | 准入阈值 | 状态 |
|-----------|------|--------|----------|----------|------|
| Seed-TTS (zh, 全量 2020) | WER | **1.21%** | 1.414% | ≤1.56% | 通过 |
| Seed-TTS | SIM (wavlm-base-plus) | 0.8379 | 0.709 | ≥0.689 | 通过 |
| Daily-Omni (全量 1197, 官方 file:// recipe) | Accuracy | **77.94%** | 79.5% | ≥77.5% | 通过 |
| Video-MME (均衡集) | Accuracy | 67.78% | 69.0% | ≥67.0% | 通过 |

注:WER 1.21% 优于官方基线与正常路径(1.39%),来源见 OPTIMIZATION.md 第 1 条
(TF-prefill 消除了 thinker AR 复打文本的抄写错误)。
SIM 为同栈早期读数(hidden 同源,余量 0.87 vs 0.689)。

## 性能结果(官方 dfx perf 口径:seed-tts zh, 32 prompts, c1, --disable-shuffle)

| 指标 | 实测值 | 官方基线 | 倍数 |
|------|--------|----------|------|
| audio_rtf median | **0.1689** | 0.4423 | 2.62× |
| audio_rtf mean | 0.1690 | — | — |
| audio_rtf p99 | 0.1847 | — | — |
| ttft median | **60–64 ms** | 333.27 ms | 5.2× |
| audio_ttfp median | **169.7 ms** | 986.47 ms | 5.8× |

## Demo
- 视频:TODO(待录制,B 站链接)
- 启动方式:见 demo/README.md
