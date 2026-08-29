# 优化说明

所有优化以代码默认值形式生效,不依赖环境变量或自定义 yaml(官方评测的 deploy yaml
会覆盖任何 perf yaml,代码默认值是唯一稳健面)。不修改模型权重。
每项均有逃生开关或配置项,可独立回退。

## 1. TF-prefill bypass(给定文本 TTS 的教师强制预填充)—— RTF 0.25 → 0.17 主力

commit `d2630f6`。`vllm_omni/engine/orchestrator.py`。

TTS 请求(user message 即为要朗读的文本)在 stage-0(Thinker)本需自回归生成整段
speech token 计划,再由 Talker/codec 合成。观察到:给定文本场景下,stage-0 的输出
完全由输入决定,自回归解码纯属重复劳动。

补丁在检测到 TTS 模板指纹(prompt 尾 token == `<|tts_bos|>`)且非全双工时:

- 把给定文本 tokenize 后直接拼入 prompt 尾部(`<|tts_bos|> text <|tts_eos|>`),
  stage-0 `max_tokens=1`(只需出 1 帧结束标记);
- Talker/codec 从 prefill hidden 状态起解码(因果 prefill 等价于教师强制),
  与原路径逐 span 对齐;
- 流式文本通道回显给定文本。

效果(官方 dfx 口径 zh/32/c1):median RTF 0.2511 → 0.1708;TTFT 276 → 60 ms;
TTFP 457 → 192 ms。zh 全量 2020 WER 1.39% → **1.21%**(消除 thinker 复打文本的
抄写/重复/截断错误)。

逃生:`MINICPMO_TTS_BYPASS=0` 恢复正常 thinker 解码路径。
非 TTS 请求(plain chat / Daily-Omni / Video-MME 形状)零命中,已验证隔离。

## 2. Stage input prep fast path

commit `cab8279`(B2a+B2b)。`_prepare_inputs` 的 per-request 配对快速路径,
stage1/2 的输入重组开销 −40%+。stage0(thinker)同路径,全量 Daily 逐题等价验证。

## 3. Slice A plumb-fast(stage 间张量搬运)

commit `de9d347`。hidden/codec 跨 stage 传递路径收窄,e2e −0.0025 RTF,
WER 三臂归因无回归。

## 4. Codec/Talker 解码默认值栈

- codec 采样链在 CPU 完成(去除逐帧 NPU 采样同步开销),默认开启;
- `codec_chunk_frames` 25 → 512(吐段大块出帧);
- `initial_codec_chunk_frames` 4(首音频块覆盖更多 mel 帧,摊薄首块固定成本);
- CFM vocoder `n_timesteps` 1(单步流匹配出 mel)。

## 5. 流式 TTFT 注入

文本 delta 逐 token 注入 SSE(FINAL_ONLY guard),TTFT 从首句完成提前到首 token,
不改音频流水线。

## 6. stage0 FULL cudagraph

修复 float NORMDIAG 诊断张量导致的构图中断,stage0 全图捕获,
thinker prefill/decode 步入图内。

## 7. Code2Wav prompt/setup 缓存

commit `1c2ad5f`。同参考音频的后续请求复用 prompt 特征与 CFM 初态
(owner-aware LRU 容量 4 + setup-state LRU 容量 1,输出逐位不变,
容量归零可恢复原生命周期)。官方 zh 评测集 2020 条含 1010 个 unique ref
且同 ref 相邻,c1 负载下约半数请求命中:audio TTFP 191.8 → 166-170 ms。

## 质量验证矩阵

| 验证 | 结果 |
|------|------|
| zh WER 全量 2020(bypass 路径) | 1.21% ≤ 1.56% |
| Daily-Omni 全量 1197(官方 file:// recipe) | 77.94% ≥ 77.5% |
| ASV en/zh(wavlm-base-plus) | 0.8678 / 0.8694 ≥ 0.689 |
| plain chat 隔离(bypass 零命中) | engaged 计数零漂移 |
