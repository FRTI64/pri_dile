# Triple-v4：规则策略的 Protocol v4 定向优化

> **Status:** 已注册规则候选（`triple-v4` / `Triple-v4`），**未 promotion**。
> **协议：** Protocol v4 dev（30 fields × 2048）+ 独立 seed = 20261029；
> test_v7 仅 provisional。
> **日期：** 2026-10-29。

## 结论

在 Triple-majority 的 Final 组件上加入一条窄的“噪声互惠者”分支后，
规则候选在 4/5 个 dev tier 上稳定提升：

| Tier | Triple-majority | Triple-v4 | Δ |
|---|---:|---:|---:|
| dev_param | 536.87 | **538.66** | **+1.79** |
| dev_family | 583.06 | **583.07** | +0.01 |
| dev_complex | 441.17 | **447.08** | **+5.91** |
| dev_axelrod | 566.93 | **568.38** | **+1.45** |
| dev_ood | 511.62 | **514.41** | **+2.79** |

独立 seed 20261029 复现：dev_param +1.72、dev_complex +5.78、
dev_ood +2.92。`dev_level4` +11.92、`dev_complexity4` +9.04；
test_v7 provisional +2.14（不参与选模）。

## 机制

Triple-majority 的 Final 在对手“偶尔 D、但被我们 D 后强烈报复”时，
会进入惩罚/和平测试循环。`triple-v4` 在统计量满足

```text
p_d >= 0.70, 0.20 <= p_c <= 0.62, p_d - p_c >= 0.30,
d_rate >= 0.15, n_c >= MIN_C, n_d >= MIN_D
```

时，把对手视为噪声互惠者：对手刚 C 则 C；刚 D 则以 0.75 概率原谅。
该条件不会命中干净 TFT（`p_c≈0`）、AllD（`p_c≈p_d`）或时间表型
late defector（`p_c≈p_d`），也没有新增主动 probe，因此 `dev_family`
的 `DelayedGrim` / `MirrorAdaptive` 不受影响。

## 收益来源

- `pattern_breaker`：dev_complex +65.7、dev_ood +77.9；
- `lookahead`：dev_complex +18.5、dev_ood +15.0；
- `extort` / `noisy` / `temporal` / `adaptive`：+3～6；
- `adaptive_mix` 在 dev_ood -2.3、`portfolio` -1.0，是主要代价；
- `min_norm` 0.992–1.000，没有 worst-family 崩塌。

## 判定

**注册为规则候选，不 promotion。** 它是最佳纯规则候选，但 sample-mode
`dev_param` 仍低于 RL-Repair（547.96）、RL-Wide160（548.26）和
RL-PPO（540.35）；只比 RL-PPO 低约 1.7，不能替代学习策略。最大单项
收益来自合成 `pattern_breaker` / `lookahead`；真实 300 人赛场的迁移
未验证，不能只凭 synthetic complex 收益 promotion。

## 后续

不要继续扫同一组 Triple 参数。规则线下一步应做：

1. 显式 extortion / ZD 识别（当前 extort 469.0 vs RL-Repair 497.6）；
2. probe-style opening 响应（probe_exploit 228.8 vs 282.9）；
3. 如需提交，先补 C++ 移植和 action parity；
4. 不要加入全局主动 probe——`dev_family` 的 DelayedGrim 会直接崩掉
   （Harvest45 的历史教训）。

## 文件

- 完整报告：`results/rule_v4_20261043/README.md`
- 候选：`ipd/policies/candidates/triple_v4.py`
- 测试：`tests/test_candidates.py::TripleV4Tests`
