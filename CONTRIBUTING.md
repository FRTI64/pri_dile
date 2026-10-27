# 贡献指南

## 1. 环境

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python3 -m ipd test
```

- Python ≥ 3.10；
- 默认安装**无第三方依赖**；推理只用标准库；
- 训练需要 `pip install -r requirements-train.txt`（torch/numpy）；
- Axelrod 对手库需要 `pip install -r requirements-axelrod.txt`，版本固定
  `4.14.0`。

## 2. 代码结构

```text
ipd/core/        模拟内核、对手库、特征、规则状态机
ipd/policies/    策略接口、候选、模型推理、注册表
ipd/evaluation/  Protocol v4、四分布、blind、suite、matchup、league、split
ipd/training/    BC / PPO / adversarial / self-play / opponent-model
ipd/tools/       C++ 导出、matchup 透视
tests/           分层测试
cpp/             C++17 推理
```

依赖方向：

```text
core ← policies ← evaluation
                ← training
```

- `core` 不得导入 `policies/evaluation/training`；
- `policies` 不得导入 `training`；
- 候选之间不得互相 import；
- `evaluation` 只通过 `policies.registry` 构造候选。

详见 [docs/architecture/overview.md](docs/architecture/overview.md)。

## 3. 开发流程

1. 先读相关代码和文档，写一个短计划；
2. 新模块先加测试，再实现；
3. 本地运行 `python3 -m ipd test`；
4. 涉及策略接口时运行 `python3 -m ipd self-test --candidates all`；
5. 涉及评估/训练时，记录命令、seed、协议和输出路径；
6. 更新文档（见第 6 节）。

## 4. 新增策略

1. 在 `ipd/policies/candidates/` 新建独立文件，实现 `KIND`、`DISPLAY`、
   `build(...)`；
2. 在 `ipd/policies/candidates/__init__.py` 的 `MODULES` 注册；
3. 在 `ipd/policies/registry.py` 增加显示名、别名、集合；
4. 在 `tests/test_candidates.py` 或新测试中加边界/一致性测试；
5. 更新 [docs/reference/candidates.md](docs/reference/candidates.md)。

候选必须实现：

```python
def reset(self) -> None: ...
def act(self, t: int, mine: list[int], theirs: list[int]) -> int: ...
```

- `t = 0..99`；
- `C=0`、`D=1`；
- 第 1 天 C、对手未 D 时 C、第 100 天 D（学习策略的安全规则）；
- 有状态策略（GRU / SelfPlay）每场必须 `reset()`。

## 5. 新增评估 / 训练任务

- 评估优先复用 `evaluation.common.eval_task` / `evaluate_field`，不要自己写
  匹配循环；
- 训练任务必须暴露 `main(argv=None)`，接受 `--seed`、`--out`，学习任务接受
  `--device`；
- 产物写 JSON/CSV 到 `results/`，路径用 `ipd.paths`；
- 新 CLI 命令在 `ipd/cli.py` 注册，并更新
  [docs/reference/cli.md](docs/reference/cli.md)。

## 6. 文档

文档结构：

```text
docs/architecture/   how it works / why
docs/guides/         how-to
docs/reference/      commands, candidates, config, metrics
docs/policies/       policy methods
docs/research/       experiment records + Go/No-Go
docs/operations/     handover / next session / promotion
docs/history/        archived snapshots
docs/decisions/      ADR
```

规则：

- 当前结论写根 README / handover / reference；实验文档不回写历史数字；
- 实验文档顶部标 Status、日期、协议、Go/No-Go；
- 历史文档只加归档标记和替代链接；
- 新增/移动文档后更新 `docs/README.md`；
- 重要工程决策写 ADR，模板见
  [docs/decisions/README.md](docs/decisions/README.md)。

## 7. 评估与选模纪律

- 新候选用 **Protocol v4**：`dev_param` / `dev_family` / `dev_complex` /
  `dev_axelrod` / `dev_ood` 选模；正式 dev 用 499 人 / 30 fields；
- `test_v7` 尚未消费；消费后下一个候选需要 `test_v8`；
- 学习策略默认 `sample`，`argmax` 只作对照；
- 不混用 Protocol v4 与 legacy official/blind/heldout/v2/v3 数字；
- 不只看均值：报告 per-family、`min_norm`、CVaR、held-out red；
- 不把“对训练 red 有效”当成 anti-exploitation 解决。

## 8. 提交前检查

- [ ] `python3 -m ipd test` 通过（最近完整测试是 v3 的 204 tests）；
- [ ] `python3 -m ipd self-test --candidates all` 通过；
- [ ] `python3 -m ipd protocol audit` 通过；test 结果已写入 usage ledger；
- [ ] 新候选/命令已注册，文档已更新；
- [ ] 权重/配置/报告路径一致；
- [ ] 没有覆盖主权重或历史结果；
- [ ] 没有提交 `.pt`、临时日志或 `__pycache__`；
- [ ] `git diff --check` 通过；
- [ ] 变更描述包含实验协议、结果和 Go/No-Go。

## 9. 提交信息

建议使用动词开头的短标题：

```text
Add optional Axelrod opponent bank
Fix fair-RNG in planner evaluation
Document E1c repair candidate
```

实验类提交在正文写清：协议、seed、指标、结论（Go/No-Go）、产物路径。
