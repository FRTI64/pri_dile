# GPU 冒烟测试（2026-10-26）

## 环境

- GPU：NVIDIA GeForce RTX 5090 Laptop GPU，24463 MiB
- 驱动：621.01 / CUDA UMD 13.5
- PyTorch：2.11.0+cu128（编译 CUDA 12.8）
- `torch.cuda.is_available() = True`
- `resolve_device("auto") = cuda`

## 测试内容

用极小配置分别跑通三条训练路径，确认设备选择和 GPU 前向/反向：

```bash
.venv/bin/python -m ipd train neural --device cuda \
  --workers 4 --epochs 2 --fields-per-dist 1 --val-fields-per-dist 1 \
  --dagger-fields-per-dist 1 --n 20 --val-n 20 --batch-size 4096 \
  --out /tmp/.../neural_weights.json

.venv/bin/python -m ipd train gru --device cuda \
  --workers 4 --epochs 2 --fields-per-dist 1 --val-fields-per-dist 1 \
  --dagger-fields-per-dist 1 --n 20 --val-n 20 \
  --out /tmp/.../gru_weights.json

.venv/bin/python -m ipd train rl --device cuda \
  --workers 4 --iterations 2 --fields-per-dist 1 --n 20 \
  --val-fields-per-dist 1 --val-n 20 --epochs 2 --batch-size 4096 \
  --out /tmp/.../rl_weights.json
```

## 结果

三条路径日志均显示：

```text
device: cuda (NVIDIA GeForce RTX 5090 Laptop GPU)
training on cuda (NVIDIA GeForce RTX 5090 Laptop GPU)
```

| 任务 | 设备 | 结果 |
|---|---|---|
| MLP BC + DAgger | cuda | 正常完成，2 epochs |
| GRU BC + DAgger | cuda | 正常完成，2 epochs |
| PPO RL | cuda | 正常完成，2 iterations |

GPU 显存峰值约 **428 MiB**（`nvidia-smi` 1 秒采样）。本次只是极小规模冒烟，
数据生成仍在 CPU，模型更新量很小，所以采样到的 GPU 利用率只有 0–1%；这不能
代表完整训练时的吞吐。

原始日志、训练历史和 GPU 采样见本目录。推理层是纯 Python / C++，不依赖 GPU；
GPU 只用于训练。
