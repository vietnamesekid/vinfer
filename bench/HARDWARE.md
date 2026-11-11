# Benchmark hardware

There is no permanent machine: hardware is rented, and every result records the provider,
instance type, GPU, driver, CUDA and PyTorch it ran on.

| Provider | Instance | GPU |
|----------|----------|-----|
| RunPod (Secure Cloud, on-demand) | 1× H100 80GB pod + network volume | NVIDIA H100 80GB (SXM or PCIe: recorded with the first published results, then kept fixed) |

Rules:

- Official measurements only on **dedicated** GPUs, never spot or shared instances.
- Baselines are re-run **in the same rental session** as the engine.
- Machines are built with `scripts/setup.sh`. Weights live on a persistent volume.
  Push results off the machine before shutting it down (`scripts/push_results.sh`).
