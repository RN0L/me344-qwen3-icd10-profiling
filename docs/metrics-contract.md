# Metrics contract

Every benchmark run emits exactly one JSON file to `results/<run_id>.json` matching this schema.
All charts, tables and the README are generated from these files and nothing else. If a number
appears in the report, it came from one of these files.

```json
{
  "run_id": "tpu-v5e8-bs8-seq512",
  "backend": "tpu",
  "timestamp_utc": "2026-08-10T21:04:11Z",

  "hardware": {
    "label": "TPU v5e 2x4",
    "chips": 8,
    "chip_model": "tpu-v5-lite-podslice",
    "host_arch": "x86_64",
    "hbm_per_chip_bytes": 17179869184
  },

  "config": {
    "model": "Qwen/Qwen3-4B",
    "batch_size": 8,
    "seq_len": 512,
    "lora_rank": 32,
    "lora_alpha": 64.0,
    "dtype": "bfloat16",
    "max_steps": 100,
    "remat": "DECODER",
    "file_cache_capacity": "100Gi"
  },

  "phases": {
    "submit_to_running_s": 180.2,
    "image_pull_s": 45.1,
    "model_load_s": 1620.0,
    "compile_s": 180.0,
    "steady_state_s": 300.0,
    "checkpoint_write_s": 60.0,
    "total_wall_s": 2385.3
  },

  "steps": [
    {"step": 1, "wall_s": 182.4, "loss": 8.31},
    {"step": 2, "wall_s": 0.44, "loss": 7.90}
  ],

  "steady": {
    "median_step_s": 0.42,
    "p10_step_s": 0.40,
    "p90_step_s": 0.47,
    "tokens_per_s": 9752.4,
    "docs_per_s": 19.05,
    "warmup_steps_excluded": 10
  },

  "memory": {
    "peak_bytes": 12884901888,
    "capacity_bytes": 17179869184,
    "peak_pct": 75.0,
    "source": "jax.local_devices()[0].memory_stats()"
  },

  "utilization": {
    "mean_pct": 62.0,
    "max_pct": 94.0,
    "sample_interval_s": 1.0,
    "n_samples": 300,
    "source": "nvidia-smi | tpu-device-plugin metrics"
  },

  "eval": {
    "micro_f1": 0.0,
    "n_examples": 0,
    "note": "functional sanity check only, not a graded metric"
  },

  "status": "ok",
  "error": null,
  "notes": ""
}
```

## Field rules

- `status` is one of `ok`, `oom`, `error`, `timeout`. On `oom` the `steady` and `memory` blocks
  may be null but `phases` up to the failure point must still be present — the OOM boundary in
  the sweep chart is built from these records.
- `steps[0]` is the compile step and is always an outlier. `steady.*` excludes the first
  `warmup_steps_excluded` steps.
- `phases.*` are wall-clock seconds and must sum to approximately `total_wall_s`. Any
  unattributed remainder goes in `phases.other_s` rather than being silently dropped.
- `memory.peak_bytes` on TPU comes from `jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]`,
  on GPU from `nvidia-smi --query-gpu=memory.used`, on CPU from peak RSS.
- Numbers are never rounded on the way in. Rounding happens in the chart layer.

## Run ID convention

`<backend>-<hardware>-bs<batch>-seq<seqlen>[-<variant>]`

Examples: `cpu-x86-32c-bs1-seq512`, `gpu-gh200-bs8-seq512`, `tpu-v5e8-bs8-seq512`,
`tpu-v5e8-bs8-seq512-filecache` (the mitigation re-measurement).
