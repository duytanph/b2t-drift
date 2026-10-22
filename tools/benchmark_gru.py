"""Benchmark GRU + CTC training step time and peak VRAM, synthetically.

Uses SYNTHETIC tensors shaped like real T15 trials (no dataset load, no
checkpoint, no HDF5 array reads) to measure step time and peak VRAM across
model size, sequence length, batch size, and precision. REQUIRES A CUDA
DEVICE.

Defaults (845 full-length / 211 downsampled, batch 32 and 64) match the
real `n_time_steps` distribution's median (`docs/DATASET.md` section 6).
An earlier version of this benchmark used the wrong median (527, one
session's own statistic mistaken for the dataset's) and is the subject of
two dated `docs/FINDINGS.md` entries: "Length-distribution documentation
errors traced to one session's statistics; padding waste re-measured" and
"At the real median length, batch=64 does not fit in 8 GB; it silently
spills through WDDM" -- read the second one before trusting a "fits"
result on a memory-constrained card: a batch that exceeds physical VRAM
does not raise `torch.cuda.OutOfMemoryError` under Windows' WDDM driver,
it silently spills to system RAM and gets 10-20x slower. This script
flags that case explicitly rather than reporting a bare "fits".

Run with: uv run tools/benchmark_gru.py
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 512
OUTPUT_SIZE = 41  # blank + 40 phoneme classes (b2t_drift.phonemes.N_CLASSES)
TARGET_LEN = 16
N_WARMUP = 10
N_TIMED = 50

MODEL_CONFIGS = {
    "baseline (5L,1024H)": dict(num_layers=5, hidden_size=1024),
    "small (3L,512H)": dict(num_layers=3, hidden_size=512),
}
DEFAULT_SEQ_LEN_FULL = 845
DEFAULT_SEQ_LEN_DOWNSAMPLED = 211  # 845 / 4, matching the real-time 4x downsample ratio
DEFAULT_BATCH_SIZES = [32, 64]
DEFAULT_PRECISIONS = ["fp32", "fp16"]
DEFAULT_MODELS = list(MODEL_CONFIGS)


class GRUCTCModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        # batch_first GRU with no arguments that would prevent cuDNN's
        # fused RNN kernel (no dropout between layers here, since dropout
        # forces the non-fused per-timestep path).
        self.gru = nn.GRU(INPUT_SIZE, hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.classifier(out)


def make_batch(seq_len: int, batch_size: int, device: torch.device):
    x = torch.randn(batch_size, seq_len, INPUT_SIZE, device=device)
    targets = torch.randint(1, OUTPUT_SIZE, (batch_size, TARGET_LEN), device=device)
    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    target_lengths = torch.full((batch_size,), TARGET_LEN, dtype=torch.long, device=device)
    return x, targets, input_lengths, target_lengths


def run_config(
    model_kwargs: dict,
    seq_len: int,
    precision: str,
    batch_size: int,
    n_warmup: int = N_WARMUP,
    n_timed: int = N_TIMED,
) -> dict:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = GRUCTCModel(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp = precision == "fp16"

    def step() -> None:
        x, targets, input_lengths, target_lengths = make_batch(seq_len, batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        # CTCLoss needs float32 log-probs for numerical stability; cast
        # back out of autocast for the loss itself, which is standard
        # practice (no GradScaler here -- this is a speed benchmark on
        # random targets, not a real training run, so we don't need
        # gradient-underflow protection; a real fp16 run would want one).
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

    try:
        for _ in range(n_warmup):
            step()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_timed):
            step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        ms_per_step = (elapsed / n_timed) * 1000
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        return {"status": "ok", "ms_per_step": ms_per_step, "peak_vram_mb": peak_vram_mb}
    except torch.cuda.OutOfMemoryError:
        return {"status": "OOM"}
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to time (default: %(default)s).",
    )
    parser.add_argument(
        "--seq-len-full", type=int, default=DEFAULT_SEQ_LEN_FULL,
        help="Undownsampled sequence length (default: %(default)s, the real n_time_steps median).",
    )
    parser.add_argument(
        "--seq-len-downsampled", type=int, default=DEFAULT_SEQ_LEN_DOWNSAMPLED,
        help="4x-downsampled sequence length (default: %(default)s).",
    )
    parser.add_argument(
        "--models", choices=list(MODEL_CONFIGS), nargs="+", default=DEFAULT_MODELS,
        help="Model configs to benchmark (default: both).",
    )
    parser.add_argument(
        "--precisions", choices=["fp32", "fp16"], nargs="+", default=DEFAULT_PRECISIONS,
        help="Precisions to benchmark (default: both).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this benchmark requires a GPU.")

    props = torch.cuda.get_device_properties(0)
    total_vram_mb = props.total_memory / (1024**2)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_vram_mb:.0f} MB")
    print(f"Compute capability: sm_{props.major}{props.minor}")

    seq_lens = {
        f"none ({args.seq_len_full})": args.seq_len_full,
        f"4x downsample ({args.seq_len_downsampled})": args.seq_len_downsampled,
    }

    rows = []
    for model_name in args.models:
        model_kwargs = MODEL_CONFIGS[model_name]
        for seq_name, seq_len in seq_lens.items():
            for precision in args.precisions:
                for batch_size in args.batch_sizes:
                    print(f"\n--- {model_name} | seq_len={seq_len} ({seq_name}) | "
                          f"{precision} | batch={batch_size} ---")
                    r = run_config(model_kwargs, seq_len, precision, batch_size)
                    r.update(model=model_name, seq=seq_name, precision=precision,
                             batch_size=batch_size)
                    rows.append(r)
                    if r["status"] != "ok":
                        print(f"  *** OOM at batch_size={batch_size} ***")
                        continue
                    hours_10k = r["ms_per_step"] * 10_000 / 1000 / 3600
                    spilled = r["peak_vram_mb"] > total_vram_mb
                    r["spilled"] = spilled
                    flag = " *** EXCEEDS PHYSICAL VRAM: LIKELY WDDM SPILL, NOT A REAL FIT ***" if spilled else ""
                    print(
                        f"  {r['ms_per_step']:.2f} ms/step, peak VRAM {r['peak_vram_mb']:.0f} MB, "
                        f"10,000 steps -> {hours_10k:.2f} h{flag}"
                    )

    print("\n=== Summary ===")
    header = (f"{'model':<22}{'seq':<22}{'prec':<7}{'batch':<7}{'status':<7}"
              f"{'ms/step':<10}{'VRAM MB':<10}{'hrs@10k':<10}{'flag':<10}")
    print(header)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['model']:<22}{r['seq']:<22}{r['precision']:<7}{r['batch_size']:<7}{r['status']:<7}")
            continue
        hrs = r["ms_per_step"] * 10_000 / 1000 / 3600
        flag = "SPILLED" if r.get("spilled") else ""
        print(
            f"{r['model']:<22}{r['seq']:<22}{r['precision']:<7}{r['batch_size']:<7}"
            f"{r['status']:<7}{r['ms_per_step']:<10.2f}{r['peak_vram_mb']:<10.0f}{hrs:<10.2f}{flag:<10}"
        )

    oom_configs = [r for r in rows if r["status"] == "OOM"]
    spilled_configs = [r for r in rows if r.get("spilled")]
    if oom_configs:
        print(f"\n{len(oom_configs)} configuration(s) OOM'd:")
        for r in oom_configs:
            print(f"  {r['model']} | {r['seq']} | {r['precision']} | batch={r['batch_size']}")
    if spilled_configs:
        print(f"\n{len(spilled_configs)} configuration(s) exceeded physical VRAM (WDDM spill, not a real fit):")
        for r in spilled_configs:
            print(f"  {r['model']} | {r['seq']} | {r['precision']} | batch={r['batch_size']}")
    if not oom_configs and not spilled_configs:
        print("\nNo configuration OOM'd or spilled.")


if __name__ == "__main__":
    main()
