from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import timeit
from basics.basics.model import BasicsTransformerLM

import torch.cuda.nvtx as nvtx


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the Basics transformer for the requested model size."""
    spec = MODEL_SPECS[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10000.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if config.compile_model:
        model = torch.compile(model)
    return model


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device, dtype=torch.long)


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    
    optimizer.zero_grad(set_to_none=True)

    with autocast_context:
        with nvtx.range("forward"):
            logits = model(batch)

        if mode in ["forward-backward", "train-step"]:
            loss = logits.sum()

            with nvtx.range("backward"):
                loss.backward()

        if mode == "train-step":
            with nvtx.range("optimizer-step"):
                optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()  # set to train mode for gradients
    batch = make_random_batch(config, device)
    autocast_context = make_autocast_context(config.use_bf16)

    maybe_start_memory_history(config.use_memory_profiler)
    # Warmup
    for _ in range(config.warmup_steps):
        with nvtx.range("warmup"):
            run_single_step(model, batch, optimizer, config.mode, autocast_context)

    # Measure
    start_time = timeit.default_timer()
    for _ in range(config.measure_steps):
        with nvtx.range("measure"):
            run_single_step(model, batch, optimizer, config.mode, autocast_context)
    end_time = timeit.default_timer()

    total_time = end_time - start_time
    time_per_step = total_time / config.measure_steps
    if config.use_memory_profiler:
        maybe_dump_memory_snapshot(True, config.output_dir)
    return {"time_per_step": time_per_step}


def annotated_scaled_dot_product_attention(*args, **kwargs):
    """Optional NVTX-annotated attention path for Nsight Systems profiling."""
    raise NotImplementedError


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        # Simple: no action, could log memory
        pass


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        if torch.cuda.is_available():
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "memory_snapshot.txt", "w") as f:
                f.write(torch.cuda.memory_summary())


def make_autocast_context(use_bf16: bool):
    if use_bf16 and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        output_dir=args.output_dir,
    )
    results = benchmark_model(config)
    print(f"Benchmark results: {results}")


if __name__ == "__main__":
    main()
