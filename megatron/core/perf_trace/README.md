# perf_trace: per-step, per-rank, per-layer training timeline

An opt-in instrumentation that records **what every rank spends its step on**, cheaply enough to
leave on during a real run, plus a service that turns the records into pictures and tables.

It exists because profiling answers "what happened in one step on five ranks" while the questions
that actually decide an optimization are "which rank is late, in which layer, and what is it
waiting for" — which needs **all ranks, every step**.

---

## 1. Turn it on

```bash
--perf-trace-dir /cpfs/.../perf        # off unless given
--perf-trace-every-n-steps 1           # sample rate
--perf-trace-max-layers 256            # safety cap on instrumented layers
```

Nothing else changes. The hooks attach after the model is built, and every rank writes its own
file (`perf-rank-<rank>.jsonl`), so there is no write contention and no cross-rank communication
inside the tracer: it cannot deadlock, and disabling it leaves the run bit-identical.

## 2. What is recorded

One JSON line per traced step per rank. **Every duration is an integer number of microseconds**,
relative to the step's anchor unless noted.

```json
{"step": 19, "rank": 3,
 "phases": {"__anchor__": 0, "__end__": 242900000},
 "layers": [{"n": "l12", "fwd": 3100000},
            {"n": "l12#recompute", "fwd": 3050000}],
 "a2a":    [{"n": "dispatch", "s": 4210000, "d": 4380000}, ...],
 "spans":  [{"n": "optimizer", "s": 60500000, "d": 2200000}],
 "cpu":    {"backward-compute": 96000000, "params-all-gather": 2000000, ...},
 "data":   {"n": 4, "total": 25000, "max": 9000, "items": [6000, 7000, 9000, 3000]},
 "step_wall":  260100000,
 "anchor_wall": 1795147281.0}
```

| field | meaning | how |
|---|---|---|
| `layers[].fwd` | per transformer layer forward | CUDA events around the module's forward |
| `layers[].n` | **global** layer id (pipeline-offset included) | `get_transformer_layer_offset(config, vp_stage)` |
| `lN#recompute` | the second forward of a checkpointed layer | tagged instead of overwriting the first |
| `a2a` | MoE `dispatch` / `combine` on the comm stream | dispatcher `*_preprocess/postprocess`, class discovered |
| `spans` | named GPU regions, e.g. `optimizer` | `with perf_trace.phase('optimizer'): ...` |
| `cpu` | Megatron timer values (host-measured) | `forward/backward-compute`, `params-all-gather`, `optimizer-*` |
| `data` | host-side batch fetch, per `next()` | a transparent iterator proxy |
| `step_wall` | host wall clock for the step | for sanity, not for the GPU timeline |

### Why CUDA events and not `time.time()`

Launches are asynchronous, so the host clock says nothing about when a kernel runs.
`torch.cuda.Event` stores a marker *in the stream*; its position is the execution order no matter
when the host enqueued it. Event timestamps live in the device clock domain, so
`elapsed_time` between events on **different streams** is valid too — that is what makes the MoE
all-to-all (comm stream) comparable with layer compute (compute stream).

## 3. Look at it

```bash
# live dashboard (reads files, never touches the training process)
tools/perf_trace_dashboard.py --dir <perf-dir> --port 8080

# one self-contained HTML with every chart embedded
tools/perf_trace_dashboard.py --dir <perf-dir> --once --out report.html

# numbers only (no matplotlib; good over ssh / in CI logs)
tools/perf_trace_summary.py --dir <perf-dir> [--step N] [--top 8]

# profiling traces instead of perf-trace records
tools/analyze_profile.py single  --trace <trace.json.gz>
tools/analyze_profile.py skew    --dir <trace-dir>
tools/analyze_profile.py compare --a <a> --b <b>
```

| chart | answers |
|---|---|
| `/png/trends` | is the step getting slower over training, and is it forward, backward, A2A, data or a timer phase? |
| `/png/timeline?step=N` | one row per rank: layer blocks, A2A overlay, optimizer span, data bar — who is late and where |
| `/png/skew?step=N` | per-rank total, stragglers first |
| `/png/heatmap?step=N&metric=fwd\|bwd` | rank x layer: a hot **column** is a fixed layer cost, a hot **row** is a straggler |
| `/png/layers?rank=R` | one rank's per-layer distribution over the sampled steps |

**Reading imbalance**: `max / median` per step. ≈1.0 is level; 1.3 means the slowest rank spends
30% longer than the median rank — time every other rank then waits for in the next collective.
`perf_trace_summary.py` prints this ratio and labels it level / slight / notable / SEVERE.

## 4. Cost

| source | per step (48 layers x 4 microbatches) |
|---|---|
| layer forward start/end | 384 events |
| A2A dispatch/combine | 768 events |
| spans + anchor | ~10 |
| **total** | **~1160 events** |

At ~1.2 us host and <1 us device per event that is **~1.5-2.5 ms per step**: ~0.001% of a 244 s
step. Three design rules keep it there:

1. **no synchronize and no `elapsed_time()` on the hot path** — only `record()`;
2. a step's events are converted at the *start of the next* step, after `query()` says they are
   done (a stuck step is dropped rather than waited on);
3. file writing happens on a background thread.

`close()` does one `torch.cuda.synchronize()` — at shutdown there is nothing left to overlap, and
without it the final step's tail events would be silently dropped.

> **Measure it on your own run.** On a 13-75 ms test model the overhead is not measurable (the
> run-to-run spread is +-20%, five times larger than the effect). On production-size steps lock
> the clock, run interleaved off/on pairs, and report the paired delta — or report an upper bound.

## 5. Parallel layouts

The tracer has no cross-rank state, so support reduces to per-rank invariants. Verified on GPU:

| config | status |
|---|---|
| TP1 PP1 CP1 EP1 (dense + MoE) | PASS |
| TP1 PP1 CP1 EP2 (MoE) | PASS (A2A spans present) |
| TP>1 / PP>1 / VPP>1 / CP>1 | **not yet run** — use `tools/check_perf_trace_parallel.py` |

Two things make PP/VPP safe by construction:

* layer ids are **global** (offset by the pipeline stage), so two stages do not both number from 0;
* `vp_stage` is the model-chunk index, so VPP stages are distinguished.

The checker asserts, per configuration: rank count, one line per traced step, global layer ids,
non-zero layer times, A2A presence matching MoE, and the recompute tag appearing instead of an
overwrite.

```bash
torchrun --nproc-per-node 8 tools/validate_perf_trace.py --experts 8 --ep 2 --pairs 3
tools/check_perf_trace_parallel.py --dir <dir> --world 8 --tp 1 --pp 1 --ep 2 \
    --layers 48 --steps 3 --moe
```

## 6. Known limitations

* **Per-layer backward is not recorded.** `register_full_backward_hook` fires at points that do
  not bracket a layer's backward (measured: 20 us spans for a millisecond-scale layer), and the
  tensor-hook alternative only fired for one of four layers with Transformer Engine. Use
  `cpu['backward-compute']` for the total, and the per-layer *forward* plus the A2A/optimizer
  spans to locate a slow rank. Do not trust a `bwd` field if one appears: it is best-effort.
* **A2A spans appear at EP=1 too** (the dispatcher runs, with a group of size one), so presence
  is not evidence that expert parallelism is on.
* **No rotation/compression** of the JSONL yet: 256 ranks x ~6 KB/step is ~0.5 GB/day at one line
  per step. Sample with `--perf-trace-every-n-steps` for long runs.
* The dashboard needs matplotlib; the summary tool does not.

## 7. Tests

```bash
python -m pytest tests/unit_tests/test_perf_trace.py tests/unit_tests/test_perf_trace_report.py -q
```

They run on CPU with a fake device clock: event pairing, recompute tagging, global layer ids,
hook exception isolation, step sampling, incremental JSONL reading (including a half-written
line), and that every chart renders a PNG.
