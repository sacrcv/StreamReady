# StreamReady: Learning What to Answer and When in Long Streaming Videos

Official PyTorch implementation of [StreamReady (CVPR 2026)](https://arxiv.org/pdf/2603.08620). 


Streaming video understanding is usually scored on whether a model is right, never
on whether it answered at the right moment. Answering before the supporting
evidence appears is speculation; answering long after it has passed is of little
real-time use. StreamReady treats timing as part of the task: a lightweight
readiness mechanism decides whether enough evidence has been observed before the
model responds, and the **Answer Readiness Score (ARS)** scores that decision with
asymmetric early and late penalties. Combined with correctness it gives an
*effective accuracy* that reflects both what a model answers and when.

This repository contains the StreamReady framework, the ARS metric, and the
loaders and evaluation for **ProReady-QA** — a long streaming benchmark with
annotated answer evidence windows and proactive multi-turn questions, released
separately on the Hugging Face Hub — along with OVO-Bench evaluation.

---

## Architecture

StreamReady is a block that sits inside a frozen MLLM, between its vision tower
and its language decoder. Here that host is **Qwen2-VL** (paper Sec. 5), and both
halves come from one checkpoint so the decoder sees features from the tower it
was trained against:

```
frames ──► Qwen2-VL vision tower ──► StreamReady ──► Qwen2-VL decoder ──► answer
            (frozen, 3584-wide)      (all training)      (frozen)
```

Everything StreamReady contributes lives in one module
(`lavis/models/streamready/module.py`) behind a two-line contract — it takes
`[B, T, N, vision_width]` frame features and returns decoder-ready prefix
embeddings plus a readiness score per timestep. It does not know what produced
the features or what consumes its output, so hosting it on a different MLLM means
writing a backbone that reports its two widths and exposes an embedding-level
decoder entry point. The module is unchanged.

As the video streams in, frames are encoded once by the frozen tower and folded into a
three-level **Visual Memory Tree** whose size does not grow with video length.
When a question arrives, a dual-branch Q-Former reasons over recent raw frames and
over evidence retrieved from the compressed levels, while a `<RDY>` token riding
inside the long-term branch tracks how ready the model is to answer.

| Component | What it does | File |
| --- | --- | --- |
| Visual Memory Tree | `M_V1` FIFO of raw frames → `M_V2` EMA centroids → `M_V3` prototypes | `lavis/models/streamready/visual_memory.py` |
| Query-aware retrieval | Top-K prototypes, then Top-m of their centroids | `lavis/models/streamready/retrieval.py` |
| Contextual Memory Bank | relevance-gated reuse of earlier QA turns | `lavis/models/streamready/contextual_memory.py` |
| Readiness mechanism | `<RDY>` token + MLP head + timing loss | `lavis/models/streamready/readiness.py` |
| Short→long fusion | injects `z_s` inside each cross-attending long block (Eq. 6) | `lavis/models/streamready/fusion.py` |
| StreamReady module | the pluggable block: all of the above, nothing else | `lavis/models/streamready/module.py` |
| Backbone | frozen Qwen2-VL vision tower + decoder | `lavis/models/streamready/backbone.py` |
| Model | composes backbone + module + the two losses | `lavis/models/blip2_models/streamready.py` |
| ARS | early/late penalties, effective accuracy | `lavis/metrics/ars.py` |

```
            frame_t ──► Qwen2-VL tower ──► Visual Memory Tree
                                         │        │
                                       M_V1     M_V2 / M_V3
                                         │        │
                            ┌────────────┘        └──── Top-K / Top-m retrieval
                            ▼                                  │
             question ──► Q_s (short-term) ──► z_s              │
                                                │              ▼
                                                │   Q_l (long-term) + <RDY>
                                   per-layer    │      │              │
                                   fusion  └────┴──────┘              │
                                                       ▼              ▼
                                           Contextual Memory    Readiness Head
                                                       │              │
                                                       ▼          R_pred ≥ τ ?
                                            Qwen2-VL decoder ◄─────────┘
```

The two branches interact exactly as in [HierarQ](https://arxiv.org/pdf/2503.08585). Eq. 6 reads
`z_l = Q_l(Concat[retrieved slots], z_s)` — the short-term output is a **second
argument** to the long branch, not extra tokens appended to its query sequence.
Concretely, every cross-attending block of the long Q-Former gets an extra
attention + gate pair that attends from its own hidden states over `z_s`
(`lavis/models/streamready/fusion.py`). The only token StreamReady appends to the long branch's query
sequence is `<RDY>`. Unlike HierarQ there is **no query memory bank**; the single
Visual Memory Tree is the only memory on the visual side.

`long_term_fusion_mode` selects the routing: `per_layer` (every long block sees
the final `z_s`), `per_layer_matched` (long block at depth `d` sees the short
branch's depth-`d` output), or `post` (one fusion pass after the tower, for
ablations).

The memory levels are maintained under `no_grad` as a running summary of detached
evidence; gradients reach the Q-Formers through the current frame and the
retrieved slots. The readiness loss updates **only** the `<RDY>` token and the
readiness head, so the reasoning pathway learns *what* to answer from the
video-text loss while readiness independently learns *when*.

### Answering is online

At inference the readiness decision is made **inside** the streaming pass:
`z_ℓ` is frozen at the timestep the score first crosses the threshold, and that
is the representation the decoder answers from. A turn whose score never crosses
is forced to answer at the last frame, which ARS scores as late.


---

## Installation

```bash
git clone <this-repo> && cd StreamReady
pip install -e .
```

`lavis/` is a vendored and extended copy of Salesforce LAVIS, so there is no
`salesforce-lavis` dependency. Beyond `requirements.txt` you need `ffmpeg` on
`PATH` for frame extraction. `requirements-lock.txt` records the exact environment
the code was developed against (Python 3.10, CUDA 11.8, torch 2.7.1, transformers
4.45.2).

> **Keep `transformers < 4.46`.** Newer releases change the `BertSelfAttention`
> internals that the vendored Q-Former builds on.

---

## Datasets

Every dataset is read as pre-extracted JPEG frames at 1 FPS (paper supplementary
Sec. D), laid out as `<root>/<video_id>/frame%06d.jpg`, 1-indexed. Bare
`%06d.jpg` dumps are also accepted, so an existing extraction can be reused
without renaming. Extract with:

```bash
python data/extract_frames.py --src-dir /path/to/videos \
    --dst-dir data/<dataset>/frames --fps 1
```

Paths resolve against `$STREAMREADY_DATA_ROOT` when set, otherwise `<repo>/data`.

Both ProReady-QA and OVO-Bench are **evaluation-only**. Training uses a separate
corpus, so nothing the model is scored on is ever trained on.

### Training data

The reasoning module and readiness mechanism are trained on 62k video-QA pairs
(paper supplementary Table 11). None of these are redistributed here — download
each from its original source:

| Dataset |  Source |
| --- | --- |
| MSRVTT-QA | [xudejing/video-question-answering](https://github.com/xudejing/video-question-answering) |
| MSVD-QA | [xudejing/video-question-answering](https://github.com/xudejing/video-question-answering) |
| ActivityNet-QA | [MILVLG/activitynet-qa](https://github.com/MILVLG/activitynet-qa) |
| MovieChat-1k | [Enxin/MovieChat-1K_train](https://huggingface.co/datasets/Enxin/MovieChat-1K_train) |

Convert them into the unified annotation schema documented at the top of
`lavis/datasets/datasets/streaming_qa_datasets.py` and concatenate into
`data/video_qa_train/annotation/train.json`, with frames under
`data/video_qa_train/frames`. These are plain video-QA pairs with no ask time, so
nothing is truncated and the whole clip is observable. The readiness mechanism
derives its own pseudo evidence timestamps (Eqs. 7–8), which is why training
needs no timing labels — and why ProReady-QA can stay held out.

### ProReady-QA

The benchmark introduced with the paper, on the Hugging Face Hub:
**[shehreen3764/ProReady-QA](https://huggingface.co/datasets/shehreen3764/ProReady-QA)**.
Every question is posed at a `question_timestamp` and is not yet answerable at
that moment; its evidence appears later, inside an annotated window
`[evidence_start, evidence_end]`. Evaluation-only — it measures answer timing, so
there is no train split.

**1. Get the annotations.**

```bash
huggingface-cli download shehreen3764/ProReady-QA --repo-type dataset \
    --local-dir data/proready_qa
```

Read in place, no conversion step: the loader takes the released flat record
list, groups turns into dialogues by `dialogue_id`, aliases `id` → `qa_id` and
`answer_type` → `task`, and reads per-video frame counts from
`frames_manifest.json`.

**2. Get the frames.**  Follow the
dataset card: obtain [Ego4D](https://ego4d-data.org/) and
[MovieNet](https://movienet.github.io/), cut the clips listed in `clips/ego.json`
and `clips/movie.json`, name each directory after its `video_id`, then extract at
**1 FPS** into `data/proready_qa/frames`:

```bash
python data/extract_frames.py --src-dir /path/to/proready_clips \
    --dst-dir data/proready_qa/frames --fps 1
```

The 1 FPS rate is load bearing — timestamps are in the same units as frame
indices. Verify the extraction against the shipped manifest:

```bash
python data/extract_frames.py --dst-dir data/proready_qa/frames \
    --verify data/proready_qa/frames_manifest.json
```

**3. Evaluate.**

```bash
bash run_scripts/proready_qa/test.sh /path/to/checkpoint.pth
```

An item is a whole dialogue, answered in one streaming pass; see
[Multi-turn dialogues stream once](#multi-turn-dialogues-stream-once).



### OVO-Bench

The online video understanding benchmark, also evaluation-only. Its native JSON
holds three different record shapes, so it is converted once into the unified
schema and read by the shared loader
(`lavis/datasets/datasets/streaming_qa_datasets.py`):

| Mode | Tasks | Answer encoding |
| --- | --- | --- |
| backward | EPM, ASI, HLD | MC, 0-based index `gt` |
| real-time | STU, OJR, ATR, ACR, OCR, FPD | MC, 0-based index `gt` |
| forward | REC | integer count |
| forward | SSR, CRR | Yes / No |

Two steps. Convert the annotations, then extract frames at 1 FPS:

```bash
python tools/convert_benchmarks.py --benchmark ovo_bench \
    --raw /path/to/OVO-Bench \
    --frames-root data/ovo_bench/frames \
    --output data/ovo_bench/annotation/ovo_bench_test.json

python data/extract_frames.py --src-dir /path/to/OVO-Bench/videos \
    --dst-dir data/ovo_bench/frames --fps 1
```

`frame_length` is what the frame sampler indexes into, so the converter counts
frames on disk and drops videos whose frames are missing — the annotation it
writes is loadable by construction. Run the converter again after extracting
more. `--no-require-frames` estimates the count from the annotation's duration
instead, for preparing annotations ahead of extraction.

Then evaluate:

```bash
bash run_scripts/benchmarks/eval.sh /path/to/checkpoint.pth
```


---

## Pretrained weights


**The backbone.** `model.backbone_model` points at a Qwen2-VL checkpoint, a local
directory or a hub id. Both frozen halves come from it:

```bash
BACKBONE=/path/to/Qwen2-VL-7B-Instruct bash run_scripts/train.sh
```

> **Note on trained checkpoints.** Trained StreamReady weights will not be released,
> as the method is currently under the patent process noted at the end of this
> README. Everything needed to train the model from the publicly available
> initialisations above is provided here.

---

## Training

Reasoning and readiness are trained **separately** (paper Sec. 3.4): reasoning by
cross-entropy on the answer, readiness by the Eq. 7–8 contrastive loss over
pseudo evidence timestamps. Run the two stages in order, carrying the first
stage's checkpoint into the second:

```bash
STAGE=reasoning bash run_scripts/train.sh
STAGE=readiness INIT_FROM=<stage-1 checkpoint> bash run_scripts/train.sh
```

Each stage writes to its own `output/streamready_train_<stage>/`. `INIT_FROM`
loads weights only; `RESUME` is for picking an interrupted run back up in the
same stage and also restores the optimizer, which only matches when the trainable
parameter set is unchanged. Across a stage boundary use `INIT_FROM`.

- `STAGE=reasoning` optimises cross-entropy only. The `<RDY>` token and the
  readiness head are frozen.
- `STAGE=readiness` optimises the contrastive timing loss only. The whole
  reasoning pathway is frozen, leaving the `<RDY>` token and the readiness head
  as the only trainable tensors, and the LLM forward is skipped since nothing
  needs it.


Each stage trains for 5 epochs at lr 2e-5 with cosine annealing and AdamW, frames
at 1 FPS and 224×224, with the visual encoder and language decoder frozen.
at 1 FPS and 224×224, with the whole Qwen2-VL backbone frozen. Trainable in the
reasoning stage: both Q-Former branches, the vision projection, the retrieval
projections, the contextual fusion and the output projection into the decoder.

Override without editing configs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STAGE` | `reasoning` | `reasoning` / `readiness` / `joint` |
| `INIT_FROM` | — | checkpoint to initialise weights from |
| `RESUME` | — | resume an interrupted run (weights + optimizer) |
| `BACKBONE` | `llm/Qwen2-VL-7B-Instruct` | frozen Qwen2-VL checkpoint |
| `NUM_FRAMES` | `100` | frames per question |
| `BATCH_SIZE` / `ACCUM` | `2` / `8` | per-GPU batch and accumulation |
| `EPOCHS` / `LR` | `5` / `2e-5` | schedule |
| `NUM_GPUS` | autodetected | GPU count |
| `EXTRA` | — | extra `--options key value` pairs |


---

## Evaluation

```bash
bash run_scripts/proready_qa/test.sh /path/to/checkpoint.pth   # ProReady-QA
bash run_scripts/benchmarks/eval.sh /path/to/checkpoint.pth    # OVO-Bench
```

ProReady-QA reports accuracy, ARS and effective accuracy with a per-task
breakdown. OVO-Bench reports accuracy (the metric the paper uses for it), both
per-question and mode-averaged as its official scorer does:



---

## Repository layout

```
StreamReady-release/
├── train.py / evaluate.py          entry points (train.py also evaluates)
├── data/extract_frames.py          1 FPS frame extraction
├── lavis/
│   ├── models/streamready/
│   │   ├── backbone.py             frozen Qwen2-VL tower + decoder
│   │   ├── module.py               the pluggable StreamReady block
│   │   └── ...                     memory, retrieval, readiness, Eq. 6 fusion
│   ├── models/blip2_models/        vendored Q-Former + the composition layer
│   ├── metrics/ars.py              Answer Readiness Score
│   ├── metrics/answer_matching.py  MC / counting / Yes-No / open-ended scoring
│   ├── datasets/                   ProReady-QA + unified QA loader
│   ├── tasks/readiness_vqa.py      Acc / ARS / effective accuracy
│   ├── tasks/video_qa.py           accuracy, for OVO-Bench
│   ├── configs/ projects/          model, dataset and experiment configs
│   └── runners/ processors/        vendored LAVIS plumbing
├── run_scripts/
│   ├── train.sh                    two-stage training
│   ├── proready_qa/test.sh         ProReady-QA eval (accuracy + ARS)
│   └── benchmarks/eval.sh          OVO-Bench eval
└── tools/
    ├── convert_benchmarks.py       OVO-Bench -> unified schema
    ├── evaluate_ars.py             standalone ARS scorer
    └── extract_vit_features.py     frozen-ViT feature cache
```

---

## Citation

```bibtex
@inproceedings{azad2026streamready,
  title     = {StreamReady: Learning What to Answer and When in Long Streaming Videos},
  author    = {Azad, Shehreen and Vineet, Vibhav and Rawat, Yogesh Singh},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgements

- [HierarQ](https://arxiv.org/pdf/2503.08585) - the dual branch QFormer design
- [LAVIS](https://github.com/salesforce/LAVIS) — the vendored library this builds on
- [VStream-QA](https://github.com/rese1f/StreamingBench) — source videos for ProReady-QA

---

## License

This code is released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
for **non-commercial research use only**. See `LICENSE` for the full terms and
attribution.

**Third-party exception.** This repository vendors
[Salesforce LAVIS](https://github.com/salesforce/LAVIS); every file under
`lavis/` that is not part of the StreamReady framework itself remains under the
**BSD 3-Clause** license. Those files are not relicensed, and the NonCommercial
and ShareAlike conditions do not apply to them. `LICENSE` Part A and Part B list
exactly which files fall on each side.

### Patent notice

The technology implemented here is the subject of a **pending patent
application** owned by the **University of Central Florida**. The application is
in process, so no granted patent number is stated.

Because of this, **no trained checkpoints are released** — only the code and the
publicly available initialisations needed to train from scratch.

**No patent rights are granted by this license.** CC BY-NC-SA 4.0 is a
copyright license only — its Section 2(b)(2) states that "Patent and trademark
rights are not licensed under this Public License." Commercial use, and any use
requiring patent rights, requires a separate license from UCF:
[UCF Office of Technology Transfer](https://www.research.ucf.edu/tech-transfer/contact/).
