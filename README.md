
# Chart Specification: Structural Representations for Incentivizing VLM Reasoning in Chart-to-Code Generation

## Overview

**Chart Specification** is a specification-driven reinforcement learning framework for chart-to-code generation. Given a rasterized chart image, the goal is to produce executable plotting code that faithfully reconstructs the original visualization.

Existing approaches largely rely on supervised fine-tuning (SFT) with token-level objectives. While effective at surface imitation, such methods often fail to recover the underlying structural logic of charts, leading to:

- Structural hallucinations (e.g., incorrect data dependencies)
- Semantically inconsistent layouts
- Executable but visually incorrect outputs

To address this supervision mismatch, we introduce **Chart Specification**, a canonical structural intermediate representation that bridges visual perception and code execution. Built upon this representation, we further propose:

- **ChartStruct**: A structurally balanced training corpus curated via specification-aware sampling.
- **Spec-Align Reward**: A fine-grained, verifiable reward mechanism for reinforcement learning (RL), enabling structure-consistent optimization.

Our approach achieves state-of-the-art results on **ChartMimic**, **Plot2Code**, and **ChartX**, demonstrating strong structural fidelity and superior data efficiency.

---

## Key Contributions

### 1. Chart Specification (Spec)

Chart Specification is a structured JSON representation that abstracts plotting scripts into semantically verifiable components:

- **Global Topology** (chart type, panel layout)
- **Coordinate Systems** (Cartesian, Polar, 3D)
- **Data Domains** (axis ranges, categories)
- **Analytic Representations** (functional forms, transformations)
- **Runtime Numerical Facts** (intercepted via execution hooks)

Formally:

```

S = ⟨S_sem, S_code⟩

```

- `S_sem`: Declarative semantic structure
- `S_code`: Execution-grounded numerical primitives

This representation removes syntactic noise from plotting libraries and enables deterministic structural comparison.

---

### 2. ChartStruct: Spec-Driven Data Curation

Instead of naively sampling chart data, we construct a structurally balanced corpus guided by specification signatures:

- 20 canonical chart families
- 55 structural signatures across:
  - Coordinate space
  - Data mode
  - Composition topology

We apply **Complexity-Adaptive Sampling** to prioritize difficult structural configurations (e.g., 3D, multi-panel, contour).

Two training scales are provided:

- **3K-scale** (3,008 samples) — data efficiency studies
- **4K-scale** (4,000 samples) — main experiments

Each sample consists of:

```

(v, c, S)

```

- `v`: chart image  
- `c`: executable plotting code  
- `S`: extracted Chart Specification  

---

### 3. Spec-Align Fine-Grained Reward (RLVR)

We train models using **Group Relative Policy Optimization (GRPO)** with a hierarchical reward tree.

The total reward is:

```

R(y) = R_format + R_exec + R_sem + β R_code

````

#### Reward Stages

1. **Format Gate**
   - Enforces reasoning structure (`<think>` → `<answer>`)

2. **Execution Gate**
   - Sandbox compilation check

3. **Topological Gate**
   - Chart type
   - Panel layout
   - Global structure

4. **Semantic Alignment**
   - Coordinate systems
   - Domain IoU
   - Series consistency
   - Function matching

5. **Code-Level Calibration**
   - Statistical metrics (box/violin)
   - Relational F1 (graph/treemap)
   - Vector similarity (quiver)
   - Numerical distances

Unlike binary execution rewards, Spec-Align provides dense, structure-aware supervision suitable for RL optimization.

---

## Installation

https://verl.readthedocs.io/en/latest/start/install.html

---

## Data Preparation

### 1. Build Chart Specification

We provide tools for extracting both semantic and runtime specifications.

```bash
cd spec_extraction
python extract_spec.py \
  --code_path /path/to/script.py \
  --output_path /path/to/spec.json
```

This step:

* Parses plotting scripts into `S_sem`
* Executes sandboxed rendering
* Intercepts numerical primitives to construct `S_code`

---

### 2. Construct ChartStruct

To build a balanced training set:

```bash

```


---

## Training

We use **Qwen2.5-VL-7B** as the backbone and optimize with GRPO.

### Launch Training

```bash
bash train_grpo.sh
```

Important arguments in `train_grpo.sh`:

```bash
--model_path /path/to/Qwen2.5-VL-7B
--data_path ./data/chartstruct_4k
--rollout_num 16
--batch_size 32
--lr 4e-7
--kl_coef 0.001
```

Training setup (default):

* 3 epochs
* 8 × A100 (80GB)
* ~62 hours
* Visual encoder frozen
* Full LM + projector fine-tuned

---

## Reproducing Main Results

To reproduce the 4K ChartSpec model:

### Step 1 — Build 4K ChartStruct

```bash
```

### Step 2 — Train with Spec-Align RL

```bash
```

## Project Structure

```
ChartSpec/
├── spec_extraction/        # Chart Specification extraction
├── data/                   # ChartStruct construction
├── reward/                 # Spec-Align reward implementation
├── training/               # GRPO optimization
├── evaluation/             # Benchmark evaluation scripts
└── scripts/                # Launch scripts
```

---
<!--

## Citation

If you find this work useful, please cite:

```bibtex
@article{,
}
```
-->

---

## License

Released under the Apache-2.0 license.

---

## Acknowledgements

This work builds upon:

* Qwen2.5-VL
* Chart2code-160k
* ReachQA
* ChartMimic
* Plot2Code
* ChartX
* verl
