
# Chart Specification: Structural Representations for Incentivizing VLM Reasoning in Chart-to-Code Generation

📄[arXiv 2602.10880](https://arxiv.org/abs/2602.10880)

## Overview

**Chart Specification** is a specification-driven reinforcement learning framework for chart-to-code generation. Given a chart image, the goal is to produce the executable plotting code that faithfully reconstructs the original visualization.

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

For installation instructions, please see the official verl documentation: https://verl.readthedocs.io/en/latest/start/install.html

---

## Data Preparation

*ChartStruct-1k*, *ChartStruct-2k*, *ChartStruct-3k* and *ChartStruct-4k* are mainly built from **Chart2code-160k** and **ReachQA** datasets:
- **Chart2code-160k** can be found on: https://huggingface.co/datasets/xxxllz/Chart2Code-160k
- **ReachQA** can be found on: https://github.com/hewei2001/ReachQA/tree/main/data/reachqa_train

### 0. LLM API setup

Please start a [**Qwen3-32B**](https://huggingface.co/Qwen/Qwen3-32B) API Server.

### 1. Parse dataset with Qwen3-32B

We provide tools for extracting both semantic and runtime code specifications.

In the code file `./verl/utils/reward_score/chart2code_mllm/qwen_v2_5_72b.py`:
- set the *Qwen3-32B* API Address for the list variable `provider_url_pool` at Line 15. Add multiple *Qwen3-32B* API addresses are preferred if available.

In the code file `./verl/utils/reward_score/chart2code_mllm/example_extract_final_ir.py`, please configure the following parameters:
- Line 96, `chart2code_160k_full_json_filepath`: the  filepath for `chart2code.json` in deflated archive file https://huggingface.co/datasets/xxxllz/Chart2Code-160k/blob/main/json.tar.gz
- Line 97, `chart2code_160k_img_base_dir`: the path of the folder for deflated archive file https://huggingface.co/datasets/xxxllz/Chart2Code-160k/blob/main/images.tar.gz
- Line 128, `cache_result_base_dir`: the output dir for temp storage of **Chart2code-160k**
- Line 155, `input_raw_jsonl_filepath`: the JSONL filepath built by `./examples/data_preprocess/chart2code_train_dataset_factory/build_jsonl_from_reachqa_dir.py`, which will convert **ReachQA** dataset folder into JSONL file.
- Line 156, `cache_result_base_dir`: the output dir for temp storage of **ReachQA**

After builder parameters set, now we will toggle and initialize the conversion in file `./verl/utils/reward_score/chart2code_mllm/example_extract_final_ir.py`: 
- for **Chart2code-160k** dataset conversion, turn ON Line 124 and turn OFF Line 154
- for **ReachQA** dataset conversion, turn ON Line 154 and turn OFF Line 124

In the working directory `./verl/utils/reward_score/`:
- rename `math.py` to `math_utils.py` to avoid incorrect package importation
- start conversion:

```bash
python -m chart2code_mllm.example_extract_final_ir
```

This command runs the pipeline for generating the training dataset:

- Executes sandboxed rendering to filter out illegal dataset items.
- Parses plotting scripts into `S_sem`
- Intercepts numerical primitives to construct `S_code`

---

### 2. Construct ChartStruct Parquet file

Convert the cache dir into JSONL for parquet building:

```bash
python examples/data_preprocess/chart2code_train_dataset_factory/build_jsonl_from_cache_process_pool_worker_dir.py
```

Select the appropriate Parquet builder code file, and modify the `--local_dir` for destination of Parquet file:

```bash
# ChartStruct-1k with CoT
python examples/data_preprocess/verl_grpo_mllm_chart2code_ChartStruct_1k_dataset.py

# ChartStruct-2k with CoT
python examples/data_preprocess/verl_grpo_mllm_chart2code_ChartStruct_2k_dataset.py

# ChartStruct-3k with CoT
python examples/data_preprocess/verl_grpo_mllm_chart2code_combo_3k_dataset.py
python examples/data_preprocess/verl_grpo_mllm_chart2code_random_select_3k_dataset.py

# ChartStruct-3k without CoT
python examples/data_preprocess/verl_grpo_mllm_chart2code_combo_3k_dataset_RL_no_CoT_Ablation.py

# ChartStruct-4k with CoT
python examples/data_preprocess/verl_grpo_mllm_chart2code_ChartStruct_4k_dataset.py
python examples/data_preprocess/verl_grpo_mllm_chart2code_random_select_4k_dataset.py

# ChartStruct-4k without CoT
python examples/data_preprocess/verl_grpo_mllm_chart2code_ChartStruct_4k_dataset_RL_no_CoT_Ablation.py
```

---

## Training

We use **Qwen2.5-VL-7B-Instruct** as the backbone and optimize with GRPO.

### Launch Training

To reproduce the main experiment, run the following command
```bash
# Train Qwen2.5-VL-7B-Instruct with ChartStruct-4k
bash chart2code-mllm-grpo-Qwen2.5-VL-7B-Instruct-Ablation-scaling-ChartStruct_4k-select-charts.sh
```

However, important arguments in all `chart2code-*.sh` and their detailed meanings:

```bash
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.max_prompt_length=3000 \
    data.max_response_length=3000 \
    +actor_rollout_ref.actor.freeze_vision_tower=True \
    +actor_rollout_ref.rollout.freeze_vision_tower=True \
    +actor_rollout_ref.rollout.limit_images=1 \
    +actor_rollout_ref.rollout.limit_videos=0 \
    +actor_rollout_ref.rollout.limit_audios=0 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.n=16 \
    trainer.total_epochs=5 \
    ......
```
- **Core Algorithm Settings**
  - `algorithm.adv_estimator=grpo`: This parameter sets the advantage estimator to GRPO (Group Relative Policy Optimization), making it ideal for the chart-to-code task.
- **Data Loading & Sequence Length Configurations**
  - `data.train_batch_size=32`: Defines the batch size for training, representing the number of samples processed per training step. This value balances training efficiency, GPU memory usage, and training stability.
  - `data.max_prompt_length=3000`: Specifies the maximum token length for input prompts. Any input prompts exceeding this limit will be truncated to prevent out-of-memory errors and maintain computational efficiency.
  - `data.max_response_length=3000`: Controls the maximum token length for model-generated responses. This caps the output length to avoid overly long generations and aligns with the requirements of the chart-to-code task.
- **Multimodal Constraints**
  - `+actor_rollout_ref.actor/rollout.freeze_vision_tower=True`: Freezes the vision tower (visual encoder) of the multimodal model during the actor training phase. The vision encoder is already pre-trained on general visual data and retains fixed weights, reducing memory consumption and training complexity while preserving strong visual understanding capabilities.
  - `+actor_rollout_ref.rollout.limit_images=1`: Restricts each input sample to contain exactly one image. This setting is customized for chart-based tasks, as each training sample corresponds to a single chart image paired with textual prompts.
  - `+actor_rollout_ref.rollout.limit_videos=0`: Disables video input entirely, as the target chart-to-code task does not involve video data.
  - `+actor_rollout_ref.rollout.limit_audios=0`: Disables audio input entirely, since no audio modalities are used in this training pipeline.
- **Training Schedule**
  - `actor_rollout_ref.rollout.n=16`: Sets the number of candidate responses generated per input prompt during the rollout phase. This is a core hyperparameter for GRPO, as the algorithm uses 16 parallel outputs to compute group-based rewards and refine the model policy.
  - `trainer.total_epochs=5`: Defines the total number of full training epochs, meaning the model will iterate over the entire training dataset 5 times. This controls overall training duration and helps the model converge without overfitting to the training data.

**Note**: In lower version of vLLM rollout engine, the stuck may happen during actor model rollout, update the version of vLLM, or locally patch the vllm according to this patch PR:
https://github.com/vllm-project/vllm/pull/16371/changes


**Training Summary**:
- 5 epochs
- 8 × NVIDIA A100 (80GB)
- ~62 hours
- Visual encoder frozen, while only the MLP and Projector trained

---

## Project Structure (Modification from verl)

```
├───data-rl-parquet  # training data Parquet  
├───examples
│   └───data_preprocess  # Parquet builders
├───verl
│   └───utils
│       └───reward_score
│           └───chart2code_mllm  # the chart2code reward computing codes
│               └───lowlevel  # Chart Code Spec related files 
└ chart2code-*.sh  # the training scripts
```

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


## Citation

```
@misc{he2026chartspec4vlm,
      title={Chart Specification: Structural Representations for Incentivizing VLM Reasoning in Chart-to-Code Generation}, 
      author={Minggui He and Mingchen Dai and Jian Zhang and Yilun Liu and Shimin Tao and Pufan Zeng and Osamu Yoshie and Yuya Ieiri},
      year={2026},
      eprint={2602.10880},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.10880}, 
}
```