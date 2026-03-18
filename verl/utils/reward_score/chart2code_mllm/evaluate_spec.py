# evaluate_spec.py
"""
将 GT-Spec 与 Pred-Spec 比对，输出 0–100 分 + 6 维子得分。
可独立调用：evaluate_spec(gt_dict, pred_dict) → dict
"""
import collections
import json
import yaml
from scipy.optimize import linear_sum_assignment
import numpy as np
import re
from typing import Dict, Any, List, Optional, Callable
import logging
logger = logging.getLogger(__file__[__file__.find("/verl/")+1:])

from . import qwen_v2_5_72b
from .lowlevel.code_ir_eval import get_code_ir_dict_from_subprocess_stdout
from .lowlevel.similarity_utils import get_lowlevel_color_similarity, get_matrix_Chamfer_similarity, get_vector_L2_similarity, get_vector_group_chamfer_distance, get_histogram_similarity, get_treemap_similarity, get_string_similarity, get_val_similarity, get_cosine_sim, get_list_similarity


spec_extraction_from_prompt_template = """/no_think
You are a large language model that specializes in extracting a structured chart specification
(Intermediate Representation, IR) directly from Python matplotlib or networkx code.

Your goal is to return a **single, valid YAML document** that accurately captures the chart’s:
- chart family (chart_type),
- panel count and layout,
- and data structure for each panel.

You will extract this information **only from the provided Python code**.
Never invent any value. If a value cannot be determined from the code, set it to `null`.
If there are multiple subplots, extract one panel object per subplot.

When values are defined by **literal lists/arrays or simple deterministic expressions**
(e.g., list comprehensions, `np.arange/linspace` with numeric bounds, `reshape` on a literal range),
you **must materialize** the numbers explicitly (do not keep variable names only).
Do not attempt to evaluate random or data-loading code.

---

# YAML OUTPUT SCHEMA (USE THIS EXACT STRUCTURE)

```yaml
chart_type: null           # string, one of the allowed chart families listed below
panel_count: null          # integer, total number of subplots
panel_layout: null         # [rows, cols], subplot grid arrangement

panels:                    # list of panel objects, one per subplot
  - title: null            # string or null, subplot title if set
    panel_type: null       # string or null. Only used when chart_type == "mix".
                           # Set to the specific chart family used in this subplot.
                           # Must be one of the same allowed values as `chart_type` (see lists below).
                           # If chart_type != "mix", panel_type MUST be null.

    # 1. Coordinate system -- strong classification signals
    coord: null            # "cartesian" | "polar" | "geographic" | "3d"

    # 2. Data inputs
    x_domain: null         # array (categories or [min, max]) or null
    y_domain: null         # array (categories or [min, max]) or null
    series: null           # array of legend/group labels or null

    # 3. Function-generated data
    y_function:
      expression: null     # string of the function expression from code (e.g., "np.sin(x)")
      x_range: null        # [min, max] if determinable from code; else null

    z_function:            # used for 3D analytic surface functions
      expression: null
      x_range: null
      y_range: null
    
    # 4. Manually coded data if not function-generated
    values: null            # array (2-D, list-of-pairs, or list per series) or null
                            # IMPORTANT: If chart data is generated from a y_function (y_function expression not null),
                            # then values MUST be set to null.

    z_values: null         # 2D numeric grid, used for heatmap-like charts or null
                           # IMPORTANT: If chart data is generated from a z_function (z_function expression not null),
                           # then z_values MUST be set to null.
```
---

# ALLOWED `chart_type` VALUES

## Common chart types (recognized directly from matplotlib/networkx code)
- "bar": discrete category comparison (vertical, horizontal, funnel).
- "line": continuous trends (line, multi-line, area).
- "scatter": discrete data points (point, bubble).
- "heatmap": values encoded as colored 2D grid.
- "pie": circular chart where wedge angle encodes value.
- "ring": donut chart (pie with a central hole via wedge width).
- "rose": polar bar chart (bars on polar axes where angle is category).
- "3d": analytic surface or grid-based 3D plot. Use `z_values` or `z_function`.
- "multi_axes": single subplot with multiple y-axes/twin axes (twinx/twiny) or secondary axes transforms.
- "mix": composite charts combining multiple types (e.g., line + bar in one axes).
- "density": 1D kernel density estimate (KDE) curves (e.g., via y_function expression `scipy.stats.gaussian_kde`).

## Specialized chart types (structure extracted separately)
- "radar": Polar chart with radial axes and spokes (spider plot).
- "boxplot": Distribution with min, q1, mean, q3, max.
- "violin": Distribution with min, median, max, density curves.
- "error": Charts with error bars/points via matplotlib `errorbar`.
- "graph": Network graph with nodes and edges.
- "quiver": Vector fields via matplotlib `quiver`.
- "treemap": Hierarchical part-to-whole with sizes and labels.
- "histogram": Binned distributions with bins and tops arrays.
- "contour": Contour plots with multiple level lines per series.

---

# EXTRACTION GUIDE (FROM CODE ONLY)

## CLASSIFICATION RULES (decide `chart_type`)

**bar**
- `plt.bar` / `ax.bar` → vertical bar
- `plt.barh` / `ax.barh` → horizontal bar
- Multiple `bar` calls in a loop on the same axes → still `"bar"` (multi-series)

**line**
- `plt.plot`, `ax.plot`, `ax.step`, `ax.stairs` → `"line"`
- `ax.fill_between`, `ax.fill_betweenx` → `"line"` (area/band)
- `ax.stackplot` → `"line"` (stacked area)
- Pandas/Series/DataFrame `.plot(kind="line"/"area")` → `"line"`

**density**
- If using a kernel density estimator **(e.g., `scipy.stats.gaussian_kde`)**, classify as `"density"`  
  even if rendered via `ax.plot(xs, density(xs))` or `ax.fill_between(xs, density(xs))`.
- Typical pattern: build KDE → create grid `xs` → evaluate → plot lines/filled curves.
- Multiple categories (looping over datasets) → still `"density"` with `series`.

**scatter**
- `plt.scatter`, `ax.scatter` → `"scatter"`
- Bubble charts using `s=` marker size → still `"scatter"`

**heatmap**
- `imshow`, `matshow`, `pcolormesh`, `pcolor` → `"heatmap"`

**pie / ring**
- `pie(...)` without hole → `"pie"`
- `pie(..., wedgeprops={"width": w>0})` or equivalent hole-setting → `"ring"`

**rose**
- On `projection="polar"`, `ax.bar(theta, radii, ...)` → `"rose"`

**3d**
- Any axes with `projection="3d"` → `"3d"`  
  (includes `plot_surface`, `plot_wireframe`, `contour3D`, `contourf`, `bar3d`, `scatter3D`, `plot_trisurf`)

**multi_axes**
- Subplot uses `twinx()`, `twiny()`, `secondary_y=True`, or a secondary axis transform → `"multi_axes"`
- **This is distinct from multiple subplots**:
  - Multiple subplots → separate axes objects → separate panels.
  - Multi-axes/twin axes → **one** subplot with secondary scale → single panel in IR.

**mix**
- **Within the same subplot/axes**, multiple families combined (e.g., `bar` + `line`) → `"mix"`
- **Across subplots with different families** → also `"mix"`
- When `"mix"`, each panel **must** set `panel_type` to its specific family from the allowed lists.

**Specialized**
- `nx.draw*` → `"graph"`
- `ax.quiver` → `"quiver"`
- `ax.boxplot` → `"boxplot"`
- `ax.violinplot` → `"violin"`
- `ax.errorbar` → `"error"` (**includes error “points” made via matplotlib errorbar**)
- `squarify.plot(...)` → `"treemap"`
- `ax.contour`, `ax.contourf` → `"contour"`
- `projection="polar"` with `ax.plot`/`ax.fill` forming radial polygons → `"radar"`

---

## TYPE-SPECIFIC MINIMUMS & FALLBACKS (avoid null-only panels)

**bar**
- Vertical `bar(x, y)`: `x_domain = x`, `values = y`
- Horizontal `barh(y, x)`: `y_domain = y`, `values = x`
- If only `y`: `x_domain = [0, len(y)-1]`, `values = y`
- Multi-series (looped bars or multiple calls):  
  `series = ["group1","group2",...]`, `values = [y_group1, y_group2, ...]`

**line**
- `plot(x, y)`: `x_domain = x`, `values = y`
- `plot(y)` only: `x_domain = [0, len(y)-1]`, `values = y`
- Multi-line (`label=` or multiple calls): `series = [labels...]`, `values = [y1, y2, ...]`
- `fill_between(x, y1[, y2])`: `x_domain = x`, `values = [y1, y2]` (if vs baseline 0 → `values = y1`)
- `fill_betweenx(y, x1[, x2])`: `y_domain = y`, `values = [x1, x2]` (or single x vs 0)
- `stackplot(x, y1, y2, ...)`: `x_domain = x`, `series = ["series1","series2",...]`, `values = [[y1], [y2], ...]`
- Pandas/DF plot rules:
  - If `x=` column specified → `x_domain = that column`
  - Else `x_domain = df.index` (if literal/simple)
  - Multi-column → `series = [colnames]`, `values = [col1_vals, col2_vals, ...]`
- Time axis (dates/datetimes): materialize literal dates (ISO strings) or set `[min,max]` if range-like

**density**
- Treat as a special sub-case of continuous curves (often built using `gaussian_kde` or similar).
- If a literal sampling grid (e.g. `xs = np.linspace(...)`) exists → set `x_domain = xs`  
  (if not fully determinable, at least record `[min, max]`).
- **Preferred:** If the density curve(s) are explicitly materialized in code
  (e.g., computed array like `curve = density(xs)`), extract them as  
  `values = curve` (or a list of curves per category).
- **Fallback:** If values cannot be reliably materialized, set  
  `y_function.expression = "gaussian_kde(...)"` (or the detected density function call)  
  to indicate that the y-axis is generated from this analytic density.
- If there are multiple categories (e.g., loop over several datasets) →  
  `series = [cat1, cat2, ...]` and either  
  `values = [curve1, curve2, ...]` **or** associate each with its own `y_function.expression`.

**scatter**
- `values = [[x1, y1], [x2, y2], ...]`
- If literal arrays provided:
  `x_domain = [min(x), max(x)]`, `y_domain = [min(y), max(y)]`
- If x is categorical and y numeric: `x_domain = x_categories`, `y_domain = [min(y), max(y)]`

**heatmap**
- Grid inputs (`imshow(Z)`, `pcolormesh(X,Y,Z)`, `matshow(Z)`):
  - If `Z` is literal or built from simple deterministic code → **materialize** `z_values = Z`
  - Do **NOT** put grid Z into `values` (keep `values = null`)
  - If ticks/extent are literal, set `x_domain`/`y_domain` accordingly; else null

**pie / ring**
- Map `labels` → `series`, sizes (1st positional array) → `values`
- `x_domain`/`y_domain` can remain null

**rose**
- `values = radii` (list of bar lengths)
- If labels provided: `series = labels`; else keep `series = null`
- If radii are numeric: `y_domain = [0, max(radii)]`

**3d**
- Treat **any** `projection="3d"` axes as `"3d"`
- Always capture Z data in `z_values` or analytic `z_function` (never in `values`)
  - Surfaces (`plot_surface`, `wireframe`, `contour3D`, `contourf`):
    - If a literal 2D array is passed → `z_values = Z`
    - If `Z = f(X,Y)` is defined → `z_function.expression = "f(X,Y)"` and set `x_range`/`y_range` if determinable
  - Bars (`bar3d(x, y, z0, dx, dy, dz)`):
    - Extract heights (`dz`) as primary data → store as a **1-D list** in `z_values`
    - Derive `x_domain`/`y_domain` from literal X/Y arrays or tick labels when possible
    - `values = null`
  - Scatter (`scatter3D(x, y, z, ...)` or `ax.scatter(x,y,z,...)`):
    - `z_values = [[x1, y1, z1], ...]`
    - If ranges are literal, fill `x_domain`/`y_domain`
  - Triangular/irregular meshes (`plot_trisurf`, etc.):
    - If literal Z provided → put into `z_values`
    - `values = null`

**error** (matplotlib `errorbar`)
- Classify as `"error"` whenever `ax.errorbar` or `plt.errorbar` is used (covers “error points” too)
- Central data → store in `values` (y or x depending on orientation)
- Error intervals → represent as `[lower, center, upper]` per point
  - If only symmetric error given (e.g., `yerr = 0.2`) → interpret as `center ± 0.2`
  - If asymmetric provided (2 arrays) → compute lower/upper accordingly
- If x-error is used (`xerr`), treat analogously and set `x_domain` when inferable

**multi_axes**
- Classify as `"multi_axes"` **when twinx/twiny/secondary axes are used**
- Still extract data according to the primitives drawn (bar/line/scatter/etc.)
- Secondary axes do **not** change how `values` is structured; keep **one panel** per subplot

**mix**
- If `"mix"`, each `panels[i].panel_type` **must** be filled with its specific family
- Apply the above rules within each panel based on its `panel_type`

**radar** (specialized)
- On `projection="polar"` with `ax.plot`/`ax.fill` forming radial polygons:
  - Set `chart_type = "radar"`, `coord = "polar"`
  - If category labels exist → `x_domain = labels`; else `x_domain = null`
  - If a single literal radial series is clearly present → `values = that list` (else keep `values = null`)

---

## DERIVING DOMAINS FROM LITERALS
- Numeric arrays → `[min(arr), max(arr)]`
- Categorical arrays → the literal list (e.g., `["A","B",...]`)
- If only counts appear (`len(y)`) → implicit index `[0, len(y)-1]`
- Datetime arrays (`pd.date_range`, literal date lists) → materialize as ISO strings or `[min,max]`

---

## MULTIPLE SUBPLOTS
- Compute `panel_count` from `plt.subplots(...)` or repeated `plt.subplot(...)`
- Set `panel_layout = [rows, cols]` from subplots args if present; otherwise infer from usage

---

## `panel_type` (only if `chart_type == "mix"`)
- For `"mix"`, **every panel must** set `panel_type` to one of the **same allowed values** as `chart_type`
  (including specialized types like "radar", "graph", "quiver", "3d", "density", etc.)
- Use the same detection rules as above based on the primitives used in that panel
- If `chart_type` ≠ `"mix"`, `panel_type` **must** be `null`

---

## OUTPUT RULES (STRICT YAML FORMAT)
- Return **ONLY** the YAML document. Surround with ```yaml and ```. No explanations.
- Use YAML 1.2 syntax with **two-space** indentation. **Never** use tabs.
- Top-level keys must be in the **exact order** shown in the schema.
- Represent unknown values as `null`.
- Numbers must be plain scalars (not quoted).
- **All string values MUST be wrapped in double quotes** (e.g., `"bar"` or `"Jan"`).
- Lists should be YAML arrays (e.g., `[1, 2, 3]`) or multi-line `- item` style.
- Do **not** omit any fields in the schema, even if they are `null`.

---

## INPUT

<code>
{code}
</code>

## FINAL ACTION

Output one complete and valid YAML document conforming to the schema above.

"""


def _mean(vs: List[Optional[float]]) -> float:
    valid = [v for v in vs if v is not None]
    return float(sum(valid) / len(valid)) if valid else 1.0


# ---------- 通用 field 评分 ---------- #
def _safe(
    gt: Any,
    pr: Any,
    fn: Callable[[Any, Any], float],
) -> Optional[float]:
    if gt is None and pr is None:
        return None
    if gt is None or pr is None:
        return 0
    try:
        return fn(gt, pr)
    except Exception:
        return 0


_exact = lambda g, p: 1.0 if g == p else 0.0
_rel05 = lambda g, p: 1.0 - min(abs(g - p) / max(abs(g), 1e-6) / 0.05, 1) * 2


def _jaccard(gs, ps) -> float:
    gs, ps = set(gs or []), set(ps or [])
    if not gs and not ps:
        return 1.0
    inter = len(gs & ps)
    union = len(gs | ps)
    return 2 * (inter / union) - 1.0


def _mse(ga, pa) -> float:
    ga, pa = np.asarray(ga), np.asarray(pa)
    if ga.shape != pa.shape:
        n = min(ga.size, pa.size)
        ga, pa = ga.flatten()[:n], pa.flatten()[:n]
    if ga.size == 0 or pa.size == 0:
        return 0.0
    mse = ((ga - pa) ** 2).mean()
    var = ga.var() + 1e-6
    return 1.0 - np.clip(mse / var, 0.0, 1.0) * 2


# ---------- 各维度打分 ---------- #
def _score_chart_type(gt: Dict, pr: Dict) -> float:
    try:
        s1 = _safe(gt.get("chart_type"), pr.get("chart_type"), _exact)
        return _mean([s1])
    except Exception as e:
        logger.error(f"failed to score chart_type:" + json.dumps({
            "predict": pr,
            "GT": gt,
        }, ensure_ascii=False), exc_info=True)
        return 0


def _score_layout(gt: Dict, pr: Dict) -> float:
    panel_count_score = _safe(gt.get("panel_count"), pr.get("panel_count"), _exact)
    if not panel_count_score:
        return 0.0
    panel_layout_score = _safe(gt.get("panel_layout"), pr.get("panel_layout"), _exact)
    return _mean([panel_layout_score])


def _score_boxplot_code_ir(boxplot_vectors_code_ir_gt: List[dict], boxplot_vectors_code_ir_pt: List[dict]) -> float:
    """
    计算 boxplot 的 Chamfer 距离相似度
    """
    def unpack_boxplot_dict_list(boxplot_dict_list: list) -> list:
        """
        将 boxplot 的字典形式转为 list，顺序为 [min, q1, median, q3, max]
        """
        res = []
        for boxplot_dict in boxplot_dict_list:
            if len(boxplot_dict) != 5:
                continue
            res.append([
                boxplot_dict["min"],
                boxplot_dict["q1"],
                boxplot_dict["median"],
                boxplot_dict["q3"],
                boxplot_dict["max"],
            ])
        return res
    gt_boxplot_list = unpack_boxplot_dict_list(boxplot_vectors_code_ir_gt)
    pt_boxplot_list = unpack_boxplot_dict_list(boxplot_vectors_code_ir_pt)

    if not gt_boxplot_list and not pt_boxplot_list:
        return 1.0
    elif not gt_boxplot_list or not pt_boxplot_list:
        return 0.0

    shape_score = get_val_similarity(len(gt_boxplot_list), len(pt_boxplot_list))
    if shape_score < 0.8:
        return shape_score / 2
    boxplot_sim_score_list = []
    for gt_arr in gt_boxplot_list:
        max_sim_score = 0.0
        for pt_arr in pt_boxplot_list:
            if max_sim_score >= 0.95:
                max_sim_score = 1.0
                break
            cur_sim_score = get_cosine_sim(gt_arr, pt_arr)
            max_sim_score = max(max_sim_score, cur_sim_score)
        boxplot_sim_score_list.append(max_sim_score)
    sim_score = _mean(boxplot_sim_score_list)
    return _mean([shape_score, sim_score])


def _score_color_code_ir(color_list_gt: List[str], color_list_pt: List[str]) -> float:
    """
    使用贪心匹配替代全排列匹配，降低时间复杂度防止运行卡死
    """
    def group_color(color_list):
        color_dict = collections.defaultdict(list)
        for color_str in color_list:
            chart_type, color_val = color_str.split("--")
            color_dict[chart_type].append(color_val)
        return color_dict

    assert isinstance(color_list_gt, list) and isinstance(color_list_pt, list)
    if not color_list_gt and not color_list_pt:
        return 1.0
    elif not color_list_gt or not color_list_pt:
        return 0.0

    shape_score = get_val_similarity(len(color_list_gt), len(color_list_pt))
    if shape_score < 0.5:
        # 颜色数量相差超过 50%，提前截断低分返回
        return shape_score / 2

    # 将GT/PT的 [f'{type}--{color_val}', ...] 列表拆解成 {type: color_val} 字典形式
    gt_colors_dict = group_color(color_list_gt)
    pt_colors_dict = group_color(color_list_pt)

    color_sim_score_list = []
    # 遍历每种 GT Type 名字
    for key_gt, category_color_list_gt in gt_colors_dict.items():
        # 遍历 GT Type 对应列表下的每个颜色
        for gt_color in category_color_list_gt:
            best_match_score = 0.0
            # 找GT 的 type 对应于 PT 中列表中最相似的颜色值，放入列表
            for color_pt in pt_colors_dict.get(key_gt, []):
                if best_match_score == 1.0:
                    break
                cur_sim_score = get_lowlevel_color_similarity(gt_color, color_pt)
                best_match_score = max(best_match_score, cur_sim_score)
            color_sim_score_list.append(best_match_score)
    
    color_sim_score = _mean(color_sim_score_list)
    return _mean([shape_score, color_sim_score])


def _score_contour_code_ir(contour_num_list_gt: List[int], contour_num_list_pt: List[int], eps=1e-8) -> float:
    """
    计算 Chart Code IR 中 contour 相似度：等高线条数之和应该相同
    """
    if not contour_num_list_gt and not contour_num_list_pt:
        return 1
    elif not contour_num_list_gt or not contour_num_list_pt:
        return 0
    shape_score = get_val_similarity(len(contour_num_list_gt), len(contour_num_list_pt))
    if shape_score < 0.8:
        return shape_score / 2
    sim_score = get_list_similarity(contour_num_list_gt, contour_num_list_pt)
    return _mean([shape_score, sim_score])


def _score_error_code_ir(err_dict_gt: dict, err_dict_pt: dict) -> float:
    if len(err_dict_gt) == len(err_dict_pt) == 0:
        return 1.0
    elif len(err_dict_gt) == 0 or len(err_dict_pt) == 0:
        return 0

    shape_score = get_val_similarity(len(err_dict_gt), len(err_dict_pt))
    if shape_score < 0.8:
        return shape_score / 2
    
    # 检查召回率为主，遍历每一个 GT error range，看能否在 predict 对应 key_name 找到最大相似度
    max_recall_sim_list = []
    for key_name, error_bar_dist_list_list in err_dict_gt.items():
        for gt_err_bar_list in error_bar_dist_list_list:
            max_sim_score = 0.0
            for pt_err_bar_list in err_dict_pt.get(key_name, []):
                if max_sim_score == 1.0:
                    break
                cur_sim_score = get_vector_L2_similarity(gt_err_bar_list, pt_err_bar_list)
                max_sim_score = max(max_sim_score, cur_sim_score)
            max_recall_sim_list.append(max_sim_score)
    sim_score = 0 if not max_recall_sim_list else _mean(max_recall_sim_list)
    return _mean([shape_score, sim_score])


def _score_graph_code_ir(graph_dict_gt: dict, graph_dict_pt: dict) -> float:
    """
    计算 graph 生成相似度，越简单越好，只检查顶点个数和边个数是否与GT一致（否则部分程序中使用 "name" 或 ID 会导致紊乱），返回 [0, 1]
        graph dict schema: {"nodes": [], "edges": []}
    """
    gt_num_nodes = len(graph_dict_gt.get("nodes", []))
    pt_num_nodes = len(graph_dict_pt.get("nodes", []))
    node_shape_score = get_val_similarity(gt_num_nodes, pt_num_nodes)
    if node_shape_score < 0.8:
        return node_shape_score / 2
    gt_num_edges = len(graph_dict_gt.get("edges", []))
    pt_num_edges = len(graph_dict_pt.get("edges", []))
    edge_shape_score = get_val_similarity(gt_num_edges, pt_num_edges)
    return _mean([node_shape_score, edge_shape_score])


def _score_histogram_code_ir(histogram_list_gt: dict, histogram_list_pt: dict) -> float:
    shape_score = get_val_similarity(len(histogram_list_gt), len(histogram_list_pt))
    if shape_score < 0.8:
        return shape_score / 2
    sim_score_list = []
    for bins_gt, tops_gt in histogram_list_gt:
        max_sim_score = 0
        for bins_pt, tops_pt in histogram_list_pt:
            cur_sim_score = get_histogram_similarity(bins_gt, bins_pt, tops_gt, tops_pt)
            max_sim_score = max(max_sim_score, cur_sim_score)
        sim_score_list.append(max_sim_score)
    sim_score = 0.0 if not sim_score_list else _mean(sim_score_list)
    return _mean([shape_score, sim_score])


def _score_violin_code_ir(violin_list_gt: list, violin_list_pt: list) -> float:
    violin_box_gt = [(violin_dict["min"], violin_dict["median"], violin_dict["max"]) for violin_dict in violin_list_gt]
    violin_box_pt = [(violin_dict["min"], violin_dict["median"], violin_dict["max"]) for violin_dict in violin_list_pt]
    shape_score = get_val_similarity(len(violin_box_gt), len(violin_box_pt))
    if shape_score < 0.8:
        return shape_score / 2
    sim_score = get_vector_group_chamfer_distance(violin_box_gt, violin_box_pt)
    return _mean([shape_score, sim_score])


def _score_polar_code_ir(polar_dict_gt: dict, polar_dict_pt: dict, eps: float = 1e-8) -> float:
    if not polar_dict_gt and not polar_dict_pt:
        return 1.0
    elif not polar_dict_gt or not polar_dict_pt:
        return 0.0
    
    # 1. 系列个数与 GT 一致
    shape_score = 1 - abs(len(polar_dict_gt) - len(polar_dict_pt)) / max(len(polar_dict_gt), len(polar_dict_pt), eps)
    if shape_score < 0.8:
        return shape_score / 2
    sim_score_list = []
    for key_gt, list_gt in polar_dict_gt.items():
        best_match_pt_key = ''
        max_sim_key = 0
        for key_pt in polar_dict_pt.keys():
            cur_sim_score = get_string_similarity(key_gt, key_pt)
            if cur_sim_score > max_sim_key:
                max_sim_key = cur_sim_score
                best_match_pt_key = key_pt
            # 2. 找到最相似的 key
            list_pt = polar_dict_pt.get(best_match_pt_key, [])
            if len(list_gt) == len(list_pt) > 0:
                # 3. 当数组形状一致，计算 predict 数组和 GT 数组的余弦相似度
                cur_sim_list = []
                for gt_val, pt_val in zip(list_gt, list_pt):
                    cur_val_score = 0.0
                    if isinstance(gt_val, (int, float)) and isinstance(pt_val, (int, float)):
                        cur_val_score = get_val_similarity(gt_val, pt_val)
                    elif isinstance(gt_val, str) and isinstance(pt_val, str):
                        cur_val_score = get_string_similarity(gt_val, pt_val)
                    cur_sim_list.append(cur_val_score)
                sim_score_list.append(_mean(cur_sim_list))

    cosine_sim_score = _mean(sim_score_list)
    final_polar_sim_score = shape_score * 0.5 + cosine_sim_score * 0.5
    return final_polar_sim_score


def _score_pie_code_ir(pie_dict_gt: dict, pie_dict_pt: dict) -> float:
    def _get_total_plot_list_num(pie_dict: dict) -> int:
        total_plot_ring_num = 0
        for val_list in pie_dict.values():
            total_plot_ring_num += len(val_list)
        return total_plot_ring_num
    
    # 判断形状是否相似（误差<=20%）
    gt_shape_len, pt_shape_len = _get_total_plot_list_num(pie_dict_gt), _get_total_plot_list_num(pie_dict_pt)
    shape_score = get_val_similarity(gt_shape_len, pt_shape_len)
    if shape_score < 0.8:
        return shape_score / 2

    sim_score_list = []
    for radius_str_gt, val_list_of_list_gt in pie_dict_gt.items():
        best_match_key_str = None
        best_match_key_sim_score = 0.0
        for radius_str_pt in pie_dict_pt.keys():
            try:
                cur_key_sim_score = get_val_similarity(float(radius_str_gt), float(radius_str_pt))
                if cur_key_sim_score > best_match_key_sim_score:
                    best_match_key_sim_score = cur_key_sim_score
                    best_match_key_str = radius_str_gt
            except Exception:
                pass
        if not best_match_key_str:
            continue
        val_list_of_list_pt = pie_dict_pt.get(best_match_key_str, [])
        for gt_list in val_list_of_list_gt:
            best_match_val_score = 0.0
            for pt_list in val_list_of_list_pt:
                cur_val_score = get_list_similarity(gt_list, pt_list)
                best_match_val_score = max(best_match_val_score, cur_val_score)
            sim_score_list.append(best_match_val_score)

    sim_score = _mean(sim_score_list)
    return _mean([shape_score, sim_score])


def _score_events_plot_code_ir(events_list_gt: list, events_list_pt: list):
    if not events_list_gt and not events_list_pt:
        return 1.0
    elif not events_list_gt or not events_list_pt:
        return 0.0
    
    shape_score = get_val_similarity(len(events_list_gt), len(events_list_pt))
    if shape_score < 0.8:
        return shape_score / 2
    
    sim_score_list = []
    for list_gt in events_list_gt:
        best_match_score = 0.0
        for list_pt in events_list_pt:
            cur_sim_score = get_list_similarity(list_gt, list_pt)
            best_match_score = max(best_match_score, cur_sim_score)
        sim_score_list.append(best_match_score)
    sim_score = _mean(sim_score_list)
    return _mean([shape_score, sim_score])


def _score_text_str_pos_code_ir(text_dict_gt: dict, text_dict_pt: dict) -> float:
    text_list_gt, text_list_pt = text_dict_gt.get("str", []), text_dict_pt.get("str", [])
    pos_list_gt, pos_list_pt = text_dict_gt.get("pos", []), text_dict_pt.get("pos", [])
    if not text_list_gt and not text_list_pt:
        return 1.0
    elif not text_list_gt or not text_list_pt:
        return 0.0
    
    assert len(text_list_gt) == len(pos_list_gt) and len(text_list_pt) == len(pos_list_pt)
    shape_score = get_val_similarity(len(text_list_gt), len(text_list_pt))
    if shape_score < 0.8:
        return shape_score / 3
    
    text_sim_score_list = []
    pos_sim_score_list = []
    for idx_gt, text_str_gt in enumerate(text_list_gt):
        best_match_idx = -1
        best_text_match_sim_score = 0.0
        best_pos_sim_score = 0.0
        for idx_pt, text_str_pt in enumerate(text_list_pt):
            current_sim_score = get_string_similarity(text_str_gt, text_str_pt)
            if current_sim_score >= best_text_match_sim_score:
                best_match_idx = idx_pt
                cur_pos_sim_score = get_vector_L2_similarity(pos_list_gt[idx_gt], pos_list_pt[best_match_idx])
                best_pos_sim_score = max(best_pos_sim_score, cur_pos_sim_score)
                best_text_match_sim_score = current_sim_score

        text_sim_score_list.append(best_text_match_sim_score)
        pos_sim_score_list.append(best_pos_sim_score)
    
    text_sim_score = _mean(text_sim_score_list)
    pos_sim_score = _mean(pos_sim_score_list)
    return _mean([shape_score, text_sim_score, pos_sim_score])


def _score_code_ir(code_ir_gt: dict, code_ir_pt: dict) -> Optional[float]:
    """
    计算 Code IR(结构如下) 的 RL Reward，返回 [0, 100]

        ```yaml
        color: [颜色属性字符串（哪一个方法的哪种颜色）]

        boxplot:
          - [min, q1, mean, q3, max]
        
        contour:[ 每个系列的 contour 图线条的个数]
        
        error:
          元素名:  # 每个元素的数组误差允许范围，可能有多个
            - [y_err_lower, y, y_err_upper]  # y误差允许最小值，y值，y误差允许最大值
        
        graph:  # 每个字段直接用交并比 IoU相似度
          nodes: [顶点]
          labels:[顶点标签]
          edges:[边连接方式]
          edge_labels:[边标签]
        
        histogram:
          - [ bins_list, tops_list] # 划分区间 bins 方式和区间频数统计
        
        quiver:
          - [delta_x, delta_y]  # 每个向量的坐标：终点减去起点
        
        treemap:
          sizes: []  # 每个元素大小（依次对应）
          label: []  # 每个元素标签（依次对应）
        
        violin:
          # 每个数据
          - min: 小提琴下界,
            median: 小提琴中位数,
            max: 小提琴上界,
            coords: []        # 长轴方向坐标
            vals: []          # 长轴方向的小提琴宽度

        polar:
          key_name: []        # 每个 key_name 对应的值列表

        # 记录绘制的文本和相对像素位置 {"str": ["foo", "bar"], "pos": [(0.2, 0.2), (0.3, 0.3)]}
        text:
          str: []            # 每个文本对应字符串
          pos: []            # 每个文本对应位置
        
        ```
    """
    final_score_dict = {}

    # boxplot 打分
    boxplot_gt, boxplot_pt = code_ir_gt.get("boxplot", []), code_ir_pt.get("boxplot", [])
    if len(boxplot_gt) > 0:
        final_score_dict["boxplot"] = _score_boxplot_code_ir(boxplot_gt, boxplot_pt)
        if np.isnan(final_score_dict["boxplot"]):
            logger.error("boxplot code IR score should not be NaN")
            final_score_dict["boxplot"] = 0

    # contour 打分
    contour_num_list_gt, contour_num_list_pt = code_ir_gt.get("contour", []), code_ir_pt.get("contour", [])
    if len(contour_num_list_gt) > 0:
        final_score_dict["contour"] = _score_contour_code_ir(contour_num_list_gt, contour_num_list_pt)
        if np.isnan(final_score_dict["contour"]):
            logger.error("contour code IR score should not be NaN")
            final_score_dict["contour"] = 0

    # errorbar 打分
    err_dict_gt, err_dict_pt = code_ir_gt.get("error", {}), code_ir_pt.get("error", {})
    if len(err_dict_gt) > 0:
        final_score_dict["error"] = _score_error_code_ir(err_dict_gt, err_dict_pt)
        if np.isnan(final_score_dict["error"]):
            logger.error("errorbar code IR score should not be NaN")
            final_score_dict["error"] = 0

    # graph 打分
    DEFAULT_GRAPH_IR_SCHEMA = {"nodes": [], "edges": []}
    graph_dict_gt, graph_dict_pt = code_ir_gt.get("graph", DEFAULT_GRAPH_IR_SCHEMA), code_ir_pt.get("graph", DEFAULT_GRAPH_IR_SCHEMA)
    if sum(len(v) for v in graph_dict_gt.values()) > 0:
        final_score_dict["graph"] = _score_graph_code_ir(graph_dict_gt, graph_dict_pt)
        if np.isnan(final_score_dict["graph"]):
            logger.error("graph code IR score should not be NaN")
            final_score_dict["graph"] = 0

    # histogram 打分
    hist_list_gt, hist_list_pt = code_ir_gt.get("histogram", []), code_ir_pt.get("histogram", [])
    if len(hist_list_gt) > 0:
        final_score_dict["histogram"] = _score_histogram_code_ir(hist_list_gt, hist_list_pt)
        if np.isnan(final_score_dict["histogram"]):
            logger.error("histogram code IR score should not be NaN")
            final_score_dict["histogram"] = 0

    # quiver 打分  这里是不是需要考虑一下gt失效的情况,都给满分这样就可以跳过这个case？
    quiver_list_gt, quiver_list_pt = code_ir_gt.get("quiver", []), code_ir_pt.get("quiver", [])
    if len(quiver_list_gt) > 0 or len(quiver_list_pt) > 0:
        final_score_dict["quiver"] = get_vector_group_chamfer_distance(quiver_list_gt, quiver_list_pt)
        if np.isnan(final_score_dict["quiver"]):
            logger.error("quiver code IR score should not be NaN")
            final_score_dict["quiver"] = 0

    # treemap 打分
    treemap_gt, treemap_pt = code_ir_gt.get("treemap", []), code_ir_pt.get("treemap", [])
    if len(treemap_gt) > 0:
        final_score_dict["treemap"] = get_treemap_similarity(treemap_gt, treemap_pt)
        if np.isnan(final_score_dict["treemap"]):
            logger.error("treemap code IR score should not be NaN")
            final_score_dict["treemap"] = 0

    # violin 打分：简化，只比较 (min, median, max)
    violin_list_gt, violin_list_pt = code_ir_gt.get("violin", []), code_ir_pt.get("violin", [])
    if len(violin_list_gt) > 0 or len(violin_list_pt) > 0:
        final_score_dict["violin"] = _score_violin_code_ir(violin_list_gt, violin_list_pt)
        if np.isnan(final_score_dict["violin"]):
            logger.error("violin code IR score should not be NaN")
            final_score_dict["violin"] = 0

    # radar 打分：首先比较系列个数，再比较每个系列内的元素个数，最后使用余弦相似度比较
    polar_dict_gt, polar_dict_pt = code_ir_gt.get("polar", {}), code_ir_pt.get("polar", {})
    if len(polar_dict_gt) > 0:
        final_score_dict["polar"] = _score_polar_code_ir(polar_dict_gt, polar_dict_pt)
        if np.isnan(final_score_dict["polar"]):
            logger.error("polar code IR score should not be NaN")
            final_score_dict["polar"] = 0

    # ring/pie/donut 打分
    pie_dict_gt, pie_dict_pt = code_ir_gt.get("ring/pie", {}), code_ir_pt.get("ring/pie", {})
    if len(pie_dict_gt) > 0:
        final_score_dict["ring/pie"] = _score_pie_code_ir(pie_dict_gt, pie_dict_pt)
        if np.isnan(final_score_dict["ring/pie"]):
            logger.error("ring/pie code IR score should not be NaN")
            final_score_dict["ring/pie"] = 0

    # event plot 打分
    events_list_gt, events_list_pt = code_ir_gt.get("events", []), code_ir_pt.get("events", [])
    if len(events_list_gt) > 0:
        final_score_dict["events"] = _score_events_plot_code_ir(events_list_gt, events_list_pt)
        if np.isnan(final_score_dict["events"]):
            logger.error("events code IR score should not be NaN")
            final_score_dict["events"] = 0

    if len(final_score_dict) > 0:
        return sum(final_score_dict.values()) / len(final_score_dict)
    return None


def _score_domain_labels_or_ranges(domain_or_range_gt, domain_or_range_pt) -> Optional[float]:
    if domain_or_range_gt is None and domain_or_range_pt is None:
        return 1.0
    elif domain_or_range_gt is None or domain_or_range_pt is None:
        return 0.0
    if not isinstance(domain_or_range_gt, list) or not isinstance(domain_or_range_pt, list):
        return 0.0
    shape_score = get_val_similarity(len(domain_or_range_gt), len(domain_or_range_pt))
    if shape_score < 0.8:
        return shape_score / 2

    sim_score_list = []
    for gt_val in domain_or_range_gt:
        cur_max_sim = 0.0
        for pt_val in domain_or_range_pt:
            if gt_val == pt_val:
                cur_max_sim = 1.0
                break
            elif isinstance(gt_val, str) and isinstance(pt_val, str):
                cur_max_sim = max(cur_max_sim, get_string_similarity(gt_val, pt_val))
            elif isinstance(gt_val, (int, float)) and isinstance(pt_val, (int, float)):
                cur_max_sim = max(cur_max_sim, get_val_similarity(gt_val, pt_val))
            elif isinstance(gt_val, list) and isinstance(pt_val, list):
                # 二维矩阵，递归变为一维 list 处理
                cur_max_sim = max(cur_max_sim, _score_domain_labels_or_ranges(gt_val, pt_val))
        sim_score_list.append(cur_max_sim)
    sim_score = _mean(sim_score_list)
    return _mean([shape_score, sim_score])


def _score_data(gp: Dict, pp: Dict) -> float:
    """
    计算 LLM IR + Code IR 相似度，范围 [0, 1]
    """
    def _score_single_panel(gt_panel_dict: dict, pt_panel_dict: dict) -> float:
        """
        计算一个 panel 与另一个 panel 相似度，范围 [0, 1]
        """
        total_score_list = []

        panel_type_gt, panel_type_pt = gt_panel_dict.get("panel_type", ""), pt_panel_dict.get("panel_type", "")
        panel_type_score = get_string_similarity(panel_type_gt, panel_type_pt)
        if np.isnan(panel_type_score):
            logger.error("panel_type_score should not be NaN")
            panel_type_score = 0
        total_score_list.append(panel_type_score)
        # 提前截断返回
        if panel_type_score < 0.5:
            return panel_type_score / 13

        title_gt, title_pt = gt_panel_dict.get("title", ""), pt_panel_dict.get("title", "")
        title_score = get_string_similarity(title_gt, title_pt)
        if np.isnan(title_score):
            logger.error("title_score should not be NaN")
            title_score = 0
        total_score_list.append(title_score)
        # 提前截断返回
        if title_score < 0.7:
            return (title_score + panel_type_score) / 13

        coord_gt, coord_pt = gt_panel_dict.get("coord", ""), pt_panel_dict.get("coord", "")
        coord_score = get_string_similarity(coord_gt, coord_pt)
        if np.isnan(coord_score):
            logger.error("coord_score should not be NaN")
            coord_score = 0
        total_score_list.append(coord_score)

        x_domain_gt, x_domain_pt = gt_panel_dict.get("x_domain", None), pt_panel_dict.get("x_domain", None)
        x_domain_score = _score_domain_labels_or_ranges(x_domain_gt, x_domain_pt)
        if np.isnan(x_domain_score):
            logger.error("x_domain_score should not be NaN")
            x_domain_score = 0
        total_score_list.append(x_domain_score)

        y_domain_gt, y_domain_pt = gt_panel_dict.get("y_domain", None), pt_panel_dict.get("y_domain", None)
        y_domain_score = _score_domain_labels_or_ranges(y_domain_gt, y_domain_pt)
        if np.isnan(y_domain_score):
            logger.error("y_domain_score should not be NaN")
            y_domain_score = 0
        total_score_list.append(y_domain_score)

        series_gt, series_pt = gt_panel_dict.get("series", []), pt_panel_dict.get("series", [])
        series_score = _score_domain_labels_or_ranges(series_gt, series_pt)
        if np.isnan(series_score):
            logger.error("series_score should not be NaN")
            series_score = 0
        total_score_list.append(series_score)

        values_gt, values_pt = gt_panel_dict.get("values", []), pt_panel_dict.get("values", [])
        values_score = _score_domain_labels_or_ranges(values_gt, values_pt)
        if np.isnan(values_score):
            logger.error("values_score should not be NaN")
            values_score = 0
        total_score_list.append(values_score)

        DEFAULT_Y_FUNCTION = {"expression": None, "x_range": None}
        y_function_gt, y_function_pt = gt_panel_dict.get("y_function", DEFAULT_Y_FUNCTION), pt_panel_dict.get("y_function", DEFAULT_Y_FUNCTION)
        if not y_function_gt:
            y_function_gt = DEFAULT_Y_FUNCTION
        if not y_function_pt:
            y_function_pt = DEFAULT_Y_FUNCTION
        y_expression_gt, y_expression_pt = y_function_gt.get("expression", ""), y_function_pt.get("expression", "")
        y_expression_score = get_string_similarity(y_expression_gt, y_expression_pt)
        if np.isnan(y_expression_score):
            logger.error("y_expression_score should not be NaN")
            y_expression_score = 0
        total_score_list.append(y_expression_score)
        x_range_gt, x_range_pt = y_function_gt.get("x_range", None), y_function_pt.get("x_range", None)
        x_range_score = _score_domain_labels_or_ranges(x_range_gt, x_range_pt)
        if np.isnan(x_range_score):
            logger.error("x_range_score should not be NaN")
            x_range_score = 0
        total_score_list.append(x_range_score)

        DEFAULT_Z_FUNCTION = {"expression": None, "x_range": None, "y_range": None}
        z_function_gt, z_function_pt = gt_panel_dict.get("z_function", DEFAULT_Z_FUNCTION), pt_panel_dict.get("z_function", DEFAULT_Z_FUNCTION)
        if not z_function_gt:
            z_function_gt = DEFAULT_Z_FUNCTION
        if not z_function_pt:
            z_function_pt = DEFAULT_Z_FUNCTION
        z_expression_gt, z_expression_pt = z_function_gt.get("expression", ""), z_function_pt.get("expression", "")
        z_expression_score = get_string_similarity(z_expression_gt, z_expression_pt)
        if np.isnan(z_expression_score):
            logger.error("z_expression_score should not be NaN")
            z_expression_score = 0
        total_score_list.append(z_expression_score)
        x_range_gt, x_range_pt = z_function_gt.get("x_range", None), z_function_pt.get("x_range", None)
        x_range_score = _score_domain_labels_or_ranges(x_range_gt, x_range_pt)
        if np.isnan(x_range_score):
            logger.error("x_range_score should not be NaN")
            x_range_score = 0
        total_score_list.append(x_range_score)
        y_range_gt, y_range_pt = z_function_gt.get("y_range", None), z_function_pt.get("y_range", None)
        y_range_score = _score_domain_labels_or_ranges(y_range_gt, y_range_pt)
        if np.isnan(y_range_score):
            logger.error("y_range_score should not be NaN")
            y_range_score = 0
        total_score_list.append(y_range_score)

        z_values_gt, z_values_pt = gt_panel_dict.get("z_values", None), pt_panel_dict.get("z_values", None)
        z_values_score = _score_domain_labels_or_ranges(z_values_gt, z_values_pt)
        if np.isnan(z_values_score):
            logger.error("z_values_score should not be NaN")
            z_values_score = 0
        total_score_list.append(z_values_score)

        return _mean(total_score_list)

    final_score_list = []
    # 计算可用的 Code IR 得分
    if gp.get("code_ir") is not None and pp.get("code_ir") is not None:
        code_ir_score = _score_code_ir(gp["code_ir"], pp["code_ir"])
        if not code_ir_score:
            # 计算结果是 None，表示无可用 code_ir，跳过
            pass
        elif not isinstance(code_ir_score, (float, int)) or np.isnan(code_ir_score):
            # 得出 NaN，需要打印完整堆栈，计入最低分
            logger.error("code_ir_score should not be float or integer and should not be NaN", exc_info=True)
            code_ir_score = 0
        else:
            # 正常情况：加入结果列表
            final_score_list.append(code_ir_score)

    gt_panels_list, pt_panels_list = gp.get("panels", []), pp.get("panels", [])
    for gt_panel_dict in gt_panels_list:
        max_cur_score = 0.0
        for pt_panel_dict in pt_panels_list:
            if max_cur_score == 1.0:
                break
            cur_score = _score_single_panel(gt_panel_dict, pt_panel_dict)
            if np.isnan(cur_score):
                logger.error("any of panel scores should not be NaN")
                cur_score = 0
            max_cur_score = max(max_cur_score, cur_score)
        final_score_list.append(max_cur_score)

    return _mean(final_score_list)


def get_spec_from_matplotlib_code(code_str: str, max_retries: int = 1) -> dict:
    def unpack_yaml_codefence(text: str) -> str:
        matches = re.findall(r"```yaml\n(.*?)\n```", text, flags=re.DOTALL)
        return matches[0].strip() if matches else ''

    prompt = spec_extraction_from_prompt_template.replace(r'{code}', code_str)

    for _ in range(max_retries):
        try:
            raw_response_str = qwen_v2_5_72b.get_qwen_v2_5_72b_answer(prompt)
            yaml_str = unpack_yaml_codefence(raw_response_str).strip()
            assert len(yaml_str) > 0
            llm_ir_dict = yaml.safe_load(yaml_str)
            assert len(llm_ir_dict) > 0 and "panel_count" in llm_ir_dict
            # 当 LLM IR 中 panel_count 为 1 且 panel_layout 为 null 时，手动覆盖成 [1, 1]，防止 Reward 混淆
            if llm_ir_dict["panel_count"] == 1 and llm_ir_dict.get("panel_layout") is None:
                llm_ir_dict["panel_layout"] = [1, 1]
            return llm_ir_dict
        except Exception as e:
            continue
    logger.error(f'[chart2code] retrun empty IR: max_retries={max_retries} exceeded for qwen2.5 vl 72b to extract IR.')
    return {}


def evaluate_chart2code_spec_by_ir_new(python_code_str: str, spec_dict: Dict, code_ir_dict: Dict = {}, mock_ir_dict_inject: Dict = None) -> Dict[str, float]:
    """
    分阶段给分，总分范围   [0, 10]
    - chart_type         1分        （占比 10%），若错误提前返回
    - layout             1分        （占比 10%），若错误提前返回
    - text               1分        （占比 10%）
    - color              1分        （占比 10%）
    - LLM IR + Code IR   6分        （占比 60%）
    """
    gt = spec_dict
    
    # 记录每个维度得分
    total_spec_score_dict = {
        "score": 0.0,
        "details": dict(),
    }

    # 调用 Qwen 2.5 72B 后取得 IR1
    pt = None
    if not mock_ir_dict_inject:
        pt = get_spec_from_matplotlib_code(python_code_str)
        if not pt:
            total_spec_score_dict["score"] = 0.0
            return total_spec_score_dict
        # 从编译检查捕获的 code_ir 拼接入 predict 字典
        pt["code_ir"] = code_ir_dict    # 拼接 Code IR
    else:
        pt = mock_ir_dict_inject


    # 第一层 chart_type 占比 10%
    chart_type_score = _score_chart_type(gt, pt)
    if np.isnan(chart_type_score):
        logger.error("chart_type score should not be NaN")
        chart_type_score = 0
    total_spec_score_dict["details"]["type"] = chart_type_score
    total_spec_score_dict["score"] += chart_type_score * 1
    if chart_type_score < 1.0:
        return total_spec_score_dict

    # 第二层 layout （panel_count, panel_layout）占比 10%
    layout_score = _score_layout(gt, pt)
    if np.isnan(layout_score):
        logger.error("layout_score should not be NaN")
        layout_score = 0
    total_spec_score_dict["details"]["layout"] = layout_score
    total_spec_score_dict["score"] += layout_score * 1
    if layout_score < 1.0:
        return total_spec_score_dict

    code_ir_gt, code_ir_pt = gt.get("code_ir", {}), pt.get("code_ir", {})

    # 第三层 text 占比 10%
    # text 文本内容和位置打分：遍历 GT str 中找最相似的 PT str 和 位置，计算相似度
    DEFAULT_TEXT_DICT_SCHEMA = {"str": [], "pos": []}
    text_dict_gt, text_dict_pt = code_ir_gt.get("text", DEFAULT_TEXT_DICT_SCHEMA), code_ir_pt.get("text", DEFAULT_TEXT_DICT_SCHEMA)
    text_sim_score = 1.0
    if len(text_dict_gt["str"]) == len(text_dict_gt["pos"]) > 0:
        text_sim_score = _score_text_str_pos_code_ir(text_dict_gt, text_dict_pt)
        if np.isnan(text_sim_score):
            logger.error("text code IR score should not be NaN")
            text_sim_score = 0
    total_spec_score_dict["details"]["text"] = text_sim_score
    total_spec_score_dict["score"] += text_sim_score * 1
    
    # 第四层 color 占比 10%
    colors_gt, colors_pt = code_ir_gt.get("color", []), code_ir_pt.get("color", [])
    color_sim_score = 1.0
    if colors_gt:
        color_sim_score = _score_color_code_ir(colors_gt, colors_pt)
        if np.isnan(color_sim_score):
            logger.error("color code IR score should not be NaN")
            color_sim_score = 0
    total_spec_score_dict["details"]["color"] = color_sim_score
    total_spec_score_dict["score"] += color_sim_score * 1

    # 第四层 LLM IR + Code IR 准确率 占比 60%
    data_score = _score_data(gt, pt)
    if np.isnan(data_score):
        logger.error("data_score should not be NaN")
        data_score = 0
    total_spec_score_dict["details"]["data"] = data_score
    total_spec_score_dict["score"] += data_score * 6

    return total_spec_score_dict
