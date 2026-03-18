import numpy as np
import scipy
import math
import re

from typing import List, Optional

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


# monkey patch: 手动补上已被 numpy 高版本废弃掉的 asscalar 方法，使用 item 方法实现相同功能
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)


def _mean(vs: List[Optional[float]]) -> float:
    valid = [v for v in vs if v is not None]
    return float(sum(valid) / len(valid)) if valid else 1.0


def get_string_levenshtein_edit_distance(gt_str: str, pt_str: str) -> int:
    """
    计算 GT 字符串与 Predict 字符串之间的 Levenshtein 编辑距离
    """
    m, n = len(gt_str), len(pt_str)
    if m * n == 0:
        return m + n
    # f[i][j]: edit distance between word1[0..i] and word2[0..j]
    f = [ [0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        f[i][0] = i
    for j in range(n+1):
        f[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            d1 = f[i-1][j] + 1
            d2 = f[i][j-1] + 1
            d3 = f[i-1][j-1] + int(gt_str[i-1] != pt_str[j-1])
            f[i][j] = min(d1, d2, d3)
    return f[m][n]


def get_cosine_sim(u: list, v: list) -> float:
    """
    Compute a similarity score between two vectors, considering both
    their direction (cosine similarity) and magnitude ratio, and map
    the result to the range [0, 1].

    Parameters
    ----------
    u : array_like
        First input vector.
    v : array_like
        Second input vector.

    Returns
    -------
    float
        Similarity score in the range [0.0, 1.0].

    Notes
    -----
    - If both vectors are all zeros, similarity is defined as 1.0.
    - If only one vector is zero, similarity is defined as 0.0.
    - This metric is scale-invariant up to a global factor and
      penalizes both angular and magnitude discrepancies.
    """
    u, v = np.asarray(u, float), np.asarray(v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 and nv == 0:
        # 两个全零向量 → 定义为完全相似
        return 1.0
    if nu == 0 or nv == 0:
        # 只有一个零向量 → 完全不相似
        return 0.0
    sim = np.dot(u, v) / max(nu**2, nv**2)
    # 映射 [-1,1] → [0,1]
    return float((sim + 1) / 2.0)


def chamfer_distance(A: List[List[int|float]], B: List[List[int|float]]):
    """
    计算两个向量组的 Chamfer 距离（均值形式）
    A: (m, d)  B: (n, d)
    return: float
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, n = len(A), len(B)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float('inf')

    kdt_A = scipy.spatial.cKDTree(A)
    kdt_B = scipy.spatial.cKDTree(B)

    d1, _ = kdt_B.query(A, k=1)  # A->B 最近邻
    d2, _ = kdt_A.query(B, k=1)  # B->A 最近邻
    return float(d1.mean() + d2.mean())


def get_vector_group_chamfer_distance(vecs1, vecs2):
    if not vecs1 and not vecs2:
        return 1.0
    elif not vecs1 or not vecs2:
        return 0.0
    shape_score = get_val_similarity(len(vecs1), len(vecs2))
    if shape_score < 0.5:
        return shape_score / 2
    arr1 = np.array(vecs1)
    arr2 = np.array(vecs2)
    dist_matrix = scipy.spatial.distance.cdist(arr1, arr2)  # 每个vec1到vec2的距离
    min_dists1 = np.min(dist_matrix, axis=1)
    min_dists2 = np.min(dist_matrix, axis=0)
    cd = (np.mean(min_dists1) + np.mean(min_dists2)) / 2
    sim_score = float(1 / (1 + cd))
    return shape_score * 0.5 + sim_score * 0.5


def get_matrix_Chamfer_similarity(A: List[List[int|float]], B: List[List[int|float]]):
    """
    计算两个向量组的相似度（基于 Chamfer）
    """
    A = np.asarray(A, dtype=float)
    A = np.clip(A, -1e7, 1e7)
    B = np.asarray(B, dtype=float)
    B = np.clip(B, -1e7, 1e7)
    m, n = len(A), len(B)
    if m == 0 and n == 0:
        # 两个都是空列表，相似度为 1
        return 1.0
    elif m == 0 or n == 0:
        return 0.0
    base = chamfer_distance(A, B)
    sim = 1.0 / (1.0 + base)
    return math.sqrt(sim)


def get_histogram_similarity(bins_gt: List[int|float], bins_pt: List[int|float], tops_gt: List[int|float], tops_pt: List[int|float]) -> float:
    """
    判断两个数组形状是否一致 [0, 1.0]
    """
    if len(bins_gt) == len(bins_pt) == 0:
        return 1.0
    elif len(bins_gt) == 0 or len(bins_pt) == 0:
        return 0.0
    
    total_score = 0.0
    # step 1/3: 检查GT区间与 Predict区间重合度，必须 > 0.6 才算过关，占总分 30%
    def get_overlap_score() -> float:
        range_gt = (min(bins_gt), max(bins_gt))
        range_pt = (min(bins_pt), max(bins_pt))
        intersection_range_len = max(0, min(range_gt[1], range_pt[1]) - max(range_gt[0], range_pt[0]))
        union_range_len = max(range_gt[1], range_pt[1]) - min(range_gt[0], range_pt[0])
        if union_range_len == 0:
            return float(range_gt[0] == range_pt[0])
        return intersection_range_len / union_range_len

    overlap_score = get_overlap_score()
    total_score += overlap_score * 0.3
    if overlap_score <= 0.6:
        return total_score

    # step 2/3: 主峰位置相似度（Peak Position Similarity），偏移程度 > 0.6 才过关，占总分30%
    def get_peak_position_sim_score() -> float:
        def peak_bin_center(bins, tops):
            peak_idx = tops.index(max(tops))
            return (bins[peak_idx] + bins[peak_idx+1]) / 2
        peak_gt = peak_bin_center(bins_gt, tops_gt)
        peak_pt = peak_bin_center(bins_pt, tops_pt)
        peak_offset = abs(peak_gt - peak_pt)
        return np.exp(-peak_offset).tolist()  # 越近得分越高

    peak_offset_score = get_peak_position_sim_score()
    total_score += peak_offset_score * 0.3
    if peak_offset_score <= 0.6:
        return total_score

    # step 3/3: 归一化频率曲线的形状相似度（Distribution Shape Score）:将 tops 插值对齐到同样的 bin 序列后，用 余弦相似度 或 均方差 比较频率曲线形状
    # scipy.interpolate.interp1d()
    def get_shape_score_wasserstein(eps=1e-8):
        # Get bin centers
        def centers(bins):
            return 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))
        # 如果 tops_gt 或 tops_pt 为空，直接返回 0
        if not tops_gt or not tops_pt:
            return 0.0
        # 如果 tops_gt 或 tops_pt 全为 0，也返回 0
        if np.all(np.array(tops_gt) == 0) or np.all(np.array(tops_pt) == 0):
            return 0.0
        gt_x = centers(bins_gt)
        pt_x = centers(bins_pt)
        gt_pdf = np.array(tops_gt) / (np.sum(tops_gt) + eps)
        pt_pdf = np.array(tops_pt) / (np.sum(tops_pt) + eps)
        if np.sum(gt_pdf) <= 0 or not np.isfinite(np.sum(gt_pdf)) or np.sum(pt_pdf) <= 0 or not np.isfinite(np.sum(pt_pdf)):
            return 0.0
        dist = scipy.stats.wasserstein_distance(gt_x, pt_x, u_weights=gt_pdf, v_weights=pt_pdf)
        score = 1 / (1 + dist)  # Normalize to (0, 1)
        return score

    shape_score = get_shape_score_wasserstein()
    total_score += shape_score * 0.4
    return total_score


def get_string_similarity(gt_str: str, pt_str: str, eps: float = 1e-8) -> float:
    """
    计算 GT 字符串与 Predict字符串之间的相似度，范围在 [0, 1]
    """
    if gt_str is None:
        gt_str = ""
    if pt_str is None:
        pt_str = ""
    edit_distance = get_string_levenshtein_edit_distance(gt_str, pt_str)
    return 1 - edit_distance / max(len(gt_str), len(pt_str), eps)


def get_val_similarity(gt_val: float, pt_val: float) -> float:
    """
    计算两个数值的相似度，范围在 [0, 1]
    """
    if gt_val is None and pt_val is None:
        return 1.0
    elif gt_val is None or pt_val is None:
        return 0.0
    abs_diff = abs(gt_val - pt_val)
    return 1.0 / (1.0 + abs_diff)


def get_vector_L2_similarity(u: List[int|float], v: List[int|float]) -> float:
    """
    返回两个等长数组的L2距离相似度，范围 (0, 1]
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    dist = np.linalg.norm(u - v)   # L2 距离
    return 1.0 / (1.0 + dist)      # 距离越小，相似度越大


def get_list_similarity(list_gt: List[int|float], list_pt: List[int|float]) -> float:
    if not list_gt and not list_pt:
        return 1.0
    elif not list_gt or not list_pt:
        return 0.0
    
    len_gt, len_pt = len(list_gt), len(list_pt)
    shape_score = get_val_similarity(len_gt, len_pt)
    if shape_score < 0.8:
        return shape_score / 2
    
    sim_score_list = []
    for gt_val in list_gt:
        best_sim_score = 0.0
        for pt_val in list_pt:
            cur_sim_score = get_val_similarity(gt_val, pt_val)
            best_sim_score = max(best_sim_score, cur_sim_score)
        sim_score_list.append(best_sim_score)
    
    sim_score = _mean(sim_score_list)
    return _mean([shape_score, sim_score])


def get_lowlevel_color_similarity(color1: str, color2: str) -> float:
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    
    def rgb_to_lab(rgb):
        # Convert an RGB color to Lab color space. RGB values should be in the range [0, 255].
        # Create an sRGBColor object from RGB values
        rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
        # Convert to Lab color space
        lab_color = convert_color(rgb_color, LabColor)
        return lab_color
    
    def calculate_similarity_single(c1, c2):
        if c1.startswith("#") and c2.startswith("#"):
            c1 = hex_to_rgb(c1)
            c2 = hex_to_rgb(c2)
            lab1 = rgb_to_lab(c1)
            lab2 = rgb_to_lab(c2)
            return max(0, 1 - (delta_e_cie2000(lab1, lab2) / 100))
        elif c1.startswith("#") or c2.startswith("#"):
            return 0
        return float(c1 == c2)
    return calculate_similarity_single(color1, color2)


def normalize_size(val) -> float:
    if isinstance(val, str) and val.endswith('%'):
        return float(val.rstrip('%')) / 100
    return float(val)

def get_treemap_similarity(treemap_gt, treemap_pt):
    """
    treemap_gt / treemap_pt: 来自 code_ir_gt / code_ir_pt 的 treemap 字段（list）
    返回：一个0~1的分数
    """
    if not treemap_gt and not treemap_pt:
        return 1.0
    elif not treemap_gt or not treemap_pt:
        return 0.0

    gt_item, pt_item = treemap_gt[0], treemap_pt[0]
    gt_labels = [l for l in gt_item.get("label", [])]
    pt_labels = [l for l  in pt_item.get("label", [])]

    # Step 1: label F1
    def _unfold_treemap_set(treemap_list: list) -> set:
        res_set = set()
        for val in treemap_list:
            if isinstance(val, (list, tuple, set)):
                res_set.update(val)
            elif isinstance(val, (int, float, str, bool)):
                res_set.add(val)
        return res_set
    
    label_score = 0.0
    gt_set, pt_set = _unfold_treemap_set(gt_labels), _unfold_treemap_set(pt_labels)
    inter = gt_set & pt_set
    if not gt_set:
        label_score = 1.0
    elif not inter:
        label_score = 0.0
    else:
        precision = len(inter) / len(pt_set) if pt_set else 0.0
        recall = len(inter) / len(gt_set)
        label_score = 2 * precision * recall / (precision + recall)

    # Step 2: 如果 label 完全一致，再比较 size
    size_score = 0.0
    if gt_set == pt_set:
        gt_map = {l: normalize_size(s)
                  for l, s in zip(gt_item.get("label", []), gt_item.get("sizes", []))}
        pt_map = {l: normalize_size(s)
                  for l, s in zip(pt_item.get("label", []), pt_item.get("sizes", []))}
        diffs = [abs(gt_map[k] - pt_map[k]) for k in gt_set]
        rmse = np.sqrt(np.mean(np.square(diffs))) if diffs else 0.0
        size_score = max(0.0, 1.0 - float(rmse))

    # 可调整权重，目前简单平均
    return 0.5 * label_score + 0.5 * size_score


if __name__ == '__main__:字符串编辑距离相似度测试':
    test_gt_str = "The quick brown fox jumps over a lazy dog."
    test_pt_str = "The quick brown dog jumps over a lazy fox."
    print(get_string_similarity(test_gt_str, test_pt_str))


if __name__ == "__main__：向量 L2 距离相似度测试":
    test_nums1 = [1, 2, 3, 4, 5]
    test_nums2 = [1.05, 2, 3, 4, 5]
    print(get_vector_L2_similarity(test_nums1, test_nums2))


if __name__ == "__main__：向量组Chamfer相似度":
    test_A = [
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10],
    ]
    test_B = [
        [9.6, 0.1, 0.1],
        [0, 9.8, 0],
        [0, 0, 10],
    ]
    print(get_matrix_Chamfer_similarity(test_A, test_B))


if __name__ == "__main__：测试 来自 ChartMimic 的科技颜色相似度计算，阈值 0.7":
    test_color1 = "#0000FF"
    test_color2 = "#5555EE"
    print(get_lowlevel_color_similarity(test_color1, test_color2))


if __name__ == '__main__':
    test_vec1 = [(1, 1), (-1, 0), (1, 0)]
    test_vec2 = [(-1.01, 0), (1, 1.01)]
    test_vec3 = []
    test_vec4 = [(1, 1)]
    print(get_vector_group_chamfer_distance(test_vec1, test_vec2))
    print(get_vector_group_chamfer_distance(test_vec1, test_vec3))
    print(get_vector_group_chamfer_distance(test_vec1, test_vec4))
