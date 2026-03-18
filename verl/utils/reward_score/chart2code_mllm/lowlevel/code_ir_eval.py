from typing import List, Tuple
from itertools import permutations
from multiprocessing import Pool
import subprocess
import tempfile
import os

import json
import re

import logging

logger = logging.getLogger(__file__)


python_interpreter_filepath = '/home/MY_USERNAME/miniconda3/envs/chartmimic/bin/python'

temp_base_dir = '/dev/shm'


process_pool_per_rollout = 2

import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# This is a patch for color map, which is not updated for newer version of numpy
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb):
    """
    Convert an RGB color to Lab color space.
    RGB values should be in the range [0, 255].
    """
    # Create an sRGBColor object from RGB values
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    
    # Convert to Lab color space
    lab_color = convert_color(rgb_color, LabColor)
    
    return lab_color   

def calculate_similarity_single(c1, c2):
    if c1.startswith("#") and c2.startswith("#"):
        # c1 = rgb2lab(np.array([hex_to_rgb(c1)]))
        # c2 = rgb2lab(np.array([hex_to_rgb(c2)]))
        c1 = hex_to_rgb(c1)
        c2 = hex_to_rgb(c2)
        lab1 = rgb_to_lab(c1)
        lab2 = rgb_to_lab(c2)
        # return max(0, 1 - deltaE_cie76(c1, c2)[0] / 100)
        return max(0, 1 - (delta_e_cie2000(lab1, lab2)/100) )
    elif not c1.startswith("#") and not c2.startswith("#"):

        return 1 if c1 == c2 else 0
    else:
        return 0
    
def filter_color(color_list):
    filtered_color_list = []
    len_color_list = len(color_list)
    for i in range(len_color_list):
        if i != 0:
            put_in = True
            for item in filtered_color_list:
                similarity = calculate_similarity_single(color_list[i].split("--")[1], item.split("--")[1])
                if similarity > 0.7:
                    put_in = False
                    break
            if put_in:
                filtered_color_list.append(color_list[i])
        else:
            filtered_color_list.append(color_list[i])
    # print("Filtered color list: ", filtered_color_list)
    return filtered_color_list

def group_color(color_list):
    color_dict = {}

    for color in color_list:
        chart_type = color.split("--")[0]
        color = color.split("--")[1]

        if chart_type not in color_dict:
            color_dict[chart_type] = [color]
        else:
            color_dict[chart_type].append(color)

    return color_dict



def calculate_similarity_for_permutation(args):
    shorter, perm = args
    current_similarity = sum(calculate_similarity_single(c1, c2) for c1, c2 in zip(shorter, perm))
    return current_similarity

class ColorEvaluator:
    def __init__(self) -> None:
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

    def _log_colors(self, code_file):
        """
        Get text objects of the code
        """

        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", "_log_colors.txt")
        suffix = self._get_suffix(output_file)
        code = prefix + code + suffix

        code_log_texts_file = code_file.replace(".py", "_log_colors.py")
        with open(code_log_texts_file, 'w') as f:
            f.write(code)
        # 调用 conda 环境解释器进行 color spec 计算
        os.system(f"{python_interpreter_filepath} {code_log_texts_file}")
        if os.path.exists(output_file) == True:
            with open(output_file, 'r') as f:
                colors = f.read()
                colors = eval(colors)
            os.remove(output_file)
        else:
            colors = []
        if os.path.exists(code_log_texts_file):
            os.remove(code_log_texts_file)
        return colors


    def get_log_colors(self, code_file):
        return self._log_colors(code_file)


    def _calculate_metrics(self, generation_colors: List[Tuple], golden_colors: List[Tuple]):
        generation_colors = list(generation_colors)
        golden_colors = list(golden_colors)

        group_generation_colors = group_color(generation_colors)
        group_golden_colors = group_color(golden_colors)

        def calculate_similarity_parallel(lst1, lst2):
            if len(lst1) == 0 or len(lst2) == 0:
                return 0
            shorter, longer = (lst1, lst2) if len(lst1) <= len(lst2) else (lst2, lst1)
            perms = permutations(longer, len(shorter))
            with Pool(processes=process_pool_per_rollout) as pool:
                similarities = pool.map(calculate_similarity_for_permutation, [(shorter, perm) for perm in perms])
            return max(similarities)

        # merge keys in group_generation_colors and group_golden_colors
        merged_color_group = list( set( list(group_generation_colors.keys()) + list(group_golden_colors.keys()) ) )
        for color in merged_color_group:
            if color not in group_generation_colors:
                group_generation_colors[color] = []
            if color not in group_golden_colors:
                group_golden_colors[color] = []
        
        max_set_similarity = 0

        for color in merged_color_group:
            max_set_similarity += calculate_similarity_parallel(group_generation_colors[color], group_golden_colors[color])

        # [Mingchen Debugging]: 改进 ChartMimic Lowlevel Color 得分计算脚本
        #  - 只计算 precision，
        #  - 当计算 precision 时候的 分母prediction为空时候，不再直接返回0，而是先判断 GT 是否也为空，若GT也为空则改为返回1
        if len(generation_colors) > 0:
            self.metrics["precision"] = max_set_similarity / len(generation_colors)
        else:
             return 1.0 if len(golden_colors) == 0 else 0
        return self.metrics["precision"]


    def _get_prefix(self):
        return """
# -*- coding: utf-8 -*-
import functools  # 使用 wraps 注解保留 wrapper 包裹方法原有的元数据，包括 __name__, __doc__, __module__ 等

import matplotlib.pyplot as plt
import matplotlib

import collections
import logging
logging.basicConfig(format='[%(processName)s:%(threadName)s] %(asctime)s %(levelname)s %(name)s:%(lineno)d %(funcName)s - %(message)s', level=logging.INFO)
mingchen_logger = logging.getLogger("get_ir_dict.py")


#####################################################################################################################
#  开始拦截 matplotlib 记录前缀信息
#####################################################################################################################

tweaked_mingchen_intermediate_ir_dict = {
    # 记录 boxplot 图 五个分位点
    "boxplot": [],  # each of which is: [{"min": 4.2, "q1": 4.5, "median": 4.8, "q3": 4.9, "max": 5.0}, ...]

    # 记录 violin 图 min, median, max, vals, coords
    "violin": [],  # each of which is: [{"min": 4.2, "median": 4.5, "max": 4.8, "vals": [], "coords": []}, ...]

    # 记录 histogram 的最重要参数： (bins_tuple, tops_tuple)
    "histogram": [],  # each of which is: [(bins_tuple, tops_tuple), ...]

    # 记录 treemap 图的 sizes 和 labels
    "treemap": [],  # each of which is: [{"sizes": [], "label": []}, ...]

    # 记录 quiver 图的向量坐标 [(delta_x, delta_y)]，之所以不存完整的坐标是因为担心图片发生漂移
    "quiver": [],  # each of which is: [  [delta_x, delta_y], ...... ]

    # 记录 graph 图（networkx） 图，最后处理时候把 dict of set 转成 dict of list
    "graph": {
        "nodes": set(),
        "edges": set(),
        "labels": set(),
        "edge_labels": set(),
    },

    # 记录 contour 每个系列等高线的条数
    "contour": set(),

    # 记录 error point 图  {"{x}": [(y_lower, y, y_upper), ...], ... }
    "error": collections.defaultdict(list),  # 记录每个 自变量对应的 取值范围 (y_lower, y, y_upper)

    # 记录 极坐标 PolarAxes 类下使用 ax.plot 的值
    "polar": collections.defaultdict(list),

    # 记录 ring/pie 数据
    "ring/pie": collections.defaultdict(list),

    # 记录 EventPlot 值，计算 list 相似度 [[1, 3, 5], [2, 4], [6]]
    "events": [],

    # 记录绘制的文本和相对像素位置 {"str": ["foo", "bar"], "pos": [(0.2, 0.2), (0.3, 0.3)]}
    "text": {
        "str": [],
        "pos": [],
    },
}


import squarify

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json

import networkx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.axes._base import _process_plot_var_args
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import networkx.drawing.nx_pylab as nx_pylab
from matplotlib.projections.polar import PolarAxes
from matplotlib.image import NonUniformImage
from matplotlib.patches import Ellipse, Circle
from matplotlib_venn._common import VennDiagram
import inspect

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# 下面继续补充 名桂&明辰 IR
from matplotlib import cbook    # mingchen debugging 明辰：记录 boxplot 和 小提琴图的统计信息
from matplotlib.contour import QuadContourSet  # mingchen debugging 明辰：记录 contour 和 contourf 的环的信息
# This is a patch for color map, which is not updated for newer version of numpy
def patch_asscalar(a):
    return a.item()


setattr(np, "asscalar", patch_asscalar)


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
        # c1 = rgb2lab(np.array([hex_to_rgb(c1)]))
        # c2 = rgb2lab(np.array([hex_to_rgb(c2)]))
        c1 = hex_to_rgb(c1)
        c2 = hex_to_rgb(c2)
        lab1 = rgb_to_lab(c1)
        lab2 = rgb_to_lab(c2)
        # return max(0, 1 - deltaE_cie76(c1, c2)[0] / 100)
        return max(0, 1 - (delta_e_cie2000(lab1, lab2) / 100))
    elif not c1.startswith("#") and not c2.startswith("#"):
        return 1 if c1 == c2 else 0
    else:
        return 0


def filter_color(color_list):
    filtered_color_list = []
    len_color_list = len(color_list)
    for i in range(len_color_list):
        if i != 0:
            put_in = True
            for item in filtered_color_list:
                similarity = calculate_similarity_single(color_list[i].split("--")[1], item.split("--")[1])
                if similarity > 0.7:
                    put_in = False
                    break
            if put_in:
                filtered_color_list.append(color_list[i])
        else:
            filtered_color_list.append(color_list[i])
    # print("Filtered color list: ", filtered_color_list)
    return filtered_color_list


def group_color(color_list):
    color_dict = {}

    for color in color_list:
        chart_type = color.split("--")[0]
        color = color.split("--")[1]

        if chart_type not in color_dict:
            color_dict[chart_type] = [color]
        else:
            color_dict[chart_type].append(color)

    return color_dict


drawed_colors = []
drawed_objects = {}
in_decorator = False


def convert_color_to_hex(color):
    'Convert color from name, RGBA, or hex to a hex format.'
    try:
        # First, try to convert from color name to RGBA to hex
        if isinstance(color, str):
            # Check if it's already a hex color (start with '#' and length either 7 or 9)
            if color.startswith('#') and (len(color) == 7 or len(color) == 9):
                return color.upper()
            else:
                return mcolors.to_hex(mcolors.to_rgba(color)).upper()
        # Then, check if it's in RGBA format
        elif isinstance(color, (list, tuple, np.ndarray)) and (len(color) == 4 or len(color) == 3):
            return mcolors.to_hex(color).upper()
        else:
            raise ValueError("Unsupported color format")
    except ValueError as e:
        print(color)
        print("Error converting color:", e)
        return None


def log_function_specific_for_draw_networkx_labels(func):
    @functools.wraps(func)
    def wrapper(
            G,
            pos,
            labels=None,
            font_size=12,
            font_color="k",
            font_family="sans-serif",
            font_weight="normal",
            alpha=None,
            bbox=None,
            horizontalalignment="center",
            verticalalignment="center",
            ax=None,
            clip_on=True,
    ):
        global drawed_colors
        global in_decorator
        global tweaked_mingchen_intermediate_ir_dict

        if in_decorator == False:
            in_decorator = True

            func_name = inspect.getfile(func) + "/" + func.__name__

            result = func(
                G,
                pos,
                labels=labels,
                font_size=font_size,
                font_color=font_color,
                font_family=font_family,
                font_weight=font_weight,
                alpha=alpha,
                bbox=bbox,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                ax=ax,
                clip_on=clip_on
            )

            for item in result.values():
                color = convert_color_to_hex(item.get_color())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = item

            in_decorator = False
        else:
            return func(
                G,
                pos,
                labels=labels,
                font_size=font_size,
                font_color=font_color,
                font_family=font_family,
                font_weight=font_weight,
                alpha=alpha,
                bbox=bbox,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                ax=ax,
                clip_on=clip_on
            )

        node_label_list = list(G.nodes.keys())
        tweaked_mingchen_intermediate_ir_dict["graph"]["labels"].update(node_label_list)
        return result

    return wrapper


def log_function_specific_for_draw_networkx_edges(func):
    @functools.wraps(func)
    def wrapper(
            G,
            pos,
            edgelist=None,
            width=1.0,
            edge_color="k",
            style="solid",
            alpha=None,
            arrowstyle=None,
            arrowsize=10,
            edge_cmap=None,
            edge_vmin=None,
            edge_vmax=None,
            ax=None,
            arrows=None,
            label=None,
            node_size=300,
            nodelist=None,
            node_shape="o",
            connectionstyle="arc3",
            min_source_margin=0,
            min_target_margin=0,
    ):
        global drawed_colors
        global in_decorator
        global tweaked_mingchen_intermediate_ir_dict

        if in_decorator == False:
            in_decorator = True

            func_name = inspect.getfile(func) + "/" + func.__name__

            result = func(
                G,
                pos,
                edgelist=edgelist,
                width=width,
                edge_color=edge_color,
                style=style,
                alpha=alpha,
                arrowstyle=arrowstyle,
                arrowsize=arrowsize,
                edge_cmap=edge_cmap,
                edge_vmin=edge_vmin,
                edge_vmax=edge_vmax,
                ax=ax,
                arrows=arrows,
                label=label,
                node_size=node_size,
                nodelist=nodelist,
                node_shape=node_shape,
                connectionstyle=connectionstyle,
                min_source_margin=min_source_margin,
                min_target_margin=min_target_margin
            )

            if type(result) == list:
                for line in result:
                    color = convert_color_to_hex(line.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            else:
                for item in result.get_edgecolors().tolist():
                    color = convert_color_to_hex(item)
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result  # ! Attention

            in_decorator = False
        else:
            return func(
                G,
                pos,
                edgelist=edgelist,
                width=width,
                edge_color=edge_color,
                style=style,
                alpha=alpha,
                arrowstyle=arrowstyle,
                arrowsize=arrowsize,
                edge_cmap=edge_cmap,
                edge_vmin=edge_vmin,
                edge_vmax=edge_vmax,
                ax=ax,
                arrows=arrows,
                label=label,
                node_size=node_size,
                nodelist=nodelist,
                node_shape=node_shape,
                connectionstyle=connectionstyle,
                min_source_margin=min_source_margin,
                min_target_margin=min_target_margin
            )

        edge_set = set()
        for u, edge_dict in G.adj.items():
            for v, _ in edge_dict.items():
                edge_set.add((u, v))
        tweaked_mingchen_intermediate_ir_dict["graph"]["edges"].update(edge_set)
        return result

    return wrapper


# mingchen debugging 明辰：拦截 networkx 的 draw_networkx_edge_labels 方法，记录 graph edge labels 信息
def log_function_specific_for_draw_networkx_edge_labels(func):
    @functools.wraps(func)
    def wrapper(G, *args, **kwargs):
        global in_decorator
        global tweaked_mingchen_intermediate_ir_dict

        if not in_decorator:
            in_decorator = True
            func_name = inspect.getfile(func) + "/" + func.__name__
            result = func(G, *args, **kwargs)
            in_decorator = False
        else:
            return func(G, *args, **kwargs)

        edge_label_set = set()
        for u, adj_dict in G.adj.items():
            for v, edge_label_dict in adj_dict.items():
                edge_label_set.add((u, v, str(edge_label_dict)))
        tweaked_mingchen_intermediate_ir_dict["graph"]["edge_labels"].update(edge_label_set)
        return result

    return wrapper


def log_function_specific_for_draw_networkx_nodes(func):
    @functools.wraps(func)
    def wrapper(
            G,
            pos,
            nodelist=None,
            node_size=300,
            node_color="#1f78b4",
            node_shape="o",
            alpha=None,
            cmap=None,
            vmin=None,
            vmax=None,
            ax=None,
            linewidths=None,
            edgecolors=None,
            label=None,
            margins=None,
    ):
        global drawed_colors
        global in_decorator
        global tweaked_mingchen_intermediate_ir_dict

        if in_decorator == False:
            in_decorator = True

            func_name = inspect.getfile(func) + "/" + func.__name__

            result = func(
                G,
                pos,
                nodelist=nodelist,
                node_size=node_size,
                node_color=node_color,
                node_shape=node_shape,
                alpha=alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                linewidths=linewidths,
                edgecolors=edgecolors,
                label=label,
                margins=margins
            )
            color = None
            for item in result.get_facecolors().tolist():
                color = convert_color_to_hex(item)
                drawed_colors.append(func_name + "--" + color)
            if color is not None:
                # 若 color 有值，则保留最后一个 color 的键。
                drawed_objects[func_name + "--" + color] = result

            in_decorator = False
        else:
            return func(
                G,
                pos,
                nodelist=nodelist,
                node_size=node_size,
                node_color=node_color,
                node_shape=node_shape,
                alpha=alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                linewidths=linewidths,
                edgecolors=edgecolors,
                label=label,
                margins=margins
            )

        node_list = list(G.nodes())
        tweaked_mingchen_intermediate_ir_dict["graph"]["nodes"].update(node_list)
        return result

    return wrapper


def log_function_for_3d(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global drawed_colors
        global in_decorator

        if in_decorator == False:
            in_decorator = True

            func_name = inspect.getfile(func) + "/" + func.__name__

            result = func(*args, **kwargs)

            if func.__name__ == "scatter":
                # check whether cmap is used
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    if type(kwargs["cmap"]) == str:
                        drawed_colors.append(func_name + "_3d--" + kwargs["cmap"])
                        drawed_objects[func_name + "_3d--" + kwargs["cmap"]] = result
                    else:
                        drawed_colors.append(func_name + "_3d--" + kwargs["cmap"].name)
                        drawed_objects[func_name + "_3d--" + kwargs["cmap"].name] = result
                else:
                    for item in result.get_facecolors().tolist():
                        color = convert_color_to_hex(item)
                        drawed_colors.append(func_name + "_3d--" + color)
                    drawed_objects[func_name + "_3d--" + color] = result  # ! Attention
            elif func.__name__ == "plot":
                for line in result:
                    color = convert_color_to_hex(line.get_color())
                    drawed_colors.append(func_name + "_3d--" + color)
                    drawed_objects[func_name + "_3d--" + color] = line
            elif func.__name__ == "plot_surface":
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    if type(kwargs["cmap"]) == str:
                        drawed_colors.append(func_name + "_3d--" + kwargs["cmap"])
                        drawed_objects[func_name + "_3d--" + kwargs["cmap"]] = result
                    else:
                        drawed_colors.append(func_name + "_3d--" + kwargs["cmap"].name)  # ! Attention
                        drawed_objects[func_name + "_3d--" + kwargs["cmap"].name] = result
                else:
                    colors = result.get_facecolors().tolist()
                    drawed_colors.append(func_name + "_3d--" + convert_color_to_hex(colors[0]))
                    drawed_objects[func_name + "_3d--" + convert_color_to_hex(colors[0])] = result  # ! Attention
            elif func.__name__ == "bar3d":
                colors = result.get_facecolors().tolist()
                drawed_colors.append(func_name + "_3d--" + convert_color_to_hex(colors[0]))
                drawed_objects[func_name + "_3d--" + convert_color_to_hex(colors[0])] = result  # ! Attention
            elif func.__name__ == "bar":
                for item in result:
                    color = convert_color_to_hex(item.get_facecolor())
                    drawed_colors.append(func_name + "_3d--" + color)
                    drawed_objects[func_name + "_3d--" + color] = item
            elif func.__name__ == "add_collection3d":
                colors = result.get_facecolors().tolist()
                for color in colors:
                    drawed_colors.append(func_name + "_3d--" + convert_color_to_hex(color))
                drawed_objects[func_name + "_3d--" + convert_color_to_hex(color)] = result

            in_decorator = False
        else:
            return func(*args, **kwargs)
        return result

    return wrapper


def log_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global drawed_colors
        global in_decorator
        global tweaked_mingchen_intermediate_ir_dict

        func_name = inspect.getfile(func) + "/" + func.__name__

        # 函数签名绑定，统一处理 *args 和 **kwargs
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()  # 会把形参默认值灌进来（包括模块常量引用）

        # 只有 boxplot_stats 和 violin_stats 原方法调用不需要防止递归，其他要防止
        # mingchen_logger.info(f"intercepted func name: {inspect.getfile(func)}/{func.__name__}")
        if func.__name__ == "boxplot_stats":
            raw_boxplot_stats_result = func(*args, **kwargs)
            for boxplot_stat_dict in raw_boxplot_stats_result:
                tweaked_mingchen_intermediate_ir_dict["boxplot"].append({
                    "min": float(boxplot_stat_dict["whislo"]),
                    "q1": float(boxplot_stat_dict["q1"]),
                    "median": float(boxplot_stat_dict["med"]),
                    "q3": float(boxplot_stat_dict["q3"]),
                    "max": float(boxplot_stat_dict["whishi"]),
                })
            return raw_boxplot_stats_result
        elif func.__name__ == "violin_stats":
            raw_violin_stats_result = func(*args, **kwargs)
            for violin_stat_dict in raw_violin_stats_result:
                tweaked_mingchen_intermediate_ir_dict["violin"].append({
                    "min": float(violin_stat_dict["min"]),
                    "median": float(violin_stat_dict["median"]),
                    "max": float(violin_stat_dict["max"]),
                    "vals": list(violin_stat_dict["vals"]),
                    "coords": list(violin_stat_dict["coords"]),
                })
            return raw_violin_stats_result
        elif func.__name__ == "annotate":
            # 处理特殊 quiver 图
            raw_result = func(*args, **kwargs)
            arrow_props_dict = bound.arguments.get("arrowprops")
            xytext_tuple = bound.arguments.get("xytext")
            xy_tuple = bound.arguments.get("xy")
            is_valid_quiver = True
            if not xytext_tuple or not xy_tuple or not arrow_props_dict or "arrowstyle" not in arrow_props_dict:
                is_valid_quiver = False
            if is_valid_quiver and arrow_props_dict["arrowstyle"] in ("<-", "<|-", "->", "-|>"):
                x2, y2 = bound.arguments.get("xytext", (0, 0))
                x1, y1 = bound.arguments.get("xy", (0, 0))
                if isinstance(x1, str) or isinstance(x2, str) or isinstance(y1, str) or isinstance(y2, str):
                    is_valid_quiver = False
                delta_x = delta_y = 0
                if is_valid_quiver and arrow_props_dict["arrowstyle"] in ("<-", "<|-"):
                    # 看箭头方向，(x1, y1)是起点坐标
                    delta_x = x2 - x1
                    delta_y = y2 - y1
                elif is_valid_quiver and arrow_props_dict["arrowstyle"] in ("->", "-|>"):
                    # 看箭头方向，(x2, y2)是起点点坐标
                    delta_x = x1 - x2
                    delta_y = y1 - y2
                # 尽可能减少RL噪声，只保存向量坐标 (delta_x, delta_y)
                if is_valid_quiver:
                    tweaked_mingchen_intermediate_ir_dict["quiver"].append([float(delta_x), float(delta_y)])
            return raw_result

        if in_decorator == False:
            in_decorator = True
            result = func(*args, **kwargs)

            if func.__name__ == "_makeline":
                color = convert_color_to_hex(result[1]["color"])
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result[0]
            elif func.__name__ == "axhline":
                color = convert_color_to_hex(result.get_color())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "axvline":
                color = convert_color_to_hex(result.get_color())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "_fill_between_x_or_y":
                color = convert_color_to_hex(list(result.get_facecolors()[0]))
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "bar":
                for item in result:
                    color = convert_color_to_hex(list(item._original_facecolor))
                    drawed_colors.append(func_name + "--" + color)
                    drawed_objects[func_name + "--" + color] = item
            elif func.__name__ == "scatter" and type(args[0]) != PolarAxes:
                # check whether cmap is used
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    if type(kwargs["cmap"]) == str:
                        drawed_colors.append(func_name + "--" + kwargs["cmap"])
                        drawed_objects[func_name + "--" + kwargs["cmap"]] = result
                    else:
                        drawed_colors.append(func_name + "--" + kwargs["cmap"].name)  # ! Attention
                        drawed_objects[func_name + "--" + kwargs["cmap"].name] = result
                else:
                    if len(result.get_facecolor()) != 0:
                        color = convert_color_to_hex(list(result.get_facecolor()[0]))
                        drawed_colors.append(func_name + "--" + color)
                        drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "pie":
                # 记录 pie/ring 数据
                raw_plot_data = bound.arguments.get("x", [])
                if isinstance(raw_plot_data, list) and len(raw_plot_data) > 1:
                    total_val = sum(raw_plot_data)
                    normalized_plot_data = []
                    for x in raw_plot_data:
                        normalized_plot_data.append(float(x / total_val))
                    cur_radius_distance = float(bound.arguments.get("radius", 0.0))
                    tweaked_mingchen_intermediate_ir_dict["ring/pie"][str(cur_radius_distance)].append(normalized_plot_data)

                for item in result[0]:
                    color = convert_color_to_hex(item.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
                    drawed_objects[func_name + "--" + color] = item
            elif func.__name__ == "axvspan":
                color = convert_color_to_hex(result.get_facecolor())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "axhspan":
                color = convert_color_to_hex(result.get_facecolor())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "hlines":
                for item in result.get_edgecolors():
                    color = convert_color_to_hex(list(item))
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result  # ! Attention
            elif func.__name__ == "vlines":
                for item in result.get_edgecolors():
                    color = convert_color_to_hex(list(item))
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result  # ! Attention
            elif func.__name__ == "boxplot":
                for item in result["boxes"]:
                    if type(item) == matplotlib.patches.PathPatch:
                        color = convert_color_to_hex(list(item.get_facecolor()))
                        drawed_colors.append(func_name + "--" + color)
                        drawed_objects[func_name + "--" + color] = item  # ! Attention
            elif func.__name__ == "violinplot":
                for item in result["bodies"]:
                    color = convert_color_to_hex(list(item.get_facecolor()[0]))
                    drawed_colors.append(func_name + "--" + color)
                    drawed_objects[func_name + "--" + color] = item  # ! Attention
            elif func.__name__ == "hist":
                tops, bins, patches = result
                # 保存 histogram 划分的bins 情况和 每一个 bin 的高度 tops 列表情况，转成 tuple of tuple 进行 set 去重
                tuple_bins = tuple(np.array(bins).tolist())
                tops_ndarr = np.array(tops).tolist()
                if isinstance(tops_ndarr, list) and len(tops_ndarr) > 0:
                    if isinstance(tops_ndarr[0], list):
                        # 如果 tops 是二维数组，用 extend 方法直接加入
                        for tuple_tops in tops_ndarr:
                            tweaked_mingchen_intermediate_ir_dict["histogram"].append([tuple_bins, tuple_tops])
                    else:
                        # 如果 tops 是一维数组，则用 append 加入
                        tweaked_mingchen_intermediate_ir_dict["histogram"].append([tuple_bins, tops_ndarr])
                if type(patches) != matplotlib.cbook.silent_list:
                    for item in patches:
                        color = convert_color_to_hex(list(item.get_facecolor()))
                        drawed_colors.append(func_name + "--" + color)
                        drawed_objects[func_name + "--" + color] = item
                else:
                    for container in patches:
                        for item in container:
                            color = convert_color_to_hex(list(item.get_facecolor()))
                            drawed_colors.append(func_name + "--" + color)
                            drawed_objects[func_name + "--" + color] = item
            elif func.__name__ == "quiver":
                # 首先收集所有的quiver向量坐标
                quiver_args = bound.arguments.get("args")
                if len(quiver_args) >= 4:
                    # 忽略掉 X, Y 和 末尾的C，只取出 delta_x, delta_y
                    _, _, delta_x_grid, delta_y_grid, *_ = quiver_args
                    for delta_x, delta_y in zip(np.array(delta_x_grid).ravel(), np.array(delta_y_grid).ravel()):
                        tweaked_mingchen_intermediate_ir_dict["quiver"].append([float(delta_x), float(delta_y)])
                # 然后继续收集quiver颜色属性
                for item in result.get_facecolors().tolist():
                    color = convert_color_to_hex(item)
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result  # ! Attention
            elif func.__name__ == "plot" and len(args) > 0 and type(args[0]) == PolarAxes:
                lines = result
                for line in lines:
                    # 首先记录极坐标数据
                    label_name = str(line.get_label())
                    y_data = np.array(line.get_ydata()).tolist()
                    tweaked_mingchen_intermediate_ir_dict["polar"][label_name] = y_data
                    # 继续处理颜色
                    color = convert_color_to_hex(line.get_color())
                    # print("color", color)
                    drawed_colors.append(func_name + "_polar" + "--" + color)
                    drawed_objects[func_name + "_polar" + "--" + color] = line
            elif func.__name__ == "scatter" and type(args[0]) == PolarAxes:
                # check whether cmap is used
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    if type(kwargs["cmap"]) == str:
                        drawed_colors.append(func_name + "_polar" + "--" + kwargs["cmap"])
                        drawed_objects[func_name + "_polar--" + kwargs["cmap"]] = result
                    else:
                        drawed_colors.append(func_name + "_polar" + "--" + kwargs["cmap"].name)
                        drawed_objects[func_name + "_polar" + "--" + kwargs["cmap"].name] = result
                else:
                    if len(result.get_facecolor()) != 0:
                        color = convert_color_to_hex(list(result.get_facecolor()[0]))
                        drawed_colors.append(func_name + "_polar" + "--" + color)
                        drawed_objects[func_name + "_polar" + "--" + color] = result  # ! Attention
            elif func.__name__ == "plot" and "squarify" in func_name:
                # 直接从wrapped 方法中里拿变量值
                treemap_sizes = bound.arguments.get("sizes", [])
                treemap_labels = bound.arguments.get("label", [])
                tweaked_mingchen_intermediate_ir_dict["treemap"].append({
                    "sizes": list(treemap_sizes),
                    "label": list(treemap_labels),
                })

                # get ax
                ax = result
                # get container
                containers = ax.containers
                for container in containers:
                    for item in container:
                        color = convert_color_to_hex(list(item.get_facecolor()))
                        drawed_colors.append(func_name + "_squarify" + "--" + color)
                        drawed_objects[func_name + "_squarify" + "--" + color] = item
            elif func.__name__ == "imshow":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" + colormap] = result  # ! Attention
            elif func.__name__ == "pcolor":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" + colormap] = result  # ! Attention
            elif func.__name__ == "contour":
                # 记录等高线个数列表
                num_contour_line = sum(len(coll.get_paths()) for coll in result.collections)
                tweaked_mingchen_intermediate_ir_dict["contour"].add(num_contour_line)
                # 继续记录颜色信息
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" + colormap] = result  # ! Attention
            elif func.__name__ == "contourf":
                # 记录等高线个数列表
                num_contour_line = sum(len(coll.get_paths()) for coll in result.collections)
                tweaked_mingchen_intermediate_ir_dict["contour"].add(num_contour_line)
                # 继续记录颜色信息
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" + colormap] = result  # ! Attention
            elif func.__name__ == "eventplot":
                positions = bound.arguments.get("positions", [])
                positions = np.array(positions, dtype=object).tolist()
                tweaked_mingchen_intermediate_ir_dict["events"].extend(positions)
            elif func.__name__ == "fill":
                patches = result
                for patch in patches:
                    color = convert_color_to_hex(list(patch.get_facecolor()))
                    drawed_colors.append(func_name + "--" + color)
                    drawed_objects[func_name + "--" + color] = patch
            elif func.__name__ == "errorbar":
                hasx_err, hasy_err = result.has_xerr, result.has_yerr
                data_line, caplines, barlinecols = result.lines
                for x, y, seg in zip(data_line.get_xdata(), data_line.get_ydata(), barlinecols[-1].get_segments()):
                    if hasy_err:
                        # （默认情况）纵轴是值，横轴是名称
                        _, err_low = seg[0].tolist()
                        _, err_high = seg[1].tolist()
                        label_name = str(x)
                        val = float(y)
                        tweaked_mingchen_intermediate_ir_dict["error"][label_name].append((err_low, val, err_high))
                    elif hasx_err:
                        # 横轴是值，纵轴是名称
                        err_low, _ = seg[0].tolist()
                        err_high, _ = seg[1].tolist()
                        label_name = str(y)
                        val = float(x)
                        tweaked_mingchen_intermediate_ir_dict["error"][label_name].append((err_low, val, err_high))
            elif func.__name__ == "__init__" and type(args[0]) == NonUniformImage:
                colormap = args[0].get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" + colormap] = args[0]
            elif func.__name__ == "broken_barh":
                colors = result.get_facecolors().tolist()
                for color in colors:
                    drawed_colors.append(func_name + "--" + convert_color_to_hex(color))
                drawed_objects[func_name + "--" + convert_color_to_hex(color)] = result
            elif func.__name__ == "__init__" and type(args[0]) == Ellipse:
                color = convert_color_to_hex(args[0].get_facecolor())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = args[0]
            elif func.__name__ == "tripcolor":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" + colormap] = result  # ! Attention
            elif func.__name__ == "__init__" and type(args[0]) == VennDiagram:
                for item in args[0].patches:
                    color = convert_color_to_hex(item.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = args[0]
            elif func.__name__ == "__init__" and type(args[0]) == Circle:
                color = convert_color_to_hex(args[0].get_facecolor())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = args[0]
            in_decorator = False
        else:
            return func(*args, **kwargs)
        return result

    return wrapper


def update_drawed_colors(drawed_obejcts):
    drawed_colors = []
    for name, obj in drawed_objects.items():
        func_name = name.split("--")[0]
        color = name.split("--")[1]

        if "/_makeline" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "--" + color)
        elif "/axhline" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "--" + color)
        elif "/axvline" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "--" + color)
        elif "/_fill_between_x_or_y" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolors()[0]))
            drawed_colors.append(func_name + "--" + color)
        elif "/bar" in func_name and "_3d" not in func_name:
            color = convert_color_to_hex(list(obj._original_facecolor))
            drawed_colors.append(func_name + "--" + color)
        elif "/scatter" in func_name and "polar" not in func_name and "3d" not in func_name:
            # check whether cmap is used by checking whether color is hex
            if color.startswith("#") == False:
                drawed_colors.append(func_name + "--" + color)
            else:
                if len(obj.get_facecolor()) != 0:
                    color = convert_color_to_hex(list(obj.get_facecolor()[0]))
                    drawed_colors.append(func_name + "--" + color)
        elif "/pie" in func_name:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/axvspan" in func_name:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/axhspan" in func_name:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/hlines" in func_name:
            for item in obj.get_edgecolors():
                color = convert_color_to_hex(list(item))
                drawed_colors.append(func_name + "--" + color)
        elif "/vlines" in func_name:
            for item in obj.get_edgecolors():
                color = convert_color_to_hex(list(item))
                drawed_colors.append(func_name + "--" + color)
        elif "/boxplot" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()))
            drawed_colors.append(func_name + "--" + color)
        elif "/violinplot" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()[0]))
            drawed_colors.append(func_name + "--" + color)
        elif "/hist" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()))
            drawed_colors.append(func_name + "--" + color)
        elif "/quiver" in func_name:
            for item in obj.get_facecolors().tolist():
                color = convert_color_to_hex(item)
                drawed_colors.append(func_name + "--" + color)
        elif "/plot" in func_name and "polar" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "_polar--" + color)
        elif "/scatter" in func_name and "polar" in func_name:
            # check whether cmap is used by checking whether color is hex
            if color.startswith("#") == False:
                drawed_colors.append(func_name + "_polar--" + color)
            else:
                if len(obj.get_facecolor()) != 0:
                    color = convert_color_to_hex(list(obj.get_facecolor()[0]))
                    drawed_colors.append(func_name + "_polar--" + color)
        elif "/plot" in func_name and "_squarify" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()))
            drawed_colors.append(func_name + "--" + color)
        elif "/imshow" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/pcolor" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/contour" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/contourf" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/fill" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()))
            drawed_colors.append(func_name + "--" + color)
        elif "/__init__" in func_name and type(obj) == NonUniformImage:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/broken_barh" in func_name:
            colors = obj.get_facecolors().tolist()
            for color in colors:
                drawed_colors.append(func_name + "--" + convert_color_to_hex(color))
        elif "/__init__" in func_name and type(obj) == Ellipse:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/tripcolor" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/__init__" in func_name and type(obj) == VennDiagram:
            for item in obj.patches:
                color = convert_color_to_hex(item.get_facecolor())
                drawed_colors.append(func_name + "--" + color)
        elif "/__init__" in func_name and type(obj) == Circle:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/scatter" in func_name and "3d" in func_name:
            # check whether cmap is used by checking whether color is hex
            if color.startswith("#") == False:
                drawed_colors.append(func_name + "_3d--" + color)
            else:
                for item in obj.get_facecolors().tolist():
                    color = convert_color_to_hex(item)
                    drawed_colors.append(func_name + "_3d--" + color)
        elif "/plot" in func_name and "3d" in func_name and "plot_surface" not in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "_3d--" + color)
        elif "/plot_surface" in func_name:
            if color.startswith("#") == False:
                drawed_colors.append(func_name + "_3d--" + color)
            else:
                colors = obj.get_facecolors().tolist()
                drawed_colors.append(func_name + "_3d--" + convert_color_to_hex(colors[0]))
        elif "/bar3d" in func_name:
            colors = obj.get_facecolors().tolist()
            drawed_colors.append(func_name + "_3d--" + convert_color_to_hex(colors[0]))
        elif "/bar" in func_name and "3d" in func_name:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "_3d--" + color)
        elif "/add_collection3d" in func_name:
            colors = obj.get_facecolors().tolist()
            for color in colors:
                drawed_colors.append(func_name + "_3d--" + convert_color_to_hex(color))
        elif "/draw_networkx_labels" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "--" + color)
        elif "/draw_networkx_edges" in func_name:
            if type(obj) == list:
                for line in obj:
                    color = convert_color_to_hex(line.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
            else:
                for item in obj.get_edgecolors().tolist():
                    color = convert_color_to_hex(item)
                    drawed_colors.append(func_name + "--" + color)
        elif "/draw_networkx_nodes" in func_name:
            for item in obj.get_facecolors().tolist():
                color = convert_color_to_hex(item)
                drawed_colors.append(func_name + "--" + color)

    drawed_colors = list(set(drawed_colors))

    return drawed_colors


_process_plot_var_args._makeline = log_function(_process_plot_var_args._makeline)
Axes.bar = log_function(Axes.bar)
Axes.scatter = log_function(Axes.scatter)
Axes.axhline = log_function(Axes.axhline)
Axes.axvline = log_function(Axes.axvline)
Axes._fill_between_x_or_y = log_function(Axes._fill_between_x_or_y)
Axes.pie = log_function(Axes.pie)
Axes.axvspan = log_function(Axes.axvspan)
Axes.axhspan = log_function(Axes.axhspan)
Axes.hlines = log_function(Axes.hlines)
Axes.vlines = log_function(Axes.vlines)
Axes.boxplot = log_function(Axes.boxplot)
Axes.violinplot = log_function(Axes.violinplot)
Axes.hist = log_function(Axes.hist)
# Axes.plot = log_function(Axes.plot)
PolarAxes.plot = log_function(PolarAxes.plot)
Axes.quiver = log_function(Axes.quiver)
Axes.imshow = log_function(Axes.imshow)
Axes.pcolor = log_function(Axes.pcolor)
Axes.contour = log_function(Axes.contour)
Axes.contourf = log_function(Axes.contourf)
Axes.fill = log_function(Axes.fill)
NonUniformImage.__init__ = log_function(NonUniformImage.__init__)
Ellipse.__init__ = log_function(Ellipse.__init__)
Axes.broken_barh = log_function(Axes.broken_barh)

nx_pylab.draw_networkx_nodes = log_function_specific_for_draw_networkx_nodes(nx_pylab.draw_networkx_nodes)
nx_pylab.draw_networkx_edges = log_function_specific_for_draw_networkx_edges(nx_pylab.draw_networkx_edges)
nx_pylab.draw_networkx_labels = log_function_specific_for_draw_networkx_labels(nx_pylab.draw_networkx_labels)
nx_pylab.draw_networkx_edge_labels = log_function_specific_for_draw_networkx_edge_labels(nx_pylab.draw_networkx_edge_labels)   # mingchen debugging 明辰：拦截 networkx 的 draw_networkx_edge_labels 方法，记录 graph edge labels 信息

nx.draw_networkx_nodes = log_function_specific_for_draw_networkx_nodes(nx.draw_networkx_nodes)
nx.draw_networkx_edges = log_function_specific_for_draw_networkx_edges(nx.draw_networkx_edges)
nx.draw_networkx_labels = log_function_specific_for_draw_networkx_labels(nx.draw_networkx_labels)
nx.draw_networkx_edge_labels = log_function_specific_for_draw_networkx_edge_labels(nx.draw_networkx_edge_labels)               # mingchen debugging 明辰：拦截 networkx 的 draw_networkx_edge_labels 方法，记录 graph edge labels 信息

squarify.plot = log_function(squarify.plot)

Axes3D.scatter = log_function_for_3d(Axes3D.scatter)
Axes3D.plot = log_function_for_3d(Axes3D.plot)
Axes3D.plot_surface = log_function_for_3d(Axes3D.plot_surface)
Axes3D.bar3d = log_function_for_3d(Axes3D.bar3d)
Axes3D.bar = log_function_for_3d(Axes3D.bar)
Axes3D.add_collection3d = log_function_for_3d(Axes3D.add_collection3d)

Axes.tripcolor = log_function(Axes.tripcolor)

VennDiagram.__init__ = log_function(VennDiagram.__init__)

Circle.__init__ = log_function(Circle.__init__)

# 下面补充 名桂&明辰 YAML IR
matplotlib.axes._axes.cbook.boxplot_stats = log_function(matplotlib.axes._axes.cbook.boxplot_stats)  # mingchen debugging 明辰：拦截 matplotlib.axes._axes 导入的 matplotlib.cbook.boxplot_stats 方法，记录原始的 boxplot 图计算的五个分位点
matplotlib.axes._axes.cbook.violin_stats = log_function(matplotlib.axes._axes.cbook.violin_stats)    # mingchen debugging 明辰：拦截 matplotlib.axes._axes 导入的 matplotlib.cbook.violin_stats 方法，记录原始的小提琴图的统计信息
Axes.annotate = log_function(Axes.annotate)                                                          # mingchen debugging 明辰：拦截 matplotlib 的 annotate 方法，记录 quiver 图箭头和标签的信息
Axes.errorbar = log_function(Axes.errorbar)
Axes.eventplot = log_function(Axes.eventplot)

matplotlib.contour.QuadContourSet.__init__ = log_function(matplotlib.contour.QuadContourSet.__init__)


#####################################################################################################################
#  结束拦截 matplotlib 记录前缀信息
#####################################################################################################################

"""












    def _get_suffix(self, output_file=None):
        suffix_code_str = """
#####################################################################################################################
#  开始后缀：记录拦截信息（必须清洗掉被执行代码中的 plt.savefig, plt.show, plt.clf(), plt.cla() 操作）
#####################################################################################################################
# 获取未被破坏的 matplotlib Figure 对象，强制刷新渲染，方便后期坐标和文字提取
final_fig_obj = plt.gcf()
final_fig_obj.canvas.draw()


# 工具方法：遍历 matplotlib.figure.Figure 对象的每个组成部分，提取有效文本列表和相对像素位置列表
def extract_texts_and_pos_from_matplot_fig_obj(component, renderer, total_width, total_height):
    texts = []
    positions = []
    if isinstance(component, matplotlib.text.Text) and component.get_visible():
        text_str = component.get_text().strip()
        if text_str:  # only extract non-empty text
            texts.append(text_str)
            bbox = component.get_window_extent(renderer=renderer)
            x_pix = (bbox.x0 + bbox.x1) / 2
            y_pix = (bbox.y0 + bbox.y1) / 2
            x_rel = float(x_pix / total_width)
            y_rel = float(y_pix / total_height)
            positions.append((x_rel, y_rel))

    for child in component.get_children():
        child_texts, child_positions = extract_texts_and_pos_from_matplot_fig_obj(child, renderer, total_pixel_width, total_pixel_height)
        texts.extend(child_texts)
        positions.extend(child_positions)
    return texts, positions


drawed_colors = list(set(drawed_colors))
drawed_colors = update_drawed_colors(drawed_objects)
if len(drawed_colors) > 10:
    drawed_colors = filter_color(drawed_colors)
tweaked_mingchen_intermediate_ir_dict["color"] = drawed_colors

# 对 graph/networkx 图特殊处理：将 set 转成 list，方便JSON序列化
for key, val_set in tweaked_mingchen_intermediate_ir_dict["graph"].items():
    tweaked_mingchen_intermediate_ir_dict["graph"][key] = list(val_set)

# 对 contour 图特殊处理：将 set 转成 list
tweaked_mingchen_intermediate_ir_dict["contour"] = list(tweaked_mingchen_intermediate_ir_dict["contour"])

# histogram 特殊处理：将 bins set 转成 list，将 tops 数组去重
histogram_result_tuple_set = set()
for bins, tops in tweaked_mingchen_intermediate_ir_dict["histogram"]:
    histogram_result_tuple_set.add((tuple(bins), tuple(tops)))
tweaked_mingchen_intermediate_ir_dict["histogram"] = tuple(histogram_result_tuple_set)

# 从 matplot figure 元素提取文字和相对坐标位置
total_pixel_width, total_pixel_height = final_fig_obj.canvas.get_width_height()
renderer = final_fig_obj.canvas.get_renderer()
final_text_list, final_pos_list = extract_texts_and_pos_from_matplot_fig_obj(final_fig_obj, renderer, total_pixel_width, total_pixel_height)

tweaked_mingchen_intermediate_ir_dict["text"]["str"] = final_text_list
tweaked_mingchen_intermediate_ir_dict["text"]["pos"] = final_pos_list

print("<chart_code_ir> " + json.dumps(tweaked_mingchen_intermediate_ir_dict, ensure_ascii=False) + " </chart_code_ir>")

# print("drawed_colors", drawed_colors)
# print("len(drawed_colors)", len(drawed_colors))
# print("Length of drawed_obejcts", len(drawed_objects))
# print("drawed_objects", drawed_objects)

#####################################################################################################################
#  结束后缀：记录拦截信息（必须清洗掉被执行代码中的 plt.savefig, plt.show, plt.clf(), plt.cla() 操作）
#####################################################################################################################

"""
        if output_file:
            suffix_code_str += f"""
with open('{output_file}', 'w') as f:
    f.write(str(drawed_colors))
"""
        return suffix_code_str


# 清洁代码
def clean_matplotlib_code(raw_python_code_str: str) -> str:
    bad_word_list = [
        '.savefig(',
        'PdfPages(',
        'plt.show(',
        'plt.clf(',
        'plt.cla(',
        'import os',
        "os.makedirs(",
        'import sys',
        'import shutil'
    ]
    clean_code_line_list = []
    # 保持缩进，注释掉 `.savefig(` 所在行，然后添加同样缩进的 pass，防止无意义文件生成
    for line in raw_python_code_str.split('\n'):
        if any(bad_word in line for bad_word in bad_word_list):
            leading_blank_chars = line[0 : line.find(line.strip())]
            clean_code_line_list.append(leading_blank_chars + '# ' + line.strip())
            clean_code_line_list.append(leading_blank_chars + 'pass')
        else:
            clean_code_line_list.append(line)
    return '\n'.join(clean_code_line_list)



def interprete_python_code_file(chart_py_filepath: str, subprocess_timeout_seconds: float = 10) -> str:
    """
    调用 chartmimic conda 环境运行指定路径代码文件，返回 stdout 结果
    """
    try:
        output = subprocess.check_output([python_interpreter_filepath, chart_py_filepath],
                text=True,  # 自动解码为字符串 (默认是 bytes)
                timeout=subprocess_timeout_seconds,
                stderr=subprocess.STDOUT,
        )
        return output
    except subprocess.TimeoutExpired as e:
        pass
        # logger.error(f"超时了！部分输出: {str(e)}")
    except subprocess.CalledProcessError as e:
        pass
        # logger.error(f"Compile Check Failed with details: {repr(e.stdout)}")
    except Exception as e:
        pass
        logger.error(f'unknown error: {str(e)}')

    # 抛出异常，统一返回 空白字符串
    return ''


def calc_color_sim_score(predict_code_str: str, gt_colors: list) -> float:
    evaluator = ColorEvaluator()
    try:
        with tempfile.NamedTemporaryFile(dir=temp_base_dir, mode='w+', encoding='utf-8', prefix='verl_grpo_chart2code_color_eval', suffix='.py', delete=True) as temp_file:
            temp_file.write(clean_matplotlib_code(predict_code_str))
            temp_file.flush()
            code_filepath = temp_file.name
            print("saved temp code str to:", code_filepath)
            predict_colors = evaluator.get_log_colors(code_filepath)
            print("extracted ChartMimic low level color list:", json.dumps(predict_colors, ensure_ascii=False, indent=4))
            return evaluator._calculate_metrics(predict_colors, gt_colors)
    except Exception as e:
        import logging
        logging.error(f"failed to parse code: {str(e)}", exc_info=True)
        return -1


def get_code_ir_dict_from_subprocess_stdout(predict_code_str: str, xml_tag_name: str = 'chart_code_ir', subprocess_timeout_seconds: float = 20) -> dict:
    """
    将生成的代码写入临时文件解释执行，返回 Chart IR dict
    """
    try:
        evaluator = ColorEvaluator()
        with tempfile.NamedTemporaryFile(dir=temp_base_dir, mode='w+', encoding='utf-8', prefix='verl_grpo_chart2code_code_ir_eval', suffix='.py', delete=True) as temp_file:
            temp_file.write(evaluator._get_prefix() + "\n\n")
            temp_file.write(clean_matplotlib_code(predict_code_str) + "\n")
            temp_file.write(evaluator._get_suffix())
            temp_file.flush()
            code_filepath = temp_file.name
            # print("saved temp code str to:", code_filepath)
            ret_stdout_str = interprete_python_code_file(code_filepath, subprocess_timeout_seconds)
            assert len(ret_stdout_str) > 0
            match = re.search(rf"<{xml_tag_name}>(.*?)</{xml_tag_name}>", ret_stdout_str, re.DOTALL)
            if match:
                text_inside = match.group(1).strip()
                return json.loads(text_inside)
    except Exception as e:
        # import logging
        # logging.error(f"failed to parse code with err {repr(e)}", exc_info=True)
        pass
    return {}


if __name__ == "__main__：验证调试":
    test_case_id = "chart2code-160k-40986"
    test_gt_colors = ["/home/MY_USERNAME/miniconda3/envs/chartmimic/lib/python3.9/site-packages/matplotlib/axes/_base.py/_makeline--#FF0000"]
    test_code_str = "import matplotlib.pyplot as plt\nimport numpy as np\n\n# Restructure the data\ndata = [\n    [70, 80, 85, 90, 100],  # Robotics\n    [60, 70, 75, 85, 95],   # Artificial Intelligence\n    [75, 80, 85, 90, 95],   # Machine Learning\n    [55, 60, 70, 75, 85]    # Control Theory\n]\noutliers = [[], [], [110], [50, 95]] # Example outliers\n\n# Plot the chart\nfig = plt.figure(figsize=(12, 6))\nax = fig.add_subplot(111)\nboxprops = dict(linewidth=1.5, color='#2F4F4F')\nmedianprops = dict(linewidth=2, color='#DC143C')\nwhiskerprops = dict(linewidth=1.5, color='#DC143C')\ncapprops = dict(linewidth=1.5, color='#DC143C')\n\nax.boxplot(data, \n           whis=1.5, \n           boxprops=boxprops, \n           medianprops=medianprops, \n           whiskerprops=whiskerprops, \n           capprops=capprops, \n           flierprops=dict(markerfacecolor='r', marker='o'))\n\n# Plot Outliers\nfor i, outlier in enumerate(outliers):\n    if len(outlier) > 0:\n        ax.plot(np.repeat(i+1, len(outlier)), outlier, 'ro', markersize=8, alpha=0.7)\n\n# Adjust the chart\nax.set_title('Adaptive Control Metrics', fontsize=16, fontweight='bold', fontfamily='sans-serif')\nax.set_xlabel('Category', fontsize=14, fontfamily='sans-serif')\nax.set_ylabel('Metrics Value', fontsize=14, fontfamily='sans-serif')\nax.set_xticklabels(['Robotics', 'AI', 'ML', 'Control Theory'], fontsize=12, fontfamily='sans-serif')\n\n# Add grids\nax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)\nax.set_axisbelow(True)\n\n# Resize the chart\nplt.tight_layout()\n\n# Save the chart"

    final_lowlevel_color_score = calc_color_sim_score(test_code_str, test_gt_colors)
    print("final ChartMimic Lowlevel color score =", final_lowlevel_color_score)
