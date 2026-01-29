import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from shinka.utils import get_path_to_best_node
import matplotlib.transforms as transforms


def plot_improvement(
    df: pd.DataFrame,
    title: str = "Best Combined Score Over Time",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "Number of Evaluated LLM Program Proposals",
    ylabel: str = "Evolved Performance Score",
    ylim: Optional[Tuple[float, float]] = None,
    plot_path_to_best_node: bool = True,
):
    """
    Plots the improvement of a program over generations.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

                          
                                                             
                                                      
    df = df.sort_values(by="generation")
    df_filtered = df[df["correct"]].copy()

    line1 = ax.plot(
        df_filtered["generation"],
        df_filtered["combined_score"].cummax(),
        linewidth=3,
        color="red",
        label="Best Score",
    )

                                                   
    scatter1 = ax.scatter(
        df_filtered["generation"],
        df_filtered["combined_score"],
        alpha=1.0,
        s=40,
        color="black",
        label="Individual Evals",
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

                                   
    if plot_path_to_best_node:
        best_path_df = get_path_to_best_node(df_filtered, score_column="combined_score")
    else:
        best_path_df = pd.DataFrame()
    line_best_path_plot = []                            

    if not best_path_df.empty:
                                        
        line_best_path_plot = ax.plot(
            best_path_df["generation"],                             
            best_path_df["combined_score"],
            linestyle="-.",
            marker="o",
            color="blue",
            label="Path to Best Node",
            markersize=5,
            linewidth=2,
        )
                                                       
        if "patch_name" in best_path_df.columns:
            _place_non_overlapping_annotations(
                ax, best_path_df, "generation", "combined_score", "patch_name"
            )

                                                    
    ax2 = ax.twinx()
    handles = line1 + [scatter1]
    if line_best_path_plot:                                
        handles.extend(line_best_path_plot)

    labels = [h.get_label() for h in handles]

    if "api_costs" in df_filtered.columns:
        cumulative_api_cost = df["api_costs"].cumsum().bfill()
        line2 = ax2.plot(
            df["generation"],
            cumulative_api_cost,
            linewidth=2,
            color="orange",
            linestyle="--",
            label="Cumulative Cost",
        )
        ax2.set_ylabel(
            "Cumulative API Cost ($)",
            fontsize=22,
            weight="bold",
            color="orange",
            labelpad=15,
        )
        ax2.tick_params(axis="y", which="major", labelsize=25)
        handles.extend(line2)
        labels = [h.get_label() for h in handles]                   

    ax.legend(handles, labels, fontsize=25, loc="lower right")

                    
    ax.set_xlabel(xlabel, fontsize=30, weight="bold")
    ax.set_ylabel(ylabel, fontsize=30, weight="bold", labelpad=25)
    ax.set_title(title, fontsize=40, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, alpha=0.3)

                                                      
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(
        False
    )                                                            

    if "api_cost" in df_filtered.columns and ax2:
                                                  
        ax2.spines["top"].set_visible(False)                                
        ax2.tick_params(axis="y", which="major", labelsize=30)

    fig.tight_layout()                                               

    return fig, ax


def _place_non_overlapping_annotations(
    ax: Axes, df: pd.DataFrame, x_col: str, y_col: str, text_col: str
):
    """
    Places annotations with minimal overlap using a systematic approach.
    """
                                                                      
    offset_positions = [
        (40, -30),                
        (40, 30),             
        (-40, 30),            
        (-40, -30),               
        (60, 0),         
        (-60, 0),        
        (0, 40),       
        (0, -40),          
        (70, -50),                    
        (-70, 50),                
    ]

    placed_boxes = []                                              

    for _, row in df.iterrows():
        patch_name_val = str(row.get(text_col, ""))
        if pd.notna(patch_name_val) and patch_name_val != "":
            if patch_name_val == "nan" or patch_name_val == "none":
                patch_name_val = "Base"

                                   
            patch_name_to_plot = _wrap_text(patch_name_val, max_length=15)

            x_pos = float(row[x_col])
            y_pos = float(row[y_col])

                                                         
            best_offset, best_ha, best_va = _find_best_position(
                ax, x_pos, y_pos, patch_name_to_plot, offset_positions, placed_boxes
            )

                                  
            annotation = ax.annotate(
                patch_name_to_plot,
                (x_pos, y_pos),
                textcoords="offset points",
                xytext=best_offset,
                ha=best_ha,
                va=best_va,
                fontsize=11,
                fontweight="bold",
                color="darkgreen",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="lightyellow",
                    ec="black",
                    alpha=0.7,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    shrinkA=5,
                    shrinkB=5,
                    connectionstyle="arc3,rad=0.2",
                    color="black",
                ),
                zorder=10,
            )

                                                                   
            try:
                                                          
                bbox = annotation.get_window_extent()
                inv_transform = ax.transData.inverted()
                bbox_data = inv_transform.transform_bbox(bbox)
                placed_boxes.append(bbox_data)
            except Exception:
                                                    
                approx_width = len(patch_name_to_plot) * 0.01                  
                approx_height = patch_name_to_plot.count("\n") * 0.02 + 0.02
                placed_boxes.append(
                    transforms.Bbox.from_bounds(
                        x_pos - approx_width / 2,
                        y_pos - approx_height / 2,
                        approx_width,
                        approx_height,
                    )
                )


def _wrap_text(text: str, max_length: int = 15) -> str:
    """
    Wraps text at word boundaries for better readability.
    """
    if len(text) <= max_length:
        return text

                                       
    mid_point = len(text) // 2

                                      
    for offset in range(min(5, mid_point)):
                               
        if mid_point - offset > 0 and text[mid_point - offset] == " ":
            break_point = mid_point - offset
            part1 = text[:break_point].strip()
            part2 = text[break_point + 1 :].strip()
            return f"{part1}\n{part2}"

                              
        if mid_point + offset < len(text) and text[mid_point + offset] == " ":
            break_point = mid_point + offset
            part1 = text[:break_point].strip()
            part2 = text[break_point + 1 :].strip()
            return f"{part1}\n{part2}"

                                            
    return f"{text[:mid_point]}\n{text[mid_point:]}"


def _find_best_position(
    ax: Axes,
    x_pos: float,
    y_pos: float,
    text: str,
    offset_positions: List[Tuple[int, int]],
    placed_boxes: List[transforms.Bbox],
) -> Tuple[Tuple[int, int], str, str]:
    """
    Finds the best annotation position with minimal overlap.
    """
    best_offset = offset_positions[0]
    best_overlap_count = float("inf")

    for offset in offset_positions:
                                             
        ha = "left" if offset[0] >= 0 else "right"
        va = "bottom" if offset[1] >= 0 else "top"

                                                     
        estimated_bbox = _estimate_annotation_bbox(
            ax, x_pos, y_pos, text, offset, ha, va
        )

                                                  
        overlap_count = sum(1 for bbox in placed_boxes if estimated_bbox.overlaps(bbox))

                                           
        if overlap_count == 0:
            return offset, ha, va

                                                  
        if overlap_count < best_overlap_count:
            best_overlap_count = overlap_count
            best_offset = offset

                                              
    ha = "left" if best_offset[0] >= 0 else "right"
    va = "bottom" if best_offset[1] >= 0 else "top"

    return best_offset, ha, va


def _estimate_annotation_bbox(
    ax: Axes,
    x_pos: float,
    y_pos: float,
    text: str,
    offset: Tuple[int, int],
    ha: str,
    va: str,
) -> transforms.Bbox:
    """
    Estimates the bounding box of an annotation in data coordinates.
    """
                                                               
    lines = text.split("\n")
    max_line_length = max(len(line) for line in lines)
    num_lines = len(lines)

                                                        
    char_width_data = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 100
    line_height_data = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 50

    width = max_line_length * char_width_data
    height = num_lines * line_height_data

                                                                  
    x_offset_data = offset[0] * char_width_data / 8                    
    y_offset_data = offset[1] * line_height_data / 12                    

                                                      
    if ha == "left":
        left = x_pos + x_offset_data
        right = left + width
    else:                 
        right = x_pos + x_offset_data
        left = right - width

    if va == "bottom":
        bottom = y_pos + y_offset_data
        top = bottom + height
    else:               
        top = y_pos + y_offset_data
        bottom = top - height

    return transforms.Bbox.from_bounds(left, bottom, width, height)
