import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

from .plot_improvement import _wrap_text
from adjustText import adjust_text


                                                   
                                                  
def get_pareto_mask(points: np.ndarray) -> np.ndarray:
    """
    Finds the Pareto-efficient points. Assumes all objectives are
    to be maximized. A point is Pareto efficient if it is not
    dominated by any other point. Point A dominates point B if A is
    at least as good as B on all objectives, and strictly better
    than B on at least one objective.

    :param points: An (n_points, n_objectives) NumPy array.
    :return: A (n_points, ) boolean NumPy array indicating Pareto optimality.
    """
    num_points = points.shape[0]
    is_pareto = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        if not is_pareto[i]:                                        
            continue
        for j in range(num_points):
            if i == j:
                continue

                                                
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_pareto[i] = False                                   
                break
    return is_pareto


def _place_pareto_annotations_with_connections(
    ax, pareto_df, x_col, y_col, x_maximize=True
):
    """
    Place patch name annotations for Pareto points using adjustText for
    optimal positioning. Deduplicates based on coordinates and patch_name
    to avoid multiple annotations for the same point.
    """
                                                         
    ax.figure.canvas.draw_idle()

    annotations = []

                                                                       
                                                                  
    unique_points = {}
    for _, row in pareto_df.iterrows():
        patch_name_val = str(row.get("patch_name", ""))
        if (
            pd.notna(patch_name_val)
            and patch_name_val != ""
            and patch_name_val not in ["nan", "none"]
        ):
            x_pos = float(row[x_col])
            y_pos = float(row[y_col])

                                                              
            key = (x_pos, y_pos, patch_name_val)

                                                                 
            if key not in unique_points:
                unique_points[key] = row

                                                   
    for (x_pos, y_pos, patch_name_val), row in unique_points.items():
                               
        patch_name_to_plot = _wrap_text(patch_name_val, max_length=12)

                                                               
                                                                      
        x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
        y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])

                                                 
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        if x_maximize:
                                                                          
            x_offset = -x_range * 0.15                            
        else:
                                                                           
            x_offset = x_range * 0.15                            

                                          
        text_x = x_pos + x_offset
        text_y = y_pos

                                                              
        margin_x = x_range * 0.05             
        margin_y = y_range * 0.05             

        text_x = max(x_min + margin_x, min(x_max - margin_x, text_x))
        text_y = max(y_min + margin_y, min(y_max - margin_y, text_y))

                                                     
        annotation = ax.annotate(
            patch_name_to_plot,
            xy=(x_pos, y_pos),                             
            xytext=(text_x, text_y),                             
            fontsize=20,
            fontweight="bold",
            color="darkgreen",
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="lightyellow",
                ec="black",
                alpha=0.7,
            ),
            zorder=4.0,
            arrowprops=dict(
                arrowstyle="->",
                shrinkA=5,
                shrinkB=5,
                                                 
                color="black",
                linewidth=3,                    
            ),
        )
        annotations.append(annotation)

                                                                
    if annotations:
                                                                             
        for annotation in annotations:
            annotation.set_clip_on(True)

                                                                                       
        annotations_with_points = []
        for annotation in annotations:
                                                                  
            xy_pos = annotation.xy
            annotations_with_points.append((xy_pos[0], annotation))

                              
        annotations_with_points.sort(key=lambda x: x[0])

                                                                              
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min

                                                   
        n_annotations = len(annotations_with_points)

                                               
        label_zone_height = y_range * 0.6                                     
        label_zone_bottom = y_min + y_range * 0.15                         

        if n_annotations > 1:
            y_spacing = label_zone_height / (n_annotations - 1)
        else:
            y_spacing = 0

                                  
        for i, (data_x, annotation) in enumerate(annotations_with_points):
                                                  
            data_point_x, data_point_y = annotation.xy

                                                              
            if n_annotations == 1:
                label_y = label_zone_bottom + label_zone_height / 2
            else:
                label_y = label_zone_bottom + i * y_spacing

                                                                                
            if x_maximize:
                                                            
                label_x = data_point_x - x_range * 0.03
            else:
                                                             
                label_x = data_point_x + x_range * 0.03

                                              
            margin_x = x_range * 0.05
            margin_y = y_range * 0.05

            label_x = data_point_x - min(x_range * 0.03, margin_x)
                                                                     
            label_y = max(y_min + margin_y, min(y_max - margin_y, label_y))

                                  
            annotation.set_position((label_x, label_y))


def plot_pareto(
    df: pd.DataFrame,
    x_variable: str,
    y_variable: str,
    x_maximize: bool = True,
    y_maximize: bool = True,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    title: str = "Pareto Front Analysis",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    """
    Plots a 2D Pareto front with lineage connections, aiming for
    clarity and aesthetics. Axes are inverted as needed so that better
    is always higher and to the right.
    """
    x_metric_col_name, y_metric_col_name = x_variable, y_variable

                           
    final_xlabel = xlabel if xlabel is not None else x_metric_col_name
    final_ylabel = ylabel if ylabel is not None else y_metric_col_name

    required_plotting_cols = [x_metric_col_name, y_metric_col_name]
    missing_metrics = [col for col in required_plotting_cols if col not in df.columns]
    if missing_metrics:
        raise ValueError(
            f"DataFrame missing required metric columns: {missing_metrics}"
        )

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    df_plot = df.copy()

    if "correct" in df_plot.columns:
        try:
            df_plot["correct"] = df_plot["correct"].astype(bool)
        except Exception as e:
            print(
                f"Warning: Could not convert 'correct' column to boolean: "
                f"{e}. Using as is."
            )

        original_row_count = len(df_plot)
        df_plot = df_plot[df_plot["correct"]]
        if len(df_plot) < original_row_count:
            print(
                f"Filtered to {len(df_plot)} 'correct' rows from "
                f"{original_row_count} total."
            )
        if df_plot.empty:
            print("No 'correct' points found to plot.")
            ax.set_title(title, fontsize=32, fontweight="bold", pad=15)
            ax.set_xlabel(final_xlabel, fontsize=25, fontweight="bold", labelpad=15)
            ax.set_ylabel(final_ylabel, fontsize=25, fontweight="bold", labelpad=15)
            ax.grid(
                True, linestyle=":", alpha=0.9, color="lightgray"
            )                                  
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if fig:
                fig.tight_layout()
            return fig, ax
    else:
        print("Warning: 'correct' column not found. Plotting all points.")

    for col in [x_variable, y_variable]:
        df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")
    df_plot = df_plot.dropna(subset=[x_variable, y_variable])
    ax.tick_params(axis="both", which="major", labelsize=20)
    if df_plot.empty:
        print("No data to plot after processing metric columns.")
        ax.set_title(title, fontsize=32, fontweight="bold", pad=15)
        ax.set_xlabel(final_xlabel, fontsize=25, fontweight="bold", labelpad=15)
        ax.set_ylabel(final_ylabel, fontsize=25, fontweight="bold", labelpad=15)
        ax.grid(
            True, linestyle=":", alpha=0.9, color="lightgray"
        )                                  
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if fig:
            fig.tight_layout()
        return fig, ax

                                                  
                                                                          
    metric_values = df_plot[[x_variable, y_variable]].values.copy()
    if not x_maximize:
        metric_values[:, 0] = -metric_values[:, 0]
    if not y_maximize:
        metric_values[:, 1] = -metric_values[:, 1]

    pareto_mask = get_pareto_mask(metric_values)
    df_plot["is_pareto"] = pareto_mask

    pareto_df = df_plot[df_plot["is_pareto"]].copy()
    non_pareto_df = df_plot[~df_plot["is_pareto"]].copy()

                            
    if not non_pareto_df.empty:
        ax.scatter(
            non_pareto_df[x_metric_col_name],
            non_pareto_df[y_metric_col_name],
            color="dimgray",
            s=100,
            alpha=1.0,
            zorder=1,
            label="Dominated/Other",
        )

                               
    if not pareto_df.empty:
        ax.scatter(
            pareto_df[x_metric_col_name],
            pareto_df[y_metric_col_name],
            color="orangered",
            s=200,
            alpha=1.0,
            marker="o",
            edgecolor="black",
            linewidth=1,
            zorder=3,
            label="Pareto Optimal",
        )
                                          
    if not pareto_df.empty and len(pareto_df) > 1:
                                                                    
        pareto_sorted = pareto_df.sort_values(x_metric_col_name)

                                                              
        x_coords = pareto_sorted[x_metric_col_name].values
        y_coords = pareto_sorted[y_metric_col_name].values

        ax.plot(
            x_coords,
            y_coords,
            color="red",
            linewidth=4,
            alpha=0.7,
            zorder=2,
        )

                                                                         
                      
    if not x_maximize:
        ax.invert_xaxis()
    if not y_maximize:
        ax.invert_yaxis()

                                                                
    if not pareto_df.empty and "patch_name" in pareto_df.columns:
        _place_pareto_annotations_with_connections(
            ax, pareto_df, x_metric_col_name, y_metric_col_name, x_maximize
        )

    ax.set_xlabel(final_xlabel, fontsize=25, fontweight="bold", labelpad=15)
    ax.set_ylabel(final_ylabel, fontsize=25, fontweight="bold", labelpad=15)
    ax.set_title(title, fontsize=32, fontweight="bold", pad=15)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=25)

    ax.grid(
        True, linestyle=":", alpha=0.9, color="lightgray"
    )                                  
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if fig:
        fig.tight_layout()
    return fig, ax
