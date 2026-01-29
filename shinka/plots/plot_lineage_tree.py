from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import matplotlib.cm as cm_module
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_lineage_tree(
    df: pd.DataFrame,
    title="Program Lineage Tree",
    fig: Figure | None = None,
    ax: Axes | None = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Generates a tree visualization of program lineage using matplotlib and
    NetworkX.

    Args:
        df: Pandas DataFrame containing program data. Must include 'id' and
            'parent_id'.
        figsize: Size of the figure (width, height) in inches.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None. Cannot draw tree.")
        return

                                                    
    df.loc[~df["correct"], "combined_score"] = 0

                           
    G = nx.DiGraph()

                                          
    for idx, row in df.iterrows():
        node_id = str(row["id"])
        node_attrs = {}

                                                  
        for col in df.columns:
            if col in row:
                                                               
                if col != "code":
                    node_attrs[col] = row[col]

        G.add_node(node_id, **node_attrs)

               
    for idx, row in df.iterrows():
        child_id = str(row["id"])
        if "parent_id" in row and pd.notna(row["parent_id"]):
            parent_id = str(row["parent_id"])
                                                                
            if parent_id in G.nodes() and parent_id != child_id:
                G.add_edge(parent_id, child_id)

                                                                   
                                                 
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 16))

                                                         
    generation_groups = {}
    for node in G.nodes():
        attrs = G.nodes[node]
        gen = attrs.get("generation", 0)                                       
        if gen not in generation_groups:
            generation_groups[gen] = []
        generation_groups[gen].append(node)

                                                    
    roots = [n for n, d in G.in_degree() if d == 0]

    if not roots:
                                                                 
        min_gen = min(generation_groups.keys()) if generation_groups else 0
        roots = generation_groups.get(min_gen, [list(G.nodes())[0]])

    root = roots[0]

                                                                               
                                                                       
    try:
                                                                           
                      
        pos = nx.nx_agraph.graphviz_layout(
            G,
            prog="dot",
            root=root,
            args="-Grankdir=TB -Gsplines=true -Goverlap=false -Gsep=1.0",
        )
    except ImportError:
        try:
                                                                        
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot", root=root)
        except ImportError:
            print("GraphViz not available, using hierarchical layout")
                                                         
            pos = {}

                                                          
            depths = {}
            for node in G.nodes():
                try:
                    path_len = len(nx.shortest_path(G, root, node)) - 1
                    depths[node] = path_len
                except nx.NetworkXNoPath:
                                                             
                    if "generation" in G.nodes[node]:
                        depths[node] = G.nodes[node]["generation"]
                    else:
                        depths[node] = 0

                                             
            levels = {}
            for node, depth in depths.items():
                if depth not in levels:
                    levels[depth] = []
                levels[depth].append(node)

                                                   
            max_depth = max(levels.keys()) if levels else 0
                                                   
            total_nodes = len(G.nodes())
            for depth in sorted(levels.keys()):
                nodes_at_level = levels[depth]
                num_nodes_at_level = len(nodes_at_level)

                if depth == 0:
                                             
                    if num_nodes_at_level == 1:
                        pos[nodes_at_level[0]] = (0, 0)
                    else:
                                                                      
                        spacing = 15.0
                        total_width = (num_nodes_at_level - 1) * spacing
                        start_x = -total_width / 2
                        for i, node in enumerate(nodes_at_level):
                            pos[node] = (start_x + i * spacing, 0)
                else:
                                                                        
                             
                                           

                                                                   
                    node_parent_info = {}
                    for node in nodes_at_level:
                        parent_x_positions = []
                        for parent in G.predecessors(node):
                            if parent in pos:
                                parent_x_positions.append(pos[parent][0])

                        if parent_x_positions:
                            avg_parent_x = sum(parent_x_positions) / len(
                                parent_x_positions
                            )
                            node_parent_info[node] = avg_parent_x
                        else:
                            node_parent_info[node] = 0

                                                                              
                    sorted_nodes = sorted(
                        nodes_at_level, key=lambda n: node_parent_info[n]
                    )

                                                                            
                                      
                                                                           
                    base_spacing = max(15.0, 10.0 * (total_nodes**0.5))
                                                                        
                             
                    depth_multiplier = max(1.0, 3.0 / (depth + 1))

                    spacing = base_spacing * depth_multiplier

                    y_pos = -depth * 3.0                              

                    if num_nodes_at_level == 1:
                        pos[sorted_nodes[0]] = (0, y_pos)
                    else:
                        total_width = (num_nodes_at_level - 1) * spacing
                        start_x = -total_width / 2

                        for i, node in enumerate(sorted_nodes):
                            x_pos = start_x + i * spacing
                            pos[node] = (x_pos, y_pos)

                                                                                   
                            if node in node_parent_info:
                                preferred_x = node_parent_info[node]
                                                                               
                                                                                         
                                min_distance = spacing * 0.8
                                can_move = True

                                for other_node in sorted_nodes:
                                    if other_node != node and other_node in pos:
                                        other_x = pos[other_node][0]
                                        if abs(preferred_x - other_x) < min_distance:
                                            can_move = False
                                            break

                                if can_move:
                                                                                 
                                    adjustment = (preferred_x - x_pos) * 0.1
                                    pos[node] = (x_pos + adjustment, y_pos)

                                                
    if pos and len(pos) > 1:
                                                                          
        for depth in sorted(levels.keys()):
            if depth == 0:             
                continue

            nodes_at_level = levels[depth]
            if len(nodes_at_level) <= 1:
                continue

                                                              
            def count_crossings(node_positions):
                crossings = 0
                for i, node1 in enumerate(nodes_at_level):
                    for j, node2 in enumerate(nodes_at_level):
                        if i >= j:
                            continue

                                                   
                        parents1 = list(G.predecessors(node1))
                        parents2 = list(G.predecessors(node2))

                        for p1 in parents1:
                            for p2 in parents2:
                                if p1 in pos and p2 in pos:
                                                          
                                    p1_x, p1_y = pos[p1]
                                    p2_x, p2_y = pos[p2]
                                    n1_x = node_positions[node1][0]
                                    n2_x = node_positions[node2][0]

                                                           
                                    if (p1_x < p2_x and n1_x > n2_x) or (
                                        p1_x > p2_x and n1_x < n2_x
                                    ):
                                        crossings += 1
                return crossings

                                                       
            improved = True
            max_iterations = 10
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                for i in range(len(nodes_at_level) - 1):
                    node1 = nodes_at_level[i]
                    node2 = nodes_at_level[i + 1]

                                                                   
                    temp_positions = dict(pos)
                    temp_positions[node1] = pos[node2]
                    temp_positions[node2] = pos[node1]

                                                     
                    original_crossings = count_crossings(pos)
                    new_crossings = count_crossings(temp_positions)

                    if new_crossings < original_crossings:
                                        
                        pos[node1] = temp_positions[node1]
                        pos[node2] = temp_positions[node2]
                                                     
                        nodes_at_level[i], nodes_at_level[i + 1] = (
                            nodes_at_level[i + 1],
                            nodes_at_level[i],
                        )
                        improved = True

                                                        
    num_nodes = len(G.nodes())
                                                    
    size_factor = max(0.3, min(1.0, 20 / (num_nodes**0.4)))

    best_node_size = int(1500 * size_factor)
    path_node_size = int(800 * size_factor)
    regular_node_size = int(600 * size_factor)

                                                         
    score_values = []
    score_field = "combined_score"                         

                                        
    best_node = None
    best_score = float("-inf")

    for node in G.nodes():
        if score_field in G.nodes[node]:
            score = G.nodes[node][score_field]
            if isinstance(score, (int, float)):
                score_values.append(score)
                if score > best_score:
                    best_score = score
                    best_node = node

                                                             
    path_to_best = []
    best_path_edges = []
    if best_node:
        try:
                                                       
            path_to_best = nx.shortest_path(G, root, best_node)
                                                    
            best_path_edges = list(zip(path_to_best[:-1], path_to_best[1:]))
        except nx.NetworkXNoPath:
                                                  
            pass

                                               
    regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in best_path_edges]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=regular_edges,
        arrows=False,
        arrowsize=12,
        width=1.5,
        edge_color="black",
        alpha=0.6,
        ax=ax,
    )

                                                                                 
    if best_path_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=best_path_edges,
            arrows=False,
            arrowsize=20,
            width=3.5,
            edge_color="black",
            alpha=0.9,
            ax=ax,
        )

    if score_values:
        if vmin is None:
            vmin = min(score_values)
        if vmax is None:
            vmax = max(score_values)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                                                                                         
        color_map = cm_module.get_cmap("viridis")

                                                        
        for node in G.nodes():
            node_attrs = G.nodes[node]
            current_node_size = regular_node_size
                                                   
            current_node_color = "lightgray"
            current_edge_color = "black"
            current_linewidth = 1.5
            current_node_shape = "o"                                    

                                                                            
            is_correct = node_attrs.get("correct", True)                   
            if not is_correct:
                current_node_shape = "x"                               
                current_node_color = "red"
                current_edge_color = "darkred"
                current_linewidth = 4.0                      
                current_node_size = int(current_node_size * 1.5)               
            else:
                                                                              
                patch_type = node_attrs.get("patch_type")
                if patch_type == "full":
                    current_node_shape = "s"          
                elif patch_type == "init":
                    current_node_shape = "^"               
                                             
                                                         
                elif patch_type == "cross":
                    current_node_shape = "P"                 

                if score_field in node_attrs:
                    score = node_attrs[score_field]
                    if pd.isna(score):                 
                        current_node_color = "purple"                        
                    elif isinstance(score, (int, float)):
                        color = color_map(norm(score))
                        current_node_color = mcolors.to_hex(color)

                                                                             
            if node == best_node and is_correct:                   
                current_node_size = best_node_size
                current_node_color = "gold"
                current_edge_color = "black"
                current_linewidth = 2.5
                current_node_shape = "*"                            
            elif node in path_to_best and is_correct:                   
                current_node_size = path_node_size
                                       
                                        
                                 
                                       
                node_score = node_attrs.get(score_field)
                if node_score is not None and not pd.isna(node_score):
                                          
                    if isinstance(node_score, (int, float)):
                        color = color_map(norm(node_score))
                        current_node_color = mcolors.to_hex(color)
                current_edge_color = "black"
                current_linewidth = 2.0
                                                                      

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_size=current_node_size,
                node_color=current_node_color,
                edgecolors=current_edge_color,
                linewidths=current_linewidth,
                ax=ax,
                node_shape=current_node_shape,
            )

                                                 
        sm = cm_module.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(
            sm,
            ax=ax,                          
            pad=-0.05,
            shrink=0.6,
        )
        cb.set_label(label="Combined Fitness Score", size=20, weight="bold")
        cb.ax.tick_params(labelsize=16)
    else:
                                                                  
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=regular_node_size,
            node_color="lightblue",
            edgecolors="black",
            linewidths=1.5,
            ax=ax,
        )

                                                                   
    node_labels = {}
    for node in G.nodes():
        attrs = G.nodes[node]
        label_parts = []

                                     
        if "generation" in attrs:
            label_parts.append(f"{attrs['generation']}")

                                           
                                  
                                        
                                          
                                                    
                   
                                                

                             
        if label_parts:
                                                        
            node_labels[node] = label_parts[0]
               
                                                                             
                                               

                                                                             
                                 
    label_pos = {}
    for node, (x, y) in pos.items():
                                              
        label_pos[node] = (x, y + 0.0)

                                                                    
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=node_labels,
        font_size=12,
        font_weight="bold",
        font_color="white",
        ax=ax,
    )

                                             
    if best_node:
        star_patch = Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markersize=20,
            label="Best Score",
        )
                                                   
        path_line = Line2D([0], [0], color="red", linewidth=4, label=r"Path$\to$Best")
                                
        diff_patch = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Diff Edit",
            markerfacecolor="gray",
            markersize=10,
        )
        full_patch = Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Full Edit",
            markerfacecolor="gray",
            markersize=10,
        )
        init_patch = Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Initial",
            markerfacecolor="gray",
            markersize=10,
        )
                               
                  
                  
                                    
                        
                                 
                                     
                            
           
        crossover_patch = Line2D(
            [0],
            [0],
            marker="P",                 
            color="w",
            label="Cross-Over",
            markerfacecolor="gray",
            markersize=10,
        )
        incorrect_patch = Line2D(
            [0],
            [0],
            marker="x",           
            color="w",
            label="Incorrect",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=15,
            markeredgewidth=3,
        )

        legend_handles = [
            star_patch,
                        
            diff_patch,
            full_patch,
            init_patch,
                          
            crossover_patch,
            incorrect_patch,
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=25, ncol=2)

    ax.set_title(title, fontsize=40, fontweight="bold")
    ax.axis("off")
                                                                            
    fig.subplots_adjust(left=0.02, right=0.85, top=0.95, bottom=0.05)
    return fig, ax
