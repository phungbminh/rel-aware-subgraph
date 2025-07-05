#!/usr/bin/env python3
"""
Visualization tool for extracted RASG subgraphs
Hiển thị subgraphs với target edge màu đỏ, relation labels, và node IDs
"""

import argparse
import os
import lmdb
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SubgraphVisualizer:
    def __init__(self, data_root: str, mapping_dir: str = "mappings"):
        """
        Initialize visualizer with data paths
        """
        self.data_root = Path(data_root)
        self.mapping_dir = self.data_root / mapping_dir
        
        # Load mappings
        self.entity2id = self._load_pickle("entity2id.pkl")
        self.relation2id = self._load_pickle("relation2id.pkl") 
        self.id2entity = self._load_pickle("id2entity.pkl")
        self.id2relation = self._load_pickle("id2relation.pkl")
        
        # Load global graph
        self.global_graph = self._load_pickle("global_graph.pkl")
        
        print(f"✅ Loaded mappings: {len(self.entity2id)} entities, {len(self.relation2id)} relations")
    
    def _load_pickle(self, filename: str):
        """Load pickle file from mapping directory"""
        filepath = self.mapping_dir / filename
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def load_subgraphs(self, lmdb_path: str, max_samples: int = 10) -> List[Dict]:
        """
        Load subgraphs from LMDB database
        """
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")
        
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        subgraphs = []
        
        with env.begin() as txn:
            cursor = txn.cursor()
            count = 0
            
            for key, value in cursor:
                if key == b'_progress':
                    continue
                    
                try:
                    data = pickle.loads(value)
                    if 'error' not in data:
                        subgraphs.append(data)
                        count += 1
                        
                    if count >= max_samples:
                        break
                        
                except Exception as e:
                    print(f"Warning: Failed to load key {key}: {e}")
                    continue
        
        env.close()
        print(f"✅ Loaded {len(subgraphs)} subgraphs from {lmdb_path}")
        return subgraphs
    
    def create_networkx_graph(self, subgraph_data: Dict) -> Tuple[nx.Graph, Tuple[int, int, int]]:
        """
        Convert subgraph data to NetworkX graph
        Returns: (graph, target_triple)
        """
        target_triple = subgraph_data['triple']  # (head, relation, tail)
        nodes = subgraph_data['nodes']
        s_dist = subgraph_data.get('s_dist', [0] * len(nodes))
        t_dist = subgraph_data.get('t_dist', [0] * len(nodes))
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for i, node_id in enumerate(nodes):
            G.add_node(node_id, 
                      s_dist=s_dist[i], 
                      t_dist=t_dist[i],
                      is_head=(node_id == target_triple[0]),
                      is_tail=(node_id == target_triple[2]))
        
        # Add edges from global graph
        head_id, rel_id, tail_id = target_triple
        node_set = set(nodes)
        
        # Extract edges between nodes in subgraph
        if hasattr(self.global_graph, 'edge_list'):
            # If global graph has edge list
            edges = self.global_graph.edge_list
        else:
            # Reconstruct from CSR format
            edges = []
            for i in range(len(nodes)):
                node_i = nodes[i]
                if node_i < len(self.global_graph.indptr) - 1:
                    start = self.global_graph.indptr[node_i]
                    end = self.global_graph.indptr[node_i + 1]
                    for j in range(start, end):
                        neighbor = self.global_graph.indices[j]
                        if neighbor in node_set:
                            edges.append((node_i, neighbor))
        
        # Add edges to graph (simplified - no relation info for now)
        for edge in edges:
            if len(edge) >= 2 and edge[0] in node_set and edge[1] in node_set:
                G.add_edge(edge[0], edge[1])
        
        # Ensure target edge exists
        G.add_edge(head_id, tail_id, relation=rel_id, is_target=True)
        
        return G, target_triple
    
    def visualize_subgraph(self, subgraph_data: Dict, save_path: Optional[str] = None, 
                          figsize: Tuple[int, int] = (12, 8), show_distances: bool = True):
        """
        Visualize a single subgraph with highlighted target edge
        """
        G, target_triple = self.create_networkx_graph(subgraph_data)
        head_id, rel_id, tail_id = target_triple
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Layout
        try:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)
        
        # Draw edges
        edge_colors = []
        edge_widths = []
        
        for edge in G.edges():
            if (edge[0] == head_id and edge[1] == tail_id) or (edge[0] == tail_id and edge[1] == head_id):
                edge_colors.append('red')
                edge_widths.append(3.0)
            else:
                edge_colors.append('black')
                edge_widths.append(1.0)
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7, ax=ax)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            if node == head_id:
                node_colors.append('lightcoral')
                node_sizes.append(800)
            elif node == tail_id:
                node_colors.append('lightblue') 
                node_sizes.append(800)
            else:
                node_colors.append('lightgray')
                node_sizes.append(400)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              alpha=0.8, ax=ax)
        
        # Draw node labels (IDs)
        node_labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold', ax=ax)
        
        # Draw edge labels for target edge
        edge_labels = {}
        if G.has_edge(head_id, tail_id):
            rel_name = self.id2relation.get(rel_id, f"R{rel_id}")
            edge_labels[(head_id, tail_id)] = f"{rel_name}"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, 
                                   font_color='red', font_weight='bold', ax=ax)
        
        # Add distance info if requested
        if show_distances:
            distance_text = []
            for node in G.nodes():
                node_data = G.nodes[node]
                s_dist = node_data.get('s_dist', 0)
                t_dist = node_data.get('t_dist', 0)
                distance_text.append(f"Node {node}: d(h)={s_dist}, d(t)={t_dist}")
            
            ax.text(0.02, 0.98, "; ".join(distance_text[:5]), transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Title and formatting
        head_name = self.id2entity.get(head_id, f"E{head_id}")
        tail_name = self.id2entity.get(tail_id, f"E{tail_id}")
        rel_name = self.id2relation.get(rel_id, f"R{rel_id}")
        
        ax.set_title(f"Subgraph for Triple: ({head_name}, {rel_name}, {tail_name}) - "
                    f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}", 
                    fontsize=12, fontweight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved visualization to {save_path}")
        
        return fig, ax
    
    def visualize_multiple_subgraphs(self, lmdb_path: str, num_samples: int = 6, 
                                   output_dir: str = "visualizations", figsize_per_plot: Tuple[int, int] = (8, 6)):
        """
        Visualize multiple subgraphs in a grid layout
        """
        subgraphs = self.load_subgraphs(lmdb_path, num_samples)
        
        if len(subgraphs) == 0:
            print("❌ No subgraphs found to visualize")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate grid layout
        cols = min(3, len(subgraphs))
        rows = (len(subgraphs) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Visualize each subgraph
        for i, subgraph_data in enumerate(subgraphs):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                G, target_triple = self.create_networkx_graph(subgraph_data)
                head_id, rel_id, tail_id = target_triple
                
                # Simple layout for grid view
                pos = nx.spring_layout(G, k=2, iterations=30, seed=42)
                
                # Draw edges
                edge_colors = ['red' if ((e[0] == head_id and e[1] == tail_id) or 
                                       (e[0] == tail_id and e[1] == head_id)) else 'black' 
                             for e in G.edges()]
                edge_widths = [2.0 if c == 'red' else 0.8 for c in edge_colors]
                
                nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6, ax=ax)
                
                # Draw nodes
                node_colors = ['lightcoral' if n == head_id else 'lightblue' if n == tail_id else 'lightgray' 
                             for n in G.nodes()]
                node_sizes = [300 if n in [head_id, tail_id] else 150 for n in G.nodes()]
                
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                                     alpha=0.8, ax=ax)
                
                # Simple labels
                labels = {n: str(n) for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)
                
                # Title
                rel_name = self.id2relation.get(rel_id, f"R{rel_id}")
                ax.set_title(f"Triple {i+1}: R{rel_id} - {len(G.nodes())} nodes", fontsize=10)
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(subgraphs), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save grid visualization
        output_path = os.path.join(output_dir, f"subgraphs_grid_{num_samples}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved grid visualization to {output_path}")
        
        # Save individual plots
        subgraphs_individual = self.load_subgraphs(lmdb_path, min(10, num_samples))
        for i, subgraph_data in enumerate(subgraphs_individual):
            individual_path = os.path.join(output_dir, f"subgraph_{i+1}.png")
            try:
                self.visualize_subgraph(subgraph_data, individual_path, figsize=(10, 8))
                plt.close()  # Close individual plots to save memory
            except Exception as e:
                print(f"⚠️ Failed to create individual plot {i+1}: {e}")
        
        return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize RASG extracted subgraphs")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing LMDB files")
    parser.add_argument("--lmdb-file", type=str, default="valid.lmdb", help="LMDB file to visualize (train.lmdb, valid.lmdb, test.lmdb)")
    parser.add_argument("--mapping-dir", type=str, default="mappings", help="Mapping directory name")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of subgraphs to visualize")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Output directory for plots")
    parser.add_argument("--individual-only", action="store_true", help="Only create individual plots, skip grid")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 8], help="Figure size for individual plots")
    
    args = parser.parse_args()
    
    print("🎨 RASG Subgraph Visualizer")
    print("=" * 50)
    
    try:
        # Initialize visualizer
        visualizer = SubgraphVisualizer(args.data_root, args.mapping_dir)
        
        # Get LMDB path
        lmdb_path = os.path.join(args.data_root, args.lmdb_file)
        
        if not args.individual_only:
            # Create grid visualization
            print(f"🎯 Creating grid visualization for {args.num_samples} subgraphs...")
            visualizer.visualize_multiple_subgraphs(
                lmdb_path, 
                args.num_samples, 
                args.output_dir,
                tuple(args.figsize)
            )
        
        # Create individual detailed plots
        print(f"🔍 Creating individual detailed visualizations...")
        subgraphs = visualizer.load_subgraphs(lmdb_path, min(5, args.num_samples))
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        for i, subgraph_data in enumerate(subgraphs):
            output_path = os.path.join(args.output_dir, f"detailed_subgraph_{i+1}.png")
            try:
                fig, ax = visualizer.visualize_subgraph(subgraph_data, output_path, tuple(args.figsize), show_distances=True)
                plt.close(fig)
                print(f"✅ Created detailed plot {i+1}")
            except Exception as e:
                print(f"❌ Failed to create detailed plot {i+1}: {e}")
        
        print(f"🎉 Visualization completed! Check {args.output_dir}/ for results")
        print(f"📁 Files created:")
        print(f"   - subgraphs_grid_{args.num_samples}.png (overview)")
        print(f"   - detailed_subgraph_*.png (individual detailed plots)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()