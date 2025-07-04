#!/usr/bin/env python3
"""
Calculate Real Subgraph Sizes from OGB-BioKG Dataset
Script for running on Kaggle to get actual average subgraph sizes for all 51 relation types
"""

import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
import pickle
import gzip
import csv
from pathlib import Path
import time
import random
from typing import Dict, List, Tuple, Set
from ogb.linkproppred import LinkPropPredDataset

class SubgraphSizeCalculator:
    def __init__(self, ogb_root: str = "./data/"):
        """
        Initialize calculator with OGB-BioKG using OGB library
        Args:
            ogb_root: Root directory for OGB datasets
        """
        self.ogb_root = ogb_root
        self.dataset = None
        self.graph = None
        
        # All 51 relation types
        self.all_relations = [
            "disease-protein",
            "drug-disease", 
            "drug-drug_acquired_metabolic_disease",
            "drug-drug_bacterial_infectious_disease",
            "drug-drug_benign_neoplasm",
            "drug-drug_cancer",
            "drug-drug_cardiovascular_system_disease",
            "drug-drug_chromosomal_disease",
            "drug-drug_cognitive_disorder",
            "drug-drug_cryptorchidism",
            "drug-drug_developmental_disorder_of_mental_health",
            "drug-drug_endocrine_system_disease",
            "drug-drug_fungal_infectious_disease",
            "drug-drug_gastrointestinal_system_disease",
            "drug-drug_hematopoietic_system_disease",
            "drug-drug_hematopoietic_system_diseases",
            "drug-drug_hypospadias",
            "drug-drug_immune_system_disease",
            "drug-drug_inherited_metabolic_disorder",
            "drug-drug_integumentary_system_disease",
            "drug-drug_irritable_bowel_syndrome",
            "drug-drug_monogenic_disease",
            "drug-drug_musculoskeletal_system_disease",
            "drug-drug_nervous_system_disease",
            "drug-drug_orofacial_cleft",
            "drug-drug_parasitic_infectious_disease",
            "drug-drug_personality_disorder",
            "drug-drug_polycystic_ovary_syndrome",
            "drug-drug_pre-malignant_neoplasm",
            "drug-drug_psoriatic_arthritis",
            "drug-drug_reproductive_system_disease",
            "drug-drug_respiratory_system_disease",
            "drug-drug_sexual_disorder",
            "drug-drug_sleep_disorder",
            "drug-drug_somatoform_disorder",
            "drug-drug_struct_sim",
            "drug-drug_substance-related_disorder",
            "drug-drug_thoracic_disease",
            "drug-drug_urinary_system_disease",
            "drug-drug_viral_infectious_disease",
            "drug-protein",
            "drug-sideeffect",
            "function-function",
            "protein-function",
            "protein-protein_activation",
            "protein-protein_binding",
            "protein-protein_catalysis",
            "protein-protein_expression",
            "protein-protein_inhibition",
            "protein-protein_ptmod",
            "protein-protein_reaction"
        ]
        
        # Graph data structures
        self.edge_dict = defaultdict(set)  # node -> set of neighbors
        self.relation_edges = defaultdict(list)  # relation -> list of (head, tail)
        self.node_to_id = {}
        self.id_to_node = {}
        
        print(f"Initialized calculator for {len(self.all_relations)} relation types")
    
    def load_graph_data(self):
        """Load graph data using OGB library (heterogeneous graph format)"""
        print("Loading OGB-BioKG dataset using OGB library...")
        
        try:
            # Load dataset using OGB
            self.dataset = LinkPropPredDataset(name='ogbl-biokg', root=self.ogb_root)
            self.graph = self.dataset[0]  # Get graph data
            
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Graph keys: {list(self.graph.keys())}")
            
            # OGB-BioKG is heterogeneous - has edge_index_dict, edge_reltype, num_nodes_dict
            edge_index_dict = self.graph['edge_index_dict']
            edge_reltype = self.graph['edge_reltype']
            num_nodes_dict = self.graph['num_nodes_dict']
            
            print(f"📊 Edge types: {list(edge_index_dict.keys())}")
            print(f"📊 Node types: {list(num_nodes_dict.keys())}")
            print(f"📊 Total node types: {len(num_nodes_dict)}")
            
            # Build unified graph structure from heterogeneous data
            total_edges = 0
            total_nodes = sum(num_nodes_dict.values())
            
            print(f"📊 Total nodes across all types: {total_nodes:,}")
            
            # Process each edge type in the heterogeneous graph
            for edge_type, edge_index in edge_index_dict.items():
                edge_count = edge_index.shape[1]
                total_edges += edge_count
                print(f"  {edge_type}: {edge_count:,} edges")
                
                # Add edges to global graph structure
                for i in range(edge_count):
                    head = edge_index[0, i].item()
                    tail = edge_index[1, i].item()
                    
                    # Add to global graph (undirected)
                    self.edge_dict[head].add(tail)
                    self.edge_dict[tail].add(head)
                    
                    # Map edge type to relation name for analysis
                    if edge_type in edge_reltype:
                        rel_types = edge_reltype[edge_type]
                        if i < len(rel_types):
                            rel_idx = rel_types[i].item() if hasattr(rel_types[i], 'item') else rel_types[i]
                            if rel_idx < len(self.all_relations):
                                relation_name = self.all_relations[rel_idx]
                                self.relation_edges[relation_name].append((head, tail))
                    else:
                        # Use edge_type as relation name if no mapping available
                        clean_edge_type = edge_type.replace('__', '_')
                        if clean_edge_type in self.all_relations:
                            self.relation_edges[clean_edge_type].append((head, tail))
            
            print(f"\n📊 Graph structure built:")
            print(f"  Total edges: {total_edges:,}")
            print(f"  Total nodes in adjacency: {len(self.edge_dict):,}")
            
            # Print relation statistics
            relations_found = 0
            for relation in self.all_relations:
                if relation in self.relation_edges and len(self.relation_edges[relation]) > 0:
                    count = len(self.relation_edges[relation])
                    print(f"  {relation}: {count:,} edges")
                    relations_found += 1
            
            print(f"📊 Found {relations_found} relations with data out of {len(self.all_relations)} total")
            
            if relations_found == 0:
                print("⚠️  No relations mapped, will use edge types directly")
                # Fallback: use edge types as relations
                for edge_type, edge_index in edge_index_dict.items():
                    clean_name = edge_type.replace('__', '_').replace('___', '_')
                    edges = [(edge_index[0, i].item(), edge_index[1, i].item()) 
                            for i in range(edge_index.shape[1])]
                    self.relation_edges[clean_name] = edges
                    print(f"  {clean_name}: {len(edges):,} edges")
            
            return total_edges > 0
            
        except Exception as e:
            print(f"❌ Error loading OGB dataset: {e}")
            import traceback
            traceback.print_exc()
            print("Make sure ogb library is installed: pip install ogb")
            return False
    
    def extract_k_hop_subgraph(self, center_nodes: List[int], k: int = 2, max_nodes: int = 2000) -> Set[int]:
        """
        Extract k-hop subgraph around center nodes with max nodes limit (matching run_comparison.sh)
        Args:
            center_nodes: List of center node IDs
            k: Number of hops (default=2, same as run_comparison.sh)
            max_nodes: Maximum nodes per subgraph (1000 for 1k, 2000 for 5k/10k, 5000 for full)
        Returns:
            Set of node IDs in the subgraph
        """
        subgraph_nodes = set(center_nodes)
        current_frontier = set(center_nodes)
        
        for hop in range(k):
            next_frontier = set()
            for node in current_frontier:
                neighbors = self.edge_dict.get(node, set())
                next_frontier.update(neighbors)
                
                # Stop if we exceed max_nodes limit
                if len(subgraph_nodes) + len(next_frontier) > max_nodes:
                    # Sample neighbors to stay within limit
                    remaining_slots = max_nodes - len(subgraph_nodes)
                    if remaining_slots > 0:
                        next_frontier = set(random.sample(list(next_frontier), 
                                                        min(remaining_slots, len(next_frontier))))
                    break
            
            subgraph_nodes.update(next_frontier)
            current_frontier = next_frontier - subgraph_nodes.union(current_frontier)
            
            if not current_frontier or len(subgraph_nodes) >= max_nodes:
                break
        
        return subgraph_nodes
    
    def calculate_relation_subgraph_sizes(self, k: int = 2, sample_size: int = 1000, max_nodes: int = 2000) -> Dict[str, Dict]:
        """
        Calculate average subgraph sizes for all relation types
        Args:
            k: Number of hops for subgraph extraction (default=2, same as run_comparison.sh)
            sample_size: Number of triples to sample per relation
            max_nodes: Maximum nodes per subgraph (1000 for 1k, 2000 for 5k/10k, 5000 for full)
        Returns:
            Dictionary with statistics for each relation
        """
        print(f"\n🔍 Calculating subgraph sizes (k={k}, sample_size={sample_size})...")
        
        results = {}
        
        for relation in self.all_relations:
            if relation not in self.relation_edges:
                print(f"⚠️  Skipping {relation} - no data loaded")
                continue
            
            edges = self.relation_edges[relation]
            
            if len(edges) == 0:
                print(f"⚠️  Skipping {relation} - no edges")
                continue
            
            # Sample edges if needed
            if sample_size and len(edges) > sample_size:
                sampled_edges = random.sample(edges, sample_size)
                print(f"📊 {relation}: Sampling {sample_size} from {len(edges):,} edges")
            else:
                sampled_edges = edges
                print(f"📊 {relation}: Using all {len(edges):,} edges")
            
            # Calculate subgraph sizes
            subgraph_sizes = []
            start_time = time.time()
            
            for i, (head, tail) in enumerate(sampled_edges):
                # Extract k-hop subgraph around head and tail
                subgraph_nodes = self.extract_k_hop_subgraph([head, tail], k=k, max_nodes=max_nodes)
                subgraph_sizes.append(len(subgraph_nodes))
                
                # Progress update
                if (i + 1) % 100 == 0 and sample_size and sample_size > 100:
                    elapsed = time.time() - start_time
                    progress = (i + 1) / len(sampled_edges) * 100
                    print(f"   Progress: {progress:.1f}% ({i+1}/{len(sampled_edges)}) - {elapsed:.1f}s")
            
            # Calculate statistics
            if subgraph_sizes:
                stats = {
                    'relation': relation,
                    'total_edges': len(edges),
                    'sampled_edges': len(sampled_edges),
                    'avg_subgraph_size': float(np.mean(subgraph_sizes)),
                    'std_subgraph_size': float(np.std(subgraph_sizes)),
                    'min_subgraph_size': int(np.min(subgraph_sizes)),
                    'max_subgraph_size': int(np.max(subgraph_sizes)),
                    'median_subgraph_size': float(np.median(subgraph_sizes)),
                    'k_hops': k,
                    'subgraph_sizes_sample': subgraph_sizes[:100] if len(subgraph_sizes) > 100 else subgraph_sizes
                }
                
                results[relation] = stats
                
                print(f"✅ {relation}: Avg={stats['avg_subgraph_size']:.1f} nodes "
                      f"(std={stats['std_subgraph_size']:.1f}, "
                      f"range={stats['min_subgraph_size']}-{stats['max_subgraph_size']})")
            else:
                print(f"❌ {relation}: No valid subgraphs extracted")
        
        return results
    
    def analyze_subgraph_statistics(self, results: Dict[str, Dict]) -> Dict[str, any]:
        """Analyze and summarize subgraph statistics"""
        
        print("\n📈 SUBGRAPH SIZE ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Sort by average subgraph size
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['avg_subgraph_size'], 
                               reverse=True)
        
        print("\n🏆 TOP 10 MOST COMPLEX RELATIONS (by avg subgraph size):")
        print("-" * 80)
        for i, (relation, stats) in enumerate(sorted_results[:10], 1):
            clean_name = relation.replace('_', '-')
            if len(clean_name) > 35:
                clean_name = clean_name[:32] + "..."
            print(f"{i:2d}. {clean_name:<35} | "
                  f"Avg: {stats['avg_subgraph_size']:6.1f} nodes | "
                  f"Edges: {stats['total_edges']:,}")
        
        print("\n🔢 STATISTICS BY CATEGORY:")
        print("-" * 60)
        
        # Group by category
        categories = {
            'drug-drug': [r for r in results.keys() if r.startswith('drug-drug_')],
            'protein-protein': [r for r in results.keys() if r.startswith('protein-protein_')],
            'other': [r for r in results.keys() if not r.startswith(('drug-drug_', 'protein-protein_'))]
        }
        
        for cat_name, relations in categories.items():
            if relations:
                avg_sizes = [results[r]['avg_subgraph_size'] for r in relations]
                print(f"{cat_name.upper():<15} | "
                      f"Relations: {len(relations):2d} | "
                      f"Avg Size: {np.mean(avg_sizes):6.1f} | "
                      f"Range: {np.min(avg_sizes):4.0f}-{np.max(avg_sizes):4.0f}")
        
        # Overall statistics
        all_avg_sizes = [stats['avg_subgraph_size'] for stats in results.values()]
        all_total_edges = [stats['total_edges'] for stats in results.values()]
        
        summary = {
            'total_relations_analyzed': len(results),
            'overall_avg_subgraph_size': float(np.mean(all_avg_sizes)),
            'overall_std_subgraph_size': float(np.std(all_avg_sizes)),
            'min_avg_subgraph_size': float(np.min(all_avg_sizes)),
            'max_avg_subgraph_size': float(np.max(all_avg_sizes)),
            'total_edges_analyzed': sum(all_total_edges)
        }
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Relations analyzed: {summary['total_relations_analyzed']}")
        print(f"   Overall avg subgraph size: {summary['overall_avg_subgraph_size']:.1f} ± {summary['overall_std_subgraph_size']:.1f}")
        print(f"   Range: {summary['min_avg_subgraph_size']:.1f} - {summary['max_avg_subgraph_size']:.1f}")
        print(f"   Total edges: {summary['total_edges_analyzed']:,}")
        
        return summary
    
    def save_results(self, results: Dict[str, Dict], summary: Dict, filename: str = "subgraph_sizes_analysis.json"):
        """Save results to JSON file"""
        
        output_data = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_relations': len(self.all_relations),
                'relations_analyzed': len(results),
                'k_hops': results[next(iter(results))]['k_hops'] if results else None
            },
            'summary_statistics': summary,
            'relation_results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        print(f"📄 File size: {Path(filename).stat().st_size / 1024:.1f} KB")
        
        return filename
    
    def run_full_analysis(self, dataset_size: str = "5k", k: int = 2, sample_size: int = 1000, output_file: str = "subgraph_sizes_analysis.json"):
        """Run complete subgraph size analysis with run_comparison.sh parameters"""
        
        # Set max_nodes based on dataset_size (matching run_comparison.sh)
        if dataset_size == "1k":
            max_nodes = 1000
            sample_size = min(sample_size, 200)  # Smaller sample for 1k
        elif dataset_size == "5k":
            max_nodes = 2000
            sample_size = min(sample_size, 500)
        elif dataset_size == "10k":
            max_nodes = 2000  # Same as 5k to avoid memory issues
            sample_size = min(sample_size, 800)
        else:  # full
            max_nodes = 5000
        
        print("🚀 STARTING SUBGRAPH SIZE ANALYSIS")
        print("=" * 80)
        print(f"Dataset size: {dataset_size}")
        print(f"Parameters: k={k} hops, max_nodes={max_nodes}, sample_size={sample_size}")
        print(f"Output file: {output_file}")
        print(f"Matching run_comparison.sh configuration for {dataset_size} dataset")
        
        start_time = time.time()
        
        # Step 1: Load graph data
        if not self.load_graph_data():
            print("❌ Failed to load graph data")
            return None, None  # Return tuple to avoid unpacking error
        
        # Step 2: Calculate subgraph sizes
        results = self.calculate_relation_subgraph_sizes(k=k, sample_size=sample_size, max_nodes=max_nodes)
        
        if not results:
            print("❌ No results generated")
            return None, None  # Return tuple to avoid unpacking error
        
        # Step 3: Analyze results
        summary = self.analyze_subgraph_statistics(results)
        
        # Step 4: Save results
        output_file = self.save_results(results, summary, output_file)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total analysis time: {total_time:.1f} seconds")
        print("✅ ANALYSIS COMPLETE!")
        
        return results, summary

def main(dataset_size="5k", k_hops=2, tau=2, sample_size=None, ogb_root=None, output_file=None):
    """Main function to run on Kaggle with run_comparison.sh parameters
    
    Args:
        dataset_size: Dataset size configuration (1k, 5k, 10k, full)
        k_hops: Number of hops for subgraph extraction (default: 2)
        tau: Tau parameter (default: 2)
        sample_size: Override sample size (None for auto)
        ogb_root: Override OGB root path (None for auto)
        output_file: Override output filename (None for auto)
    """
    
    # Configuration matching run_comparison.sh
    DATASET_SIZE = dataset_size
    K_HOPS = k_hops
    TAU = tau
    
    # Sample sizes matching run_comparison.sh limits
    SAMPLE_SIZE_MAP = {
        "1k": 100,   # TEST_LIMIT=100 in 1k preset
        "5k": 500,   # TEST_LIMIT=500 in 5k preset  
        "10k": 800,  # TEST_LIMIT=800 in 10k preset (adjusted)
        "full": 1000 # No limit, but sample for efficiency
    }
    
    # Override with provided parameters
    SAMPLE_SIZE = sample_size if sample_size is not None else SAMPLE_SIZE_MAP.get(DATASET_SIZE, 500)
    OUTPUT_FILE = output_file if output_file is not None else f"ogb_biokg_subgraph_sizes_{DATASET_SIZE}.json"
    
    # For Kaggle, adjust OGB root path
    import os
    if ogb_root is not None:
        OGB_ROOT = ogb_root
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        OGB_ROOT = "/kaggle/working/ogb_data/"  # Kaggle working directory
    else:
        OGB_ROOT = "./data/"  # Local path
    
    print("🧬 OGB-BioKG SUBGRAPH SIZE CALCULATOR")
    print("=" * 80)
    print(f"Configuration matching run_comparison.sh:")
    print(f"  📊 Dataset size: {DATASET_SIZE}")
    print(f"  🔗 K-hops: {K_HOPS}")
    print(f"  ⚡ Tau: {TAU}")
    print(f"  📈 Sample size: {SAMPLE_SIZE}")
    print(f"  📁 OGB root: {OGB_ROOT}")
    print(f"  💾 Output: {OUTPUT_FILE}")
    print("")
    
    # Initialize calculator
    calculator = SubgraphSizeCalculator(ogb_root=OGB_ROOT)
    
    # Run analysis
    try:
        results, summary = calculator.run_full_analysis(
            dataset_size=DATASET_SIZE,
            k=K_HOPS, 
            sample_size=SAMPLE_SIZE, 
            output_file=OUTPUT_FILE
        )
        
        if results:
            print(f"\n🎉 SUCCESS! Generated subgraph size data for {len(results)} relations")
            print(f"📁 Download file: {OUTPUT_FILE}")
            print(f"🔧 Configuration: {DATASET_SIZE} dataset, k={K_HOPS}, sample={SAMPLE_SIZE}")
            print(f"📊 This matches the subgraph extraction used in run_comparison.sh")
            return results
        else:
            print("❌ Analysis failed")
            return None
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate real subgraph sizes from OGB-BioKG')
    parser.add_argument('--dataset-size', '-s', default='5k', choices=['1k', '5k', '10k', 'full'],
                        help='Dataset size configuration (default: 5k)')
    parser.add_argument('--k-hops', '-k', type=int, default=2,
                        help='Number of hops for subgraph extraction (default: 2)')
    parser.add_argument('--tau', '-t', type=int, default=2,
                        help='Tau parameter (default: 2)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Override sample size (default: auto based on dataset-size)')
    parser.add_argument('--ogb-root', default=None,
                        help='OGB root directory (default: auto detect)')
    parser.add_argument('--output-file', '-o', default=None,
                        help='Output filename (default: auto generate)')
    
    args = parser.parse_args()
    
    # Run main analysis
    main(
        dataset_size=args.dataset_size,
        k_hops=args.k_hops,
        tau=args.tau,
        sample_size=args.sample_size,
        ogb_root=args.ogb_root,
        output_file=args.output_file
    )