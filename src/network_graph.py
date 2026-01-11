"""
MARITIME DISRUPTION PROJECT - Network Graph Construction & Feature Engineering

Builds a directional trade network from processed connectivity data, computes
country-level centrality and community features, and exports a cleaned numeric
feature table for downstream modeling and analysis.

Reads:  data/processed/df_Combined_Connectivity_standardized.csv
Writes: data/processed/df_Network_Centrality_Features.csv

Main steps:
- Construct directed graph from mean connectivity_index by (from_country, to_country).
- Use the largest connected component for shortest-path metrics and add an
  inverted weight (1 / connectivity_index) to represent distance/cost.
- Compute weighted degree, PageRank, closeness (distance-weighted), inverted
  eccentricity, clustering coefficient, and Louvain communities.
- Assemble results into CSV, filling missing values and rounding numerics.
"""

import pandas as pd
import networkx as nx
import community as community_louvain  
from pathlib import Path
import warnings
import numpy as np

# Configuration and Paths
PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
OUTPUT_FILE = PROCESSED_DIR / 'df_Network_Centrality_Features.csv'

def run_network_analysis():
    print("\n" + "-"*100)
    print("NETWORK GRAPH CONSTRUCTION & FEATURE ENGINEERING")
    print("-"*100)

    # 1. Load Edge Data
    edge_file = PROCESSED_DIR / 'df_Combined_Connectivity_standardized.csv'
    if not edge_file.exists():
        raise FileNotFoundError(f"Missing edge data: {edge_file}")
    
    print("- Loading Connectivity Data (Edges)...")
    df_edges = pd.read_csv(edge_file, low_memory=False)
    
    df_graph_input = df_edges.groupby(['from_country', 'to_country'])['connectivity_index'].mean().reset_index()
    df_graph_input = df_graph_input[df_graph_input['connectivity_index'] > 0] 

    print(f"  > Graph built from {len(df_graph_input)} unique trade routes.")

    # 2. Build NetworkX Graph
    print("- Building Directional Graph (DiGraph)...")
    G = nx.from_pandas_edgelist(
        df_graph_input,
        source='from_country',
        target='to_country',
        edge_attr='connectivity_index',
        create_using=nx.DiGraph()
    )
    
    G_undirected = G.to_undirected()
    
    if not nx.is_connected(G_undirected):
        print("  [CRITICAL WARNING] Graph is disconnected. Centrality results may be misleading.")
        # Ensure to only work on the largest component for shortest-path metrics
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_cc = G_undirected.subgraph(largest_cc).copy()
    else:
        G_cc = G_undirected.copy()


    print(f"  > Nodes (Countries): {G.number_of_nodes()}")
    print(f"  > Edges (Routes):    {G.number_of_edges()}")

    # Prepare Weights
    epsilon = 1e-6
    # Add inverted weight attribute (1/weight) for shortest path metrics (distance/cost)
    for u, v, data in G_cc.edges(data=True):
        data['weight_inv'] = 1.0 / (data['connectivity_index'] + epsilon)

    # 3. Calculate Network Metrics 
    print("- Calculating Network Centrality Measures...")
    
    # A. Weighted Degree Centrality
    degree_dict = dict(G.degree(weight='connectivity_index'))
    
    # B. PageRank
    print("  > Computing PageRank...")
    pagerank_dict = nx.pagerank(G, weight='connectivity_index')

    # C. Closeness Centrality 
    print("  > Computing Closeness Centrality...")
    try:
        closeness_dict_cc = nx.closeness_centrality(G_cc, distance='weight_inv')
        # Map back to full node set, filling missing nodes with 0
        closeness_dict = {node: closeness_dict_cc.get(node, 0.0) for node in G.nodes()}
    except Exception as e:
        print(f"    (Error: Closeness failed: {e}. Defaulting to 0.0)")
        closeness_dict = {node: 0.0 for node in G.nodes()}

    # D. Eccentricity
    print("  > Computing Eccentricity...")
    try:
        # Use simple nx.eccentricity calculation 
        eccentricity_dict_cc = nx.eccentricity(G_cc, weight='weight_inv')
        # Eccentricity should be inverted/scaled (higher = more central)
        max_ecc = max(eccentricity_dict_cc.values())
        eccentricity_dict_cc = {node: max_ecc / val for node, val in eccentricity_dict_cc.items() if val > 0}
        # Map back to full node set
        eccentricity_dict = {node: eccentricity_dict_cc.get(node, 0.0) for node in G.nodes()}

    except Exception as e:
        print(f"    (Error: Eccentricity failed: {e}. Defaulting to 0.0)")
        eccentricity_dict = {node: 0.0 for node in G.nodes()}
        
    # E. Clustering Coefficient
    print("  > Computing Clustering Coefficient...")
    clustering_coefficient_dict = nx.clustering(G_undirected, weight='connectivity_index') 

    # F. Community Detection (Louvain)
    print("  > Detecting Communities (Louvain)...")
    try:
        partition = community_louvain.best_partition(G_undirected, weight='connectivity_index')
    except Exception as e:
        print(f"    (Warning: Community detection failed: {e})")
        partition = {node: 0 for node in G.nodes()}

    # 4. Compile Features into a DataFrame
    print("- Compiling Country-Level Network Features...")
    df_features = pd.DataFrame({
        'country_name': list(G.nodes())
    })
    
    # Map dictionary values to the dataframe
    df_features['net_degree'] = df_features['country_name'].map(degree_dict)
    df_features['net_closeness'] = df_features['country_name'].map(closeness_dict)
    df_features['net_eccentricity_inv'] = df_features['country_name'].map(eccentricity_dict)
    df_features['net_pagerank'] = df_features['country_name'].map(pagerank_dict)
    df_features['net_clustering'] = df_features['country_name'].map(clustering_coefficient_dict)
    df_features['net_community'] = df_features['country_name'].map(partition)
    
    # 5. Final Save
    print("- Saving final feature set...")
    
    # Ensure all columns are clean
    final_numeric_cols = ['net_degree', 'net_closeness', 'net_eccentricity_inv', 'net_pagerank', 'net_clustering']
    df_features[final_numeric_cols] = df_features[final_numeric_cols].fillna(0).round(6)
    
    df_features.to_csv(OUTPUT_FILE, index=False)
    print(f"  ✓ Saved Network Features to: {OUTPUT_FILE}")
    print(f"  ✓ Shape: {df_features.shape}")
    print("="*60)

if __name__ == "__main__":
    run_network_analysis()