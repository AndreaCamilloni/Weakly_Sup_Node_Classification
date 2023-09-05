import pandas as pd
import numpy as np

import os
import sys



task = 'val/'
path = os.path.join('../results/node_classification_results',task)
files = os.listdir(path)
files = [os.path.join(path, f) for f in files if f.endswith('.json')]


def read_data(nodes, edges):
    with open(edges, 'r') as f:
        first_line = f.readline()
        edges_cols = first_line.split(',')
        edges_cols = [col.strip() for col in edges_cols]
        edges_cols = [col.strip('"') for col in edges_cols]
    
    # read the first line of edges file to get columns names
    with open(nodes, 'r') as f:
        first_line = f.readline()
        nodes_cols = first_line.split(',')
        nodes_cols = [col.strip() for col in nodes_cols]
        nodes_cols = [col.strip('"') for col in nodes_cols]

    edges_df = pd.read_csv(edges, header=None, names=edges_cols)[1:]
    nodes_df = pd.read_csv(nodes, header=None, names=nodes_cols)[1:]

    edges_df['source'] = edges_df['source'].astype(int)
    edges_df['target'] = edges_df['target'].astype(int)
    edges_df['type'] = edges_df['type'].astype(int)
    edges_df['distance'] = edges_df['distance'].astype(float)
    edges_df['pred'] = 0 


    nodes_df['id'] = nodes_df['id'].astype(int)
    nodes_df['lym'] = nodes_df['lym'].astype(float)
    nodes_df['epi'] = nodes_df['epi'].astype(float)
    nodes_df['fib'] = nodes_df['fib'].astype(float)
    nodes_df['inf'] = nodes_df['inf'].astype(float)
    nodes_df['x'] = nodes_df['x'].astype(float)
    nodes_df['y'] = nodes_df['y'].astype(float)

    return nodes_df, edges_df

precision = []
recall = []
f1 = []

TP, TN, FP, FN = 0, 0, 0, 0

for i, file in enumerate(files):

    tile = file.split('\\')[-1].split('.')[0].split('results_')[1].split('_nodes')[0]
    print(tile)
    json_annot = path + "/results_" + tile + "_nodes.json"

    pred = pd.read_json(json_annot)
    pred.index = pred.index + 1
    pred["labels"] = (pred["sorted_scores"] > 0.5).astype(int)


    edges = 'Datasets/ground_truth/'+ task + tile + '_delaunay_orig_forGraphSAGE_edges.csv'
    nodes = 'Datasets/ground_truth/'+ task + tile + '_delaunay_orig_forGraphSAGE_nodes.csv'
    #image_path = '../../IntelliGraph/slides/' + tile + '.tif' 

    nodes_df, edges_df = read_data(nodes, edges)

    for index, row in edges_df.iterrows():
    
        if pred[pred['sorted_node_ids'] == row['source']]['labels'].values[0] != pred[pred['sorted_node_ids'] == row['target']]['labels'].values[0]:
            edges_df.loc[index, 'pred'] = 1
        else:
            edges_df.loc[index, 'pred'] = 0
       
    # Save the results
    edges_df[edges_df['pred'] == 1][['source', 'target','pred']].to_json('../results/node_classification_results/edge_results/results_' + tile + '_edges.json', orient='values')

    TP0 = edges_df[(edges_df['pred'] == 1) & (edges_df['type'] == 1)].shape[0]
    TN0 = edges_df[(edges_df['pred'] == 0) & (edges_df['type'] == 0)].shape[0]
    FP0 = edges_df[(edges_df['pred'] == 1) & (edges_df['type'] == 0)].shape[0]
    FN0 = edges_df[(edges_df['pred'] == 0) & (edges_df['type'] == 1)].shape[0]

    precision = TP0 / (TP0 + FP0)
    recall = TP0 / (TP0 + FN0)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f1)

    TP += TP0
    TN += TN0
    FP += FP0
    FN += FN0

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print("Final results: ")
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', f1)