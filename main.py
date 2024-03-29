import json
import os
import random
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



from datasets import node_prediction
from models import EGNNC, MLPTwoLayers, CombinedModel
import utils

random.seed(0)

def main():
    config = utils.parse_args()
    device = 'cuda:0' if config['cuda'] and torch.cuda.is_available() else 'cpu'
    config['device'] = device

    dataset_args = ('train', config['num_layers']) if not config['val'] and not config['test'] else ('val', config['num_layers']) if config['val'] else ('test', config['num_layers'])
    datasets = utils.get_node_dataset_gcn(dataset_args, config['dataset_folder'], is_debug=config["is_debug"])
    loaders = [DataLoader(dataset=ds, batch_size=config['batch_size'], shuffle=True, collate_fn=ds.collate_wrapper) for ds in datasets]

    input_dim, hidden_dim, output_dim = datasets[0].get_dims()[0], config['hidden_dims'][0], config['out_dim']
    channel_dim = datasets[0].get_channel()

    model = EGNNC(input_dim, hidden_dim, output_dim, channel_dim, config['num_layers'], config['dropout'], config['device']).to(config['device'])
    mlp = MLPTwoLayers(input_dim=channel_dim*output_dim, hidden_dim=output_dim, output_dim=1, dropout=0.5).to(config["device"])
    combined_model = CombinedModel(model, mlp, config["device"]).to(config["device"])

    print("Combined model: ", combined_model)

    if config['val'] or config['test']:
        path = os.path.join(config['saved_models_dir'], utils.get_fname(config))
        combined_model.load_state_dict(torch.load(path, map_location=torch.device(device)))

    criterion = utils.compute_weakly_loss
    optimizer = optim.Adam(combined_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5)

    if not config['val'] and not config['test']:
        combined_model.train()
        for epoch in range(config['epochs']):
            
            for i, loader in enumerate(loaders):
                for idx, batch in enumerate(loader):
                    adj, features, edge_features, nodes, labels, _ = batch
                    adj, labels, features, edge_features = adj.to(device), labels.to(device), features.to(device), edge_features.to(device)
                    scores = combined_model(features, edge_features, nodes)
                    
                    loss = criterion(scores, labels, nodes, adj)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            
            print(f"Epoch {epoch} loss: {loss.item()}")

            scheduler.step()

        # Save trained model
        if config['save']:
            path = os.path.join(config['saved_models_dir'], utils.get_fname(config))
            torch.save(combined_model.state_dict(), path)
            print(f"Saved model to {path}")

        # save results of node classification
        combined_model.eval()
        for i, loader in enumerate(loaders):
            all_node_ids = []
            all_true_labels = []
            all_scores = []

            for idx, batch in enumerate(loader):
                adj, features, edge_features, nodes, labels, _ = batch
                labels, features, edge_features = labels.to(device), features.to(device), edge_features.to(device)
                scores = combined_model(features, edge_features, nodes)
                
                # Convert to CPU and NumPy for saving
                labels = labels.cpu().numpy()
                scores = scores.detach().cpu().numpy()
                
                # Append to lists
                all_node_ids.extend(nodes.tolist())
                all_true_labels.extend(labels.tolist())
                all_scores.extend(scores.tolist())
                
            # Sort the accumulated results by node_id
            sorted_results = sorted(zip(all_node_ids, all_true_labels, all_scores), key=lambda x: x[0])
            
            # Unzip back into separate lists
            sorted_node_ids, sorted_true_labels, sorted_scores = zip(*sorted_results)
            
            
            filtered_true_labels = [label for label in sorted_true_labels if label!=-1]
            filtered_scores = [score for label, score in zip(sorted_true_labels, sorted_scores) if label!=-1]

            # Save sorted results
            image_name = datasets[i].path[0].split("\\")[1]
            name = f'results_{image_name}_nodes.json'
            task = 'train'
            filename = os.path.join(config['results_dir'],'node_classification_results', task,  name)
            with open(filename, 'w') as f:
                json.dump({
                    'sorted_node_ids': sorted_node_ids,
                    'sorted_true_labels': sorted_true_labels,
                    'sorted_scores': sorted_scores
                }, f)

    # Evaluation (Validation or Test)
    if config['val'] or config['test']:
        combined_model.eval()
        
        overall_accuracy = 0
        overall_precision = 0
        overall_recall = 0
        overall_f1 = 0
        num_graphs = 0


        for i, loader in enumerate(loaders):
            all_node_ids = []
            all_true_labels = []
            all_scores = []
            
            for idx, batch in enumerate(loader):
                adj, features, edge_features, nodes, labels, _ = batch
                labels, features, edge_features = labels.to(device), features.to(device), edge_features.to(device)
                scores = combined_model(features, edge_features, nodes)
                
                # Convert to CPU and NumPy for saving
                labels = labels.cpu().numpy()
                scores = scores.detach().cpu().numpy()
                
                # Append to lists
                all_node_ids.extend(nodes.tolist())
                all_true_labels.extend(labels.tolist())
                all_scores.extend(scores.tolist())
                
            # Sort the accumulated results by node_id
            sorted_results = sorted(zip(all_node_ids, all_true_labels, all_scores), key=lambda x: x[0])
            
            # Unzip back into separate lists
            sorted_node_ids, sorted_true_labels, sorted_scores = zip(*sorted_results)
            
            
            filtered_true_labels = [label for label in sorted_true_labels if label!=-1]
            filtered_scores = [score for label, score in zip(sorted_true_labels, sorted_scores) if label!=-1]

            # Compute metrics only on labeled data
            if len(filtered_true_labels) > 0:  # Check to make sure there's labeled data
                accuracy = accuracy_score(filtered_true_labels, np.round(filtered_scores))
                precision = precision_score(filtered_true_labels, np.round(filtered_scores))
                recall = recall_score(filtered_true_labels, np.round(filtered_scores))
                f1 = f1_score(filtered_true_labels, np.round(filtered_scores))
                
                overall_accuracy += accuracy
                overall_precision += precision
                overall_recall += recall
                overall_f1 += f1
                num_graphs += 1

            else:
                print("No labeled data for metrics computation.")


            # Save sorted results
            image_name = datasets[i].path[0].split("\\")[1]
            name = f'results_{image_name}_nodes.json'
            task = 'val' if config['val'] else 'test'
            filename = os.path.join(config['results_dir'],'node_classification_results', task,  name)
            with open(filename, 'w') as f:
                json.dump({
                    'sorted_node_ids': sorted_node_ids,
                    'sorted_true_labels': sorted_true_labels,
                    'sorted_scores': sorted_scores
                }, f)

        # Compute average metrics
        overall_accuracy /= num_graphs
        overall_precision /= num_graphs
        overall_recall /= num_graphs
        overall_f1 /= num_graphs

        print(f"Accuracy: {overall_accuracy}")
        print(f"Precision: {overall_precision}")
        print(f"Recall: {overall_recall}")
        print(f"F1: {overall_f1}")

        

if __name__ == '__main__':
    main()







    