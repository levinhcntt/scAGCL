import argparse
import random
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from model import Encoder, AGCLModel
from utils import load_h5_data1, load_h5_data2, GraphConstruction
from train import Train
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score,  normalized_mutual_info_score)
from sklearn.preprocessing import normalize



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='Pollen.h5')
    parser.add_argument('--num_cluster', type=int, default=8)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--subgraph_size', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--random_seed', type=int, default=12345)   
    parser.add_argument('--num_epochs', type=int, default=500)   
    parser.add_argument('--num_itersAdv', type=int, default=10)
    args = parser.parse_args()

    config = {
        'num_hidden': 256,
        'num_proj_hidden': 256,
        'num_layers': 2,
        'weight_decay': 0.00001,
        'alpha': 100,
        'beta' : 0.01,
        'edge_r1': 0.4,
        'edge_r2': 0.3,
        'feature_r1': 0.3,
        'feature_r2': 0.4,
        'tau': 0.5
    }
        
    num_hidden = config['num_hidden']
    num_proj_hidden =config['num_proj_hidden']
    num_layers = config['num_layers']
    weight_decay = config['weight_decay']
    alpha = config["alpha"] 
    beta = config["beta"] 
    edge_r1 = config['edge_r1']
    edge_r2 = config['edge_r2']
    feature_r1 = config['feature_r1']
    feature_r2 = config['feature_r2']
    tau=config['tau']

    data_file=args.data_file
    num_cluster=args.num_cluster
    lam = args.lam #adversarial weight
    subgraph_size=args.subgraph_size
    learning_rate=args.learning_rate

    num_epochs = args.num_epochs
    num_itersAdv=args.num_itersAdv
    random_seed=args.random_seed

    torch.manual_seed(random_seed)
    random.seed(random_seed) 
    np.random.seed(random_seed)

    # Read input data
    data_path='./data/' + data_file
    X, Y = load_h5_data1(data_path)   # h5 file has X and Y field
    #X, Y = load_h5_data2(data_path)   #h5 file in which X in "exprs" and Y in "obs" field

    data = Data(x=torch.from_numpy(X))
    cellGraph=GraphConstruction(data, X, num_cluster)
    data.y=torch.tensor(Y, dtype=torch.int64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    #Build a model
    encoder = Encoder(data.num_features, num_hidden, num_layers=num_layers).to(device)
    model = AGCLModel(encoder, num_hidden, num_proj_hidden, tau).to(device)
    #optimizer   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    #Training
    embeddings=Train(model, data, cellGraph, device, num_epochs,lam, alpha, beta, num_itersAdv, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size)
    #Cell Assigment and Evaluation
    Z_eval = embeddings.clone()
    Y=data.y
    Z = Z_eval.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Z = normalize(Z, norm='l2')
    kmeans = KMeans(n_clusters=num_cluster, init="k-means++", random_state=0)
    pred = kmeans.fit_predict(Z)

    ari_score=adjusted_rand_score(Y, pred)
    nmi_score=normalized_mutual_info_score(Y, pred)

    print('Final result: ARI= ' + str(ari_score) + ', NMI=' + str(nmi_score))

    output_file=data_file + '_output.csv'
    with open("./result/" + output_file, "a") as f:
        print("Final result  is: " + 'ARI= ' + str(ari_score) + ', NMI=' + str(nmi_score), file=f)    
  
    y_pred_out = pd.DataFrame(pred)
    y_pred_out.to_csv("./result/" + data_file + "_labels.csv", sep='\t')