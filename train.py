from time import perf_counter as t
import numpy as np
import torch
import networkx as nx
from utils import edge2adj, EdgeDropping, GeneDropping, GraphAdversarialAttack


def Train(scAGCLmodel, data, cellGgraph, device, num_epochs, lam, alpha, beta, iters, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size):
    start_time = t()
    scAGCLmodel.train()   
   
    Z_final=None
    bessloss=None
    
    for epoch in range(1, num_epochs + 2): #do not use final round       
        scAGCLmodel.eval()
        X=data.x
        Adj = edge2adj(X, data.edge_index)
        X = X.to(device)
        Adj = Adj.to(device)
        Z_current = scAGCLmodel(X, Adj) #get embeddings    
        scAGCLmodel.train() #back to continue training

        subGraph = cellGgraph.subgraph(np.random.permutation(cellGgraph.number_of_nodes())[:subgraph_size])
        x_sub = data.x[np.array(subGraph.nodes())].to(device)
        subGraph = nx.relabel.convert_node_labels_to_integers(subGraph, first_label=0, ordering='default')      
        edgeind = np.array(subGraph .edges()).T
        edgeind = torch.LongTensor(np.hstack([edgeind,edgeind[::-1]])).to(device)

        optimizer.zero_grad()
        x_1 = GeneDropping(x_sub, feature_r1)
        x_2 = GeneDropping(x_sub, feature_r2)

        edgeind_1 = EdgeDropping(edgeind, p=edge_r1, force_undirected=True)[0]
        edgeind_2 = EdgeDropping(edgeind, p=edge_r2, force_undirected=True)[0]
        adj_1 = edge2adj(x_1, edgeind_1)
        adj_2 = edge2adj(x_2, edgeind_2)

        z_1 = scAGCLmodel(x_1, adj_1)
        z_2 = scAGCLmodel(x_2, adj_2)    
        loss1= scAGCLmodel.loss(z_1,z_2,batch_size=0)
        
        loss2=0
        if lam > 0:
            adj_3, x_3 = GraphAdversarialAttack(scAGCLmodel, edgeind, edgeind_1, x_sub, x_1, iters, 0.2, alpha, beta)
            adj_4, x_4 = GraphAdversarialAttack(scAGCLmodel, edgeind, edgeind_2, x_sub, x_2, iters, 0.2, alpha, beta)
            z_3 = scAGCLmodel(x_3,adj_3)
            z_4 = scAGCLmodel(x_4,adj_4)
            loss2 = scAGCLmodel.loss(z_3,z_4,batch_size=0)
        
        loss = loss1 + lam*loss2


        loss.backward()
        optimizer.step()

        if epoch <= num_epochs and epoch%20==0:
            now_time = t()
            print(f'Epoch={epoch:03d}, loss1={loss1:.4f}, loss2={loss2:.4f}, total loss={loss:.4f}, total time {now_time - start_time:.4f}')

        if epoch == 1:
            bessloss = loss
            Z_final=Z_current
            
        if loss < bessloss:
            bessloss = loss
            Z_final=Z_current

    return Z_final
