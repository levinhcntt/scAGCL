
# scAGCL: Deep clustering of single-cell RNA-seq using Adversarial Graph Contrastive Learning
scAGLL is a method for clustering scRNA-seq data using adversarial graph contrastive learning. 


## Project contents
+ scAGCL.py: Contains the main function where to start running the application
+ model.py: The implementation of the proposed model
+ train.py: Contains the training process
+ utils.py: Contains utility functions supporting the solution implementation


## System Requirements
python==3.12.7
h5py==3.11.0
keras==3.6.0
networkx==3.3
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
scipy==1.13.1
torch==2.4.1
torch-geometric==2.6.0
torch-scatter==2.1.2
torchaudio==2.4.1
torchvision==0.19.1

## Datasets
15 datasets used in this work are from previous studies ( [scziDesk](https://github.com/theislab/sc_mae/tree/master),  [scMAE](https://github.com/xuebaliang/scziDesk), [scCAEs](https://github.com/gushenweiz/scCAEs) and can be directly downloaded here [Datasets](https://zenodo.org/records/15272789)

## Running Experiments
+ Create two folders named 'data', and 'result' in the project folder. Put dataset file (e.g, Pollen.h) into the 'data' folder. Output of the program are stored in the 'result' folder
+ Run the program: 
python scAGCL.py --data_file Pollen.h5 --num_cluster 11 --random_seed 12345

Options:
- `---data_file`: Name of dataset file
- `--num_cluster`: Number of clusters (cell types)
- `--lam`: Adversarial weight
- `--subgraph_size`: Size of subgraph used
- `--learning_rate`: Learning rate
- `--random_seed`: Random seed
- `--num_epochs`: Number of training epochs

## Authors
Author email: vinhlv@hcmute.edu.vn
