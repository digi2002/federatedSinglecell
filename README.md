#scFed: Federated learning for cell type classification with scRNA-seq
This repo contains the source code of the Python package federatedSinglecell. See our paper for a detailed description of federatedSinglecell.

##Overview
Four major modules are included:
·FedSVM
·FedACTINN
·FedXGBoost
·FedGeneformer

##Prerequisite
NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
We recommend running this repository using Anaconda. All dependencies for defining the environment are provided in requirements.txt
Use tokenizing_scRNAseq_data.ipynb to prepare FedGeneformer data.

##Run
·FedSVM
      cd FedSVM
      python main_federatedSVM.py
·FedACTINN
      cd FedACTINN
      python main_federatedACTINN.py
·FedXGBoost
      cd FedXGBoost
      python main_xgb.py
·FedGeneformer
      cd FedGeneformer
      python fl-server.py
      python fl-gene.py {clientId}

