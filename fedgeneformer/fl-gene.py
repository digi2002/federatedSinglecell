import os
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# imports
from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns; sns.set()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
import flwr as fl
import wandb
import os
import transformers
import torch
import numpy as np
from typing import Dict, List, Tuple
from geneformer import DataCollatorForCellClassification


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('no', type=str, help='Client ID')
args = parser.parse_args()

RANDOM_SEED = 80
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

wandb.init(
    # set the wandb project where this run will be logged
    project="SC-project{}".format('100epoch-231027v1.0'),
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-5,
    "architecture": "Bert",
    "dataset": "zheng",
    "epochs": 100,
    }
)

# load cell type dataset (includes all tissues)
train_dataset=load_from_disk("/DatasetPATH")

dataset_list = []
evalset_list = []
organ_list = []
target_dict_list = []

for organ in Counter(train_dataset["organ_major"]).keys():
    # collect list of tissues for fine-tuning (immune and bone marrow are included together)
    if organ in ["bone_marrow"]:  
        continue
    elif organ=="immune":
        organ_ids = ["immune","bone_marrow"]
        organ_list += ["immune"]
    else:
        organ_ids = [organ]
        organ_list += [organ]
    
    print(organ)
    
    # filter datasets for given organ
    def if_organ(example):
        return example["organ_major"] in organ_ids
    trainset_organ = train_dataset.filter(if_organ, num_proc=16)
    
    # per scDeepsort published method, drop cell types representing <0.5% of cells
    celltype_counter = Counter(trainset_organ["cell_type"])
    total_cells = sum(celltype_counter.values())
    cells_to_keep = [k for k,v in celltype_counter.items() if v>(0.005*total_cells)]
    def if_not_rare_celltype(example):
        return example["cell_type"] in cells_to_keep
    trainset_organ_subset = trainset_organ.filter(if_not_rare_celltype, num_proc=16)
      
    # shuffle datasets and rename columns
    trainset_organ_shuffled = trainset_organ_subset.shuffle(seed=RANDOM_SEED)
    trainset_organ_shuffled = trainset_organ_shuffled.rename_column("cell_type","label")
    trainset_organ_shuffled = trainset_organ_shuffled.remove_columns("organ_major")
    
    # create dictionary of cell types : label ids
    target_names = list(Counter(trainset_organ_shuffled["label"]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    target_dict_list += [target_name_id_dict]
    
    # change labels to numerical ids
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)
    DATA_SET_SIZE = int(len(labeled_trainset) // 6)
    labeled_train_split = labeled_trainset.select([i for i in range(round(float(args.no)*DATA_SET_SIZE),round(float(args.no)+1)*DATA_SET_SIZE)])
    labeled_eval_split = labeled_trainset.select([i for i in range(round(len(labeled_trainset)*0.8),len(labeled_trainset))])
    



    # filter dataset for cell types in corresponding training set
    trained_labels = list(Counter(labeled_train_split["label"]).keys())
    def if_trained_label(example):
        return example["label"] in trained_labels
    labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)

    dataset_list += [labeled_train_split]
    evalset_list += [labeled_eval_split_subset]

trainset_dict = dict(zip(organ_list,dataset_list))
traintargetdict_dict = dict(zip(organ_list,target_dict_list))

evalset_dict = dict(zip(organ_list,evalset_list))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }

# set model parameters
# max input size
max_input_size = 2 ** 11  # 2048

# set training hyperparameters
# max learning rate
max_lr = 1e-5
# how many pretrained layers to freeze
freeze_layers = 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 128
# batch size for training and eval
# geneformer_batch_size = 12
geneformer_batch_size = 2
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 0
# number of epochs
epochs = 1
# optimizer
optimizer = "adamw"
step = 0
for organ in organ_list:
    organ_trainset = trainset_dict[organ]
    organ_evalset = evalset_dict[organ]
    organ_label_dict = traintargetdict_dict[organ]
    print(organ_trainset)
    print(organ_trainset['input_ids'][0])   
    print(organ_trainset['label'][0])
    print(organ_trainset['length'][0])

for organ in organ_list:
    print(organ)
    organ_trainset = trainset_dict[organ]
    organ_evalset = evalset_dict[organ]
    organ_label_dict = traintargetdict_dict[organ]
    print(organ_trainset)
    print(organ_label_dict)
    
    # set logging steps
    logging_steps = round(len(organ_trainset)/geneformer_batch_size/10)
    # logging_steps = 1
    print('****'*10)
    print(logging_steps)
    
    # reload pretrained model
    model = BertForSequenceClassification.from_pretrained("/PretrainedModelPATH", 
                                                      num_labels=len(organ_label_dict.keys()),
                                                      output_attentions = False,
                                                      output_hidden_states = False).to("cuda")
    
    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    output_dir = f"/home/shenbochen/Geneformer_v4.4.{args.no}/"
    
    # ensure not overwriting previously saved model
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")

    # make output directory
    subprocess.call(f'mkdir {output_dir}', shell=True)
    
    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
    }
    
    training_args_init = TrainingArguments(**training_args)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,param.size())

    # create the trainer




    
class BertClient(fl.client.NumPyClient):
    """Flower client classification using PyTorch."""

    def __init__(
        self,
        model,
    ) -> None:
        self.model = model
        self.step = 0
        self.unfreeze_layers = ['classifier','layer.11','layer.10','pooler']
        self.fed_layers = ['classifier','layer.11','layer.10','pooler']

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        fine_tuning_dict = []
        for name ,param in self.model.state_dict().items():
            for ele in self.fed_layers:
                if ele in name:
                    fine_tuning_dict.append(param)
                    break
        return [val.cpu().numpy() for val in fine_tuning_dict]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        names = []
        for name in self.model.state_dict().keys():
            for ele in self.fed_layers:
                if ele in name:
                    names.append(name)
        parameters = [torch.tensor(v) for v in parameters]
        
        params_dict = zip(names, parameters)

        update_dict = self.model.state_dict()
        update_dict.update(params_dict)
        self.model.load_state_dict(update_dict)


    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters

        self.set_parameters(parameters)
        for name ,param in self.model.named_parameters():
            param.requires_grad = False
            for ele in self.unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        trainer = Trainer(
            model=self.model,
            args=training_args_init,
            data_collator=DataCollatorForCellClassification(),
            train_dataset=organ_trainset,
            eval_dataset=organ_evalset,
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.model = BertForSequenceClassification.from_pretrained(output_dir, 
                                                            num_labels=len(organ_label_dict.keys()),
                                                            output_attentions = False,
                                                            output_hidden_states = False).to("cuda")
        
        return self.get_parameters(config={}), DatasetSize, {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        trainer = Trainer(
            model=self.model,
            args=training_args_init,
            data_collator=DataCollatorForCellClassification(),
            train_dataset=organ_trainset,
            eval_dataset=organ_evalset,
            compute_metrics=compute_metrics
        )
        predictions = trainer.predict(organ_evalset)
        with open(f"{output_dir}predictions.pickle", "wb") as fp:
            pickle.dump(predictions, fp)
        trainer.save_metrics("eval",predictions.metrics)

        wandb.log({"eval-acc": float(predictions.metrics['test_accuracy']), "eval-f1": float(predictions.metrics['test_macro_f1'])},step=self.step)
        self.step += 1
        

        return float(predictions.metrics['test_loss']), DatasetSize, {"accuracy": float(predictions.metrics['test_accuracy'])}
DatasetSize = 4000    
client = BertClient(model)
fl.client.start_numpy_client(server_address="0.0.0.0:8088", client=client)

