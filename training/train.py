#import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch, os, collections
import pandas as pd
import pickle5 as pickle
from tqdm import tqdm
from models import IonicProtein
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtEmbDataset(Dataset):
    """Protein emb dataset."""
    def __init__(self, model_name, transform=None):
        """
        Args:
            root_dir (string): Directory with all the datapoints.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = '/media/dell4/a87d4a0b-5641-40d3-8d10-416948fc2996/ION_DATA/%s_batch128/'%(model_name)
        self.filenames = os.listdir(self.data_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_ = self.filenames[idx]
        with open(self.data_path+file_, 'rb') as handle:
            datapoint = pickle.load(handle)
        embs = datapoint['embs']
        labels = datapoint['labels']
        return embs, labels

from sklearn.metrics import f1_score
def train_val(model, dataloader, optimizer, criterion_1, is_training, device, topk, interval, e):
    batch_cnt = len(dataloader)
    fields = list(dataloader.dataset[0][1].keys())
    #accuracies = [0.0]*len(fields)
    f1_scores = [0.0]*len(fields)
    status = 'training' if is_training else 'validation'

    with torch.set_grad_enabled(is_training):
        model.train() if is_training else model.eval()
        with tqdm(dataloader, unit="batch") as tepoch:
            loss_, f1_score_mc_ = [], []
            for embs, labels in tepoch:
                tepoch.set_description(f"Epoch {e}")
                embs = torch.squeeze(embs).to(device)
                #embs = torch.unsqueeze(embs, (2)).to(device)
                #print(fields)
                labels_ = [torch.squeeze(torch.tensor(labels[key], dtype=float)).to(device) for key in fields]
                #(lbl_ion1, lbl_ion2, lbl_ion3, lbl_ion4, lbl_ion5, lbl_ion6, lbl_ion7, lbl_ion8, lbl_ion9, lbl_ion10, lbl_null) = labels_

                #print(embs.shape)
                predictions = model(embs, mask=None)
                #(prd_ion1, prd_ion2, prd_ion3, prd_ion4, prd_ion5, prd_ion6, prd_ion7, prd_ion8, prd_ion9, prd_ion10, prd_null) = predictions

                def cal_loss(prd, lbl):
                    return criterion_1(torch.squeeze(prd), lbl)

                loss_final = 0
                for prds, lbls in zip(predictions, labels_):
                    loss_final+=cal_loss(prds, lbls)

                m = nn.Sigmoid()
                f1_score_mc = f1_score(torch.round(m(torch.stack(predictions))).cpu().data.numpy(),
                                        torch.stack(labels_).cpu().data.numpy(), average='micro')

                tepoch.set_postfix(loss=loss_final, f1_avg=f1_score_mc)
                if is_training:
                    optimizer.zero_grad()
                    loss_final.backward()
                    optimizer.step()

                loss_.append(loss_final)
                f1_score_mc_.append(f1_score_mc)
    return loss_, f1_score_mc_
            
from tqdm import trange
def train_loop(model, epochs, dataloader_train, dataloader_val,
               optimizer, lr_scheduler, criterion_1, interval):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    net_loss_train, net_f1_train, net_loss_test, net_f1_test = [], [], [], []

    for e in range(epochs):
        with tqdm(dataloader_train, unit="batch") as tepoch:
            lrs = [f'{lr:.6f}' for lr in lr_scheduler.get_lr()]
            #print(f'epoch {e} : lrs : {" ".join(lrs)}')
            loss_, f1_score_mc_ = train_val(model, dataloader_train, optimizer, criterion_1, True, device, 1, interval, e)
            net_loss_train.append((sum(loss_)/len(dataloader_train)).item())   # Save loss
            net_f1_train.append((sum(f1_score_mc_)/len(dataloader_train)).item()) # Save f1

            loss_, f1_score_mc_ = train_val(model, dataloader_val, optimizer, criterion_1, False, device, 1, interval, e)
            net_loss_test.append((sum(loss_)/len(dataloader_val)).item())  # Save loss
            net_f1_test.append((sum(f1_score_mc_)/len(dataloader_val)).item()) # Save f1
            lr_scheduler.step()

    return  net_loss_train, net_f1_train, net_loss_test, net_f1_test

import torch
import warnings
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import torch.utils.data as data
import numpy as np

if __name__=="__main__":
    model_list = ['esm1b_t33_650M_UR50S', 'esm1_t34_670M_UR50D', 'esm1_t34_670M_UR50S',
                  'esm1_t6_43M_UR50S', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D',
                  'esm2_t33_650M_UR50D', 'esm2_t6_8M_UR50D']
    #model_name = 'esm1b_t33_650M_UR50S'
    for model_name in model_list:
        print('\nInitializing %s'%(model_name))
        protein_dataset = ProtEmbDataset(model_name, model_name)

        # Create a validation and training set
        samples_count = len(protein_dataset)
        all_samples_indexes = list(range(samples_count))
        np.random.shuffle(all_samples_indexes)

        val_ratio = 0.2
        val_end = int(samples_count * val_ratio)
        val_indexes = all_samples_indexes[0:val_end]
        train_indexes = all_samples_indexes[val_end:]
        assert len(val_indexes) + len(train_indexes) == samples_count , 'the split is not valid' 

        sampler_train = data.SubsetRandomSampler(train_indexes)
        sampler_val = data.SubsetRandomSampler(val_indexes)

        dataloader_train = DataLoader(protein_dataset, batch_size=1, sampler = sampler_train, num_workers=4)
        dataloader_val = DataLoader(protein_dataset, batch_size=1, sampler = sampler_val, num_workers=4)

        print('Dataset lengths (train/val):', len(dataloader_train), len(dataloader_val))

        warnings.filterwarnings("ignore")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = IonicProtein(dataloader_train.dataset[0][0].shape[1]).to(device)
        print('Device:', device)
        print(model)
        criterion_1 = nn.BCEWithLogitsLoss()
        epochs = 100
        lr = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        lrsched = torch.optim.lr_scheduler.StepLR(optimizer, 10)
        print('\nTraining model %s for % epochs at %s learning rate'%(model_name, epochs, lr))
        net_loss_train, net_f1_train, net_loss_test, net_f1_test = train_loop(model, epochs, dataloader_train, dataloader_val, optimizer, lrsched, criterion_1, 100)
        training_stats = pd.DataFrame()
        training_stats['net_loss_train'] = net_loss_train
        training_stats['net_loss_test'] = net_loss_test
        training_stats['net_f1_train'] = net_f1_train
        training_stats['net_f1_test'] = net_f1_test
        training_stats.to_csv('checkpoints/%s.csv'%(model_name))