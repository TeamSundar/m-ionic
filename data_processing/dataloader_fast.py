#import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch, os, collections
import pandas as pd
import pickle5 as pickle
from tqdm import tqdm
import numpy as np

# Load guide csv file
df = pd.read_csv('data/LigID_pdbchain_partition.csv')

# Load data pkl file
with open('data/reduced.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# Add ion info in dict
print('\nAdding ion info to dict...')
for key in tqdm(data_dict.keys()):
    data_dict[key].append(df[df['pdb_chain']==key]['LigID'].item())

with open('data/reduced.pkl', 'rb') as f:
    data = pickle.load(f)

data_path = '/media/dell4/a87d4a0b-5641-40d3-8d10-416948fc2996/ION_DATA/'

# model_list = ['esm1b_t33_650M_UR50S', 'esm1_t34_670M_UR50D', 'esm1_t34_670M_UR50S',
#               'esm1_t6_43M_UR50S', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D',
#               'esm2_t33_650M_UR50D', 'esm2_t6_8M_UR50D']

model_list = ['esm1_t34_670M_UR50S',
              'esm1_t6_43M_UR50S', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D',
              'esm2_t33_650M_UR50D', 'esm2_t6_8M_UR50D']

for model_name in model_list:
    emb_batch, batch_id, file_no = [], 0, 0
    # Make new folder for batch data
    try: 
        os.mkdir(data_path+model_name+'_batch128') 
    except OSError as error: 
        print(error)  

    fields = list(collections.Counter(df['LigID']).keys())
    fields.append('null')
    label_dict = dict.fromkeys(fields, [])
    #label_dict = {k: [] for k in fields}
    label_list = []
    batch_pdbids = []

    import pickle

    print('Making batches for', model_name)
    for pdbid in tqdm(data.keys()):
        if file_no<128:
            emb = torch.load(data_path+model_name+'/'+pdbid+'.pt')  # Import embeddings
            key = list(emb['representations'].keys())
            emb_batch.append(emb['representations'][key[0]])
            label_list.append([int(a) for a in data_dict[pdbid][1]])
            batch_pdbids.append(pdbid)
            file_no+=1
            batch_id+=1
        else:
            # Make data label batch
            emb_batch_ = torch.cat(emb_batch)
            label_dict = dict.fromkeys(fields, np.array([0]*len(emb_batch_)))
            flattened_label_list = np.array([item for sublist in label_list for item in sublist])
            label_dict[data_dict[pdbid][2]] = flattened_label_list  # add label list ot corresponding ion
            null_label = np.ones(len(flattened_label_list))    # label for null datapoints
            null_label[np.where(flattened_label_list==1)[0]]=0
            label_dict['null'] = null_label

            #print(label_dict)

            batch_dict = dict.fromkeys(['pdbids', 'embs', 'labels'])

            batch_dict['pdbids'] = batch_pdbids
            batch_dict['embs'] = emb_batch_
            batch_dict['labels'] = label_dict

            # Save batch to file as pkl file
            #print('Saving batch to file')
            with open(data_path+model_name+'_batch128/'+ str(batch_id)+'.pickle', 'wb') as handle:
                pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            file_no=0
            batch_pdbids = []
            label_list = []
            emb_batch = []
            #label_dict = dict.fromkeys(fields, [0]*len(sample))
            batch_dict = dict.fromkeys(['pdbids', 'embs', 'labels'])