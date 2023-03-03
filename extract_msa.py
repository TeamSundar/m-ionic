from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import itertools
import os
import string
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from Bio import SeqIO
import biotite.structure as bs
from biotite.structure.io.pdbx import PDBxFile, get_structure
from biotite.database import rcsb
import pandas as pd
import multiprocessing as mp
from scipy.spatial.distance import squareform, pdist, cdist

#print("Number of processors: ", mp.cpu_count())
import esm

torch.set_grad_enabled(False)


MAX_TOKEN_NUM = 2 ** 14
MIN_MSA_ROW_NUM = 16
MAX_MSA_COL_NUM = 1024

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    print (len(record.seq))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa

    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

def extract_msa_transformer_features(msa_seq_label, msa_seq_str, msa_seq_token, device=torch.device("cpu")):
    msa_seq_token = msa_seq_token.to(device)
    msa_row, msa_col = msa_seq_token.shape[1], msa_seq_token.shape[2]
    #print(f"{msa_seq_label[0][0]}, msa_row: {msa_row}, msa_col: {msa_col}")

    if msa_col > MAX_MSA_COL_NUM:
        print(f"msa col num should less than {MAX_MSA_COL_NUM}. This program force the msa col to under {MAX_MSA_COL_NUM}")
    msa_seq_token = msa_seq_token[:, :, :MAX_MSA_COL_NUM]

    ### keys: ['logits', 'representations', 'col_attentions', 'row_attentions', 'contacts']
    msa_transformer_outputs = msa_transformer(
        msa_seq_token, repr_layers=[12],
        need_head_weights=True, return_contacts=True)
    msa_row_attentions = msa_transformer_outputs['row_attentions']
    msa_representations = msa_transformer_outputs['representations'][12]
    msa_query_representation = msa_representations[:, 0, 1:, :]  # remove start token
    msa_row_attentions = msa_row_attentions[..., 1:, 1:]  # remove start token

    return msa_query_representation


mfile = os.listdir('/home/ashenoy/workspace/active/LigandPredict/data/MI/MSA/')

msas = {
    name: read_msa(f"/home/ashenoy/workspace/active/LigandPredict/data/MI/MSA/{name}")
    for name in mfile
}

if torch.cuda.is_available():
    device = torch.device("cuda:0")

folder_for_outputs = '/home/ashenoy/workspace/active/LigandPredict/data/MI/msa_embeddings'

msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval().cuda()
msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()

for name, inputs in msas.items():
    inputs = greedy_select(inputs, num_seqs=128)
    msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
    msa_query_representation = extract_msa_transformer_features(msa_transformer_batch_labels,msa_transformer_batch_strs, msa_transformer_batch_tokens, device=device)
    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
    print (msa_transformer_batch_labels[0][0], msa_query_representation.shape)
    torch.save(msa_query_representation, folder_for_outputs+'/{}.pt'.format(msa_transformer_batch_labels[0][0]))


