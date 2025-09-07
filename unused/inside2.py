from boltz.data.parse.schema import parse_boltz_schema
import pickle, os, yaml

import os
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional
import copy
import random
from boltz.data import const
from boltz.data.types import MSA, Connection, Input, Structure, Interface
from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
import yaml
import shutil
from Bio.PDB import PDBParser, MMCIFParser
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import csv
import gc
import json
import logging

from mydesign_utils import *
def get_seq(batch):
    alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
    seq = "".join(
        [
            alphabet[i]
            for i in torch.argmax(
                batch["res_type"][batch["entity_id"] == chain_to_number[binder_chain], :],
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        ]
    )
    return seq

with open(os.path.expanduser("~/.boltz/ccd.pkl"), "rb") as f:
    ccd_lib = pickle.load(f)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--len', type=int, default=100)
parser.add_argument(
    '--smiles', 
    type=str,
    default=None
)
parser.add_argument('--motif', type=str, default=None)
args = parser.parse_args()

if args.motif is not None:
    from motif_utils import load_motif_spec, sample_motif_mask
    spec = load_motif_spec(args.motif)
    masks = sample_motif_mask(spec)
    motif_mask = masks['sequence']
    motif_idx = masks['group']
    import protein
    with open(args.motif) as f:
        prot = protein.from_pdb_string(f.read())
    
    ca_pos = np.zeros((len(motif_mask), 3))
    ca_pos[motif_mask] = prot.atom_positions[:,2] # use CB positions - check glycine!
    motif_dmat = np.square(ca_pos[None] - ca_pos[:,None]).sum(-1)**0.5
    motif_dmat[~motif_mask,:] = motif_dmat[:,~motif_mask] = 0

    length = len(motif_mask)
else:
    length = args.len
data = {
    "version": 1,
    "sequences": [
        {
            "protein": {
                "id": ["A"],
                "sequence": "X"*length,
                "msa": "empty",
            }
        },
    ],
}
import residue_constants
if args.motif:
    seq = ['X']*length
    for idx, aatype in zip(motif_mask.nonzero()[0], prot.aatype):
        seq[idx] = residue_constants.restypes[aatype]
    data['sequences'][0]['protein']['sequence'] = ''.join(seq)
    print(data['sequences'][0]['protein']['sequence'])
if args.smiles:
    data['sequences'].append({
        "ligand": {
            "id": ["B"],
            "smiles": args.smiles,
        }
    })
device = 'cuda'
target = parse_boltz_schema("7v11", data, ccd_lib)
torch.set_float32_matmul_precision("highest")

batch, structure = get_batch(target)
batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
diffusion_params = BoltzDiffusionParams()
diffusion_params.step_scale = 1.638  # Default value
boltz_model = Boltz1.load_from_checkpoint(
    "~/.boltz/boltz1_conf.ckpt",
    strict=False,
    predict_args={
        "recycling_steps": 0,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": True,
    },
    map_location=device,
    diffusion_process_args=asdict(diffusion_params),
    ema=False,
    structure_prediction_training=True,
    no_msa=False,
    no_atom_encoder=False,
).eval()


def reward(seq):
    
    batch['res_type'] = torch.nn.functional.pad(seq, (2, 11)).unsqueeze(0)
    batch["msa"] = batch["res_type"].unsqueeze(0).to(device).detach()
    batch["profile"] = batch["msa"].float().mean(dim=0).to(device).detach()
    total_loss = get_motif_model_loss(
        batch,
        boltz_model,
        motif_dmat=torch.from_numpy(motif_dmat).to(device),
        device=device,
        chain_mask=torch.ones_like(batch['res_type'][...,0])
    )
    return -total_loss

def get_masked_proposal(seq, mask, reward):
    mask_tok = torch.zeros(20, device=device)
    masked = torch.where(mask[...,None], seq, mask_tok).requires_grad_(True)
    rr = reward(masked)
    rr.backward()
    return rr, torch.where(mask[...,None], 0.0, masked.grad)

seq = torch.randint(0, 20, (length,), device=device)
seq[:] = 0
seq = torch.nn.functional.one_hot(seq, num_classes=20).float()

K = 1
for i in range(200):
    mask = torch.ones(len(seq), device=device, dtype=bool)
    idx = torch.randint(0, len(seq), (K,))
    mask[idx] = False
    curr_r = reward(seq)
    masked_r, prop_r = get_masked_proposal(seq, mask, reward)
    
    dist = torch.distributions.Categorical(logits=prop_r[idx]*10)
    samp = dist.sample()
    prop = seq.clone()
    prop[idx] = torch.nn.functional.one_hot(samp, num_classes=20).float()

    new_r = reward(prop)

    prob_forward = dist.log_prob(samp).sum()
    prob_backward = dist.log_prob(seq[idx].argmax(-1)).sum()

    mh = new_r + prob_backward - curr_r - prob_forward
    print('MH ratio', mh)
    if new_r > curr_r: #  np.random.rand() < mh.exp(): # accept
        seq = prop
        
    print(curr_r, new_r)