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

chain_to_number = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
}
binder_chain = "A"
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
    # path = '/data/cb/bjing/openprot/splits/motifs/1bcf.pdb'
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
    
target = parse_boltz_schema("7v11", data, ccd_lib)
torch.set_float32_matmul_precision("highest")

device = "cuda"
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

binder_mask = batch["entity_id"] == chain_to_number[binder_chain]
length = int(binder_mask.sum())

torch.manual_seed(137)
z = torch.distributions.Gumbel(0, 1).sample((length, 33)).to(device)

invalid_toks = torch.zeros(33).to(device)
invalid_toks[[0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]] = 1

batch["res_type_logits"] = batch["res_type"].clone().detach().to(device).float()

batch["res_type_logits"][binder_mask] = torch.softmax(z - 1e10 * invalid_toks, dim=-1)
if args.motif:
    batch['res_type_logits'][:,motif_mask] = batch["res_type"].clone().detach().to(device).float()[:,motif_mask]

#### NON PROTEIN TARGET ONLY
batch["msa"] = batch["res_type_logits"].unsqueeze(0).to(device)
batch["msa_paired"] = torch.ones(
    batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
).to(device)
batch["deletion_value"] = torch.zeros(
    batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
).to(device)
batch["has_deletion"] = torch.full(
    (batch["res_type"].shape[0], 1, batch["res_type"].shape[1]), False
).to(device)
batch["msa_mask"] = torch.ones(
    batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
).to(device)
batch["profile"] = batch["msa"].float().mean(dim=0).to(device)
batch["deletion_mean"] = torch.zeros(batch["deletion_mean"].shape).to(device)
batch["res_type"] = batch["res_type"].float()

batch["res_type_logits"].requires_grad = True

mask = torch.ones_like(batch["res_type_logits"])
mask[batch["entity_id"] != chain_to_number[binder_chain], :] = 0
chain_mask = (batch["entity_id"] == chain_to_number[binder_chain]).int()
# INTERESTING DIFF BETWEEN THESE TWO
mid_points = torch.linspace(2, 22, 64).to(device)

predict_args = {
    "recycling_steps": 0,
    "sampling_steps": 200,
    "diffusion_samples": 1,
    "write_confidence_summary": True,
    "write_full_pae": True,
    "write_full_pde": True,
}



mask[:,motif_mask] = 0
motif_dmat=torch.from_numpy(motif_dmat).to(device)

def do_iter(batch, opt, pre_run=False):
    batch = update_sequence(
        opt,
        batch,
        mask,
        non_protein_target=True,
        binder_chain=binder_chain,
        device=device,
    )
    print(get_seq(batch))
    total_loss = get_motif_model_loss(
        batch,
        boltz_model,
        motif_dmat=motif_dmat,
        device=device,
        chain_mask=chain_mask,
    )
    
    # total_loss = get_model_loss(
    #     batch,
    #     boltz_model,
    #     device=device,
    #     chain_mask=chain_mask,
    #     length=length,
    #     pre_run=pre_run,
    #     mask_ligand=pre_run,
    #     predict_args=predict_args,
    #     loss_scales={
    #         "con_loss": 1.0,
    #         "i_con_loss": 1.0,
    #         "plddt_loss": 0.1,
    #         "pae_loss": 0.4,
    #         "i_pae_loss": 0.1,
    #         "rg_loss": 0.0,
    #         "helix_loss": -0.2,
    #     },
    #     num_optimizing_binder_pos=int(opt["num_optimizing_binder_pos"]),
    # )

    total_loss.backward()
    print('total_loss', total_loss)
    batch["res_type_logits"].grad[
        batch["entity_id"] != chain_to_number[binder_chain], :
    ] = 0
    batch["res_type_logits"].grad *= mask
    
    batch["res_type_logits"].grad[
        ..., [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    ] = 0
    batch["res_type_logits"].grad = norm_seq_grad(
        batch["res_type_logits"].grad, chain_mask
    )
    with torch.no_grad():
        batch["res_type_logits"] -= opt["lr_rate"] * batch["res_type_logits"].grad
    batch["res_type_logits"].grad = None
    # print(
    #     f"Epoch {i}: lr: {current_lr:.3f}, soft: {opt['soft']:.2f}, hard: {opt['hard']:.2f}, temp: {opt['temp']:.2f}, total loss: {total_loss.item():.2f}, {loss_str}"
    # )


for opt in Annealer(hard=0, e_hard=0, iters=30, lr=0.2):
    do_iter(batch, opt, pre_run=True)

batch["res_type_logits"] = batch["res_type"].clone().detach().requires_grad_(True)

for opt in Annealer(
    soft=0, 
    e_soft=1,
    hard=0,
    e_hard=0,
    e_num_optimizing_binder_pos=8,
    iters=100,
):
    do_iter(batch, opt)

batch["res_type_logits"] = (2.0 * batch["res_type_logits"]).clone().detach().requires_grad_(True)


for opt in Annealer(
    e_temp=0.01,
    hard=0,
    e_hard=0,
    num_optimizing_binder_pos=8,
    e_num_optimizing_binder_pos=12,
    iters=100,
):
    do_iter(batch, opt)


for opt in Annealer(
    temp=0.01,
    e_temp=0.01,
    num_optimizing_binder_pos=12,
    e_num_optimizing_binder_pos=16,
    iters=5,
):
    do_iter(batch, opt)

# ['con_loss:2.02', 'i_con_loss:2.16', 'helix_loss:3.43']
boltz_model.eval()

data["sequences"][0]["protein"]["sequence"] = get_seq(batch)
target = parse_boltz_schema("7v11", data, ccd_lib)
new_batch, new_struct = get_batch(target)
new_batch = {key: value.unsqueeze(0).to(device) for key, value in new_batch.items()}

predict_args["recycling_steps"] = 3

output = run_model(boltz_model, new_batch, predict_args)

new_struct.atoms['coords'] = output['coords'][0,:len(new_struct.atoms)].cpu().numpy()
with open('out.pdb', 'w') as f: f.write(to_pdb(new_struct))
print(output)
