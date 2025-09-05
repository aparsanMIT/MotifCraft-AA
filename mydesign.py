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
import residue_constants

import os

with open(os.path.expanduser(os.path.join(os.environ['HOME'],".boltz/ccd.pkl")), "rb") as f:
    ccd_lib = pickle.load(f)

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--len', type=int, default=100)
# parser.add_argument('--smiles', type=str,default=None)
parser.add_argument('--num_designs',type= int, default = 1)
parser.add_argument('--motif', type=str, default="3ixt")
parser.add_argument('-o','--outpath', type=str, default = "./out/")
args = parser.parse_args()

#############

torch.set_float32_matmul_precision("highest")
device = "cuda"

#torch.autograd.set_detect_anomaly(True)

def get_motif(path):
    from motif_utils import load_motif_spec, sample_motif_mask
    spec = load_motif_spec(path)
    masks = sample_motif_mask(spec)
    motif_mask = masks['sequence']
    motif_idx = masks['group']
    import protein
    with open(path) as f:
        prot = protein.from_pdb_string(f.read())
    
    ca_pos = np.zeros((len(motif_mask), 3))
    ca_pos[motif_mask] = prot.atom_positions[:,2] # use CB positions - check glycine!

    seq = ['X']*len(motif_mask)
    for idx, aatype in zip(motif_mask.nonzero()[0], prot.aatype):
        seq[idx] = residue_constants.restypes[aatype]
    return {
        'motif_mask': motif_mask,
        'ca_pos': ca_pos,
        'motif_seq': ''.join(seq)
    }
        


def get_batch_with_ligand(seq, ligand=None, device='cuda'):
    data = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": seq,
                    "msa": "empty",
                }
            },
        ],
    }
    if ligand is not None:
        data['sequences'].append({
            "ligand": {
                "id": ["B"],
                "smiles": ligand,
            }
        })
    target = parse_boltz_schema(None, data, ccd_lib)
    batch, structure = get_batch(target)
    batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
    # batch["msa"] = batch["res_type_logits"].unsqueeze(0).to(device)
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

    return batch, structure
    
class MultistateDesigner:
    def __init__(self, num_states):
        self.motifs = [None]*num_states
        self.ligands = [None]*num_states
        self.anti_motifs = [None]*num_states
        
    def add_motif(self, motif, state):
        self.motifs[state] = motif

    def add_anti_motif(self, motif, state):
        self.anti_motifs[state] = motif
        
    def add_ligand(self, ligand, state):
        self.ligands[state] = ligand

    def initialize(self, length, device='cuda'):
        self.batches = []
        for i, lig in enumerate(self.ligands):
            self.batches.append(get_batch_with_ligand('X'*length, lig, device)[0])
        
        z = torch.distributions.Gumbel(0, 1).sample((length, 33)).to(device)
        z[...,:2] = z[...,22:] = -np.inf
        self.logits = z.softmax(-1)
        
        alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
        self.fixed_mask = torch.zeros(length, dtype=bool, device=device)
        self.fixed_aa = torch.zeros_like(self.logits)
        self.dmats = [None]*len(self.motifs)
        self.anti_dmats = [None]*len(self.anti_motifs)
        
        for i, motif in enumerate(self.motifs):
            if motif is not None:
                motif_mask = torch.from_numpy(motif['motif_mask']).to(device)
                self.fixed_mask |= motif_mask
                motif_seq = [alphabet.index(c) for c in motif['motif_seq']]                
                motif_seq = torch.nn.functional.one_hot(torch.tensor(motif_seq), num_classes=22)
                self.fixed_aa[motif_mask,:22] = motif_seq.to(device)[motif_mask].float()

                ca_pos = torch.from_numpy(motif['ca_pos']).to(device)
                dmat = torch.square(ca_pos[None] - ca_pos[:,None]).sum(-1)**0.5
                dmat[~motif_mask,:] = dmat[:,~motif_mask] = 0
                self.dmats[i] = dmat

        for i, anti_motif in enumerate(self.anti_motifs):
            if anti_motif is not None:
                motif_mask = torch.from_numpy(anti_motif['motif_mask']).to(device)
                ca_pos = torch.from_numpy(anti_motif['ca_pos']).to(device)
                dmat = torch.square(ca_pos[None] - ca_pos[:,None]).sum(-1)**0.5
                dmat[~motif_mask,:] = dmat[:,~motif_mask] = 0
                self.anti_dmats[i] = dmat

        self.logits = torch.where(self.fixed_mask[...,None], self.fixed_aa, self.logits)

    def get_seq(self):
        alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
        return ''.join([alphabet[i.item()] for i in self.logits.argmax(-1)])

    def get_final_structs(self, boltz_model):
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 5,
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": True,
        }
        out = []
        for i, ligand in enumerate(self.ligands):
            new_batch, new_struct = get_batch_with_ligand(self.get_seq(), ligand)
            
            mid_points = torch.linspace(2, 22, 64).to(device)
            predict_args["recycling_steps"] = 3
            
            output = run_model(boltz_model, new_batch, predict_args)
            breakpoint()
            new_struct.atoms['coords'] = output['coords'][0,:len(new_struct.atoms)].cpu().numpy()
            out.append((output, new_struct))
        return out

        
    def get_restype_from_logits(self, res_type_logits, opt, alpha=2.0):
        device = res_type_logits.device
        logits = alpha * res_type_logits
        X = logits - torch.sum(
            torch.eye(logits.shape[-1])[
                [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            ],
            dim=0,
        ).to(device) * (1e10)
        soft = torch.softmax(X / opt["temp"], dim=-1)  # probs
        hard = torch.zeros_like(soft).scatter_(
            -1, soft.max(dim=-1, keepdim=True)[1], 1.0
        )  # one hot
        hard = (hard - soft).detach() + soft # carries same grad
        pseudo = (
            opt["soft"] * soft + (1 - opt["soft"]) * res_type_logits
        )  # interp between probs and logits
        pseudo = (
            opt["hard"] * hard + (1 - opt["hard"]) * pseudo
        )  # interp between on hot and the above
    
        return pseudo

    def get_motif_loss(self, pdist, motif_dmat):
        
        mid_pts = get_mid_points(pdist).to(device)

        pdist = pdist[:,:len(motif_dmat),:len(motif_dmat)]
        motif_dmat_mask = (motif_dmat > 1e-3) & (motif_dmat < 22)
        
        # edist = (pdist.softmax(-1) * mid_pts).sum(-1)
        # motif_edist_loss = (motif_dmat - edist)**2
        # motif_edist_loss = (motif_edist_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()
    
        motif_mse_loss = (pdist.softmax(-1) * (mid_pts - motif_dmat[...,None])**2).sum(-1)
        motif_mse_loss = (motif_mse_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()
    
        # motif_dmat_idx = (motif_dmat[...,None] - mid_pts).abs().argmin(-1)
        # motif_ce_loss = torch.nn.functional.cross_entropy(pdist.permute(0,3,1,2), motif_dmat_idx[None], reduction='none')
    
        # motif_ce_loss = (motif_ce_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()
    
        return motif_mse_loss
    
    def get_anti_motif_loss(self, pdist, motif_dmat):
        
        mid_pts = get_mid_points(pdist).to(device)

        pdist = pdist[:,:len(motif_dmat),:len(motif_dmat)]
        motif_dmat_mask = (motif_dmat > 1e-3) & (motif_dmat < 22)
    
        motif_mse_loss = (pdist.softmax(-1) * (mid_pts - motif_dmat[...,None])**2).sum(-1)
        motif_mse_loss = (motif_mse_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()
    
        return -0.5*motif_mse_loss
        
    def get_i_contact_loss(self, pdist, opt, chain_mask):
        mid_pts = get_mid_points(pdist).to(device)
        #num_optimizing_binder_pos = 0 if pre_run else num_optimizing_binder_pos
        i_con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=2,
            seqsep=0,
            num_pos=int(opt["num_optimizing_binder_pos"]),
            cutoff=20.,
            binary=False,
            mask_1d=chain_mask,
            mask_1b=1 - chain_mask,
        )
        return i_con_loss


    def get_contact_loss(self, pdist, opt, chain_mask):
        mid_pts = get_mid_points(pdist).to(device)
        #num_optimizing_binder_pos = 0 if pre_run else num_optimizing_binder_pos
        con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=1,
            seqsep=9,
            cutoff=14.,
            binary=False,
            mask_1d=chain_mask,
            mask_1b=chain_mask,
        )
        return con_loss
            
    def get_loss(self, restype, boltz_model, opt):
        loss = 0
        loss_dict = {}
        for i, (
            dmat,
            anti_dmat,
            ligand,
            batch,
        ) in enumerate(zip(
            self.dmats,
            self.anti_dmats,
            self.ligands,
            self.batches
        )):
            batch['res_type'] = torch.cat([
                restype[None],
                batch['res_type'][:,len(restype):].detach()
            ], 1)
            batch["msa"] = batch["res_type"].unsqueeze(0).detach()
            batch["profile"] = batch["msa"].float().mean(dim=0).detach()

            dict_out = boltz_model.get_distogram(batch)[0]
            if dmat is not None:
                motif_loss = self.get_motif_loss( dict_out["pdistogram"], dmat)
                loss_dict[f'state{i}_motif'] = motif_loss
                loss = loss + motif_loss 

            if anti_dmat is not None:
                anti_motif_loss = self.get_anti_motif_loss( dict_out["pdistogram"], anti_dmat)
                loss_dict[f'state{i}_anti_motif'] = anti_motif_loss
                loss = loss + anti_motif_loss
            
            chain_mask = batch['mol_type'] == 0
            if ligand is not None:
                i_contact_loss = self.get_i_contact_loss(
                    dict_out["pdistogram"], opt, chain_mask.float()
                )
                loss_dict[f'state{i}_i_contact'] = i_contact_loss
                loss = loss + i_contact_loss

            contact_loss = self.get_contact_loss(
                dict_out["pdistogram"], opt, chain_mask.float()
            )
            loss_dict[f'state{i}_contact'] = contact_loss
            loss = loss + contact_loss
        print(loss_dict)
        print(self.get_seq())
        return loss
        
            
    def do_iter(self, boltz_model, opt, pre_run=False):

        self.logits.requires_grad = True
        restype = self.get_restype_from_logits(self.logits, opt)
        restype = torch.where(self.fixed_mask[...,None].clone(), self.fixed_aa.clone(), restype.clone())
       
        loss = self.get_loss(restype, boltz_model, opt)
        # loss = restype.sum()
    
        loss.backward()
        print('total_loss', loss)
        
        
        with torch.no_grad():
            self.logits.grad[self.fixed_mask] = 0
            self.logits.grad[
                ..., [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            ] = 0
            self.logits.grad = norm_seq_grad(
                self.logits.grad[None], 
                torch.ones_like(self.logits[:,0])
            )[0]
        
            self.logits -= opt["lr_rate"] * self.logits.grad
        self.logits.grad = None
        
    def optimize(self, boltz_model):
        for opt in Annealer(hard=0, e_hard=0, iters=30, lr=0.2):
            self.do_iter(boltz_model, opt, pre_run=True)

        
        with torch.no_grad():
            self.logits = self.get_restype_from_logits(self.logits, opt)
        
        
        for opt in Annealer(
            soft=0, 
            e_soft=1,
            hard=0,
            e_hard=0,
            e_num_optimizing_binder_pos=8,
            iters=100,
        ):
            self.do_iter(boltz_model, opt)

        with torch.no_grad():
            self.logits = 2 * self.logits
        
        for opt in Annealer(
            e_temp=0.01,
            hard=0,
            e_hard=0,
            num_optimizing_binder_pos=8,
            e_num_optimizing_binder_pos=12,
            iters=100,
        ):
            self.do_iter(boltz_model, opt)
        
        
        for opt in Annealer(
            temp=0.01,
            e_temp=0.01,
            num_optimizing_binder_pos=12,
            e_num_optimizing_binder_pos=16,
            iters=10,
        ):
            self.do_iter(boltz_model, opt)

predict_args={
    "recycling_steps": 0,
    "sampling_steps": 200,
    "diffusion_samples": 1,
    "write_confidence_summary": True,
    "write_full_pae": True,
    "write_full_pde": True,
}
diffusion_params = BoltzDiffusionParams()
diffusion_params.step_scale = 1.638  # Default value
boltz_model = Boltz1.load_from_checkpoint(
    os.path.join(os.environ['HOME'],".boltz/boltz1_conf.ckpt"),
    strict=False,
    predict_args=predict_args,
    map_location=device,
    diffusion_process_args=asdict(diffusion_params),
    ema=False,
    structure_prediction_training=True,
    no_msa=False,
    no_atom_encoder=False,
).eval().requires_grad_(False)

out_dir = os.path.join(args.outpath, args.motif)
os.makedirs(out_dir, exist_ok=True)

motif = get_motif(f'motifs/{args.motif}.pdb')

with open(os.path.join(out_dir,f"{args.motif}.pkl"), "wb") as f:
    pickle.dump(motif, f)

for trial in range(args.num_designs):
    
    designer = MultistateDesigner(num_states=2)
    designer.add_motif(motif, state=0)
    designer.add_anti_motif(motif, state=1)
    designer.add_ligand('Fc1c(Cl)ccc(n2cnnn2)c1c1c[n+]([O-])c(cc1)C(CC1CC1)n1cc(cn1)c1ccc(N)nc1C', state=1)
    designer.initialize(length=len(motif['motif_mask']))

    # breakpoint()

    designer.optimize(boltz_model)

    structs = designer.get_final_structs(boltz_model)
    
    design_dir = os.path.join(out_dir, f"design{trial}")
    os.makedirs(design_dir, exist_ok=True)
    
    for i, (out_dict, struct) in enumerate(structs):
        pdb_path = os.path.join(design_dir, f"state{i}.pdb")
        cif_path = os.path.join(design_dir, f"state{i}.cif")
        pkl_path = os.path.join(design_dir, f"state{i}.pkl")
        
        with open(pdb_path, 'w') as f:
            f.write(to_pdb(struct))
        with open(cif_path, 'w') as f:
            f.write(to_mmcif(struct))
        with open(pkl_path, "wb") as f:
            pickle.dump(out_dict, f)