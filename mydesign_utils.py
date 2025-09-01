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


def get_batch(target, max_seqs=4096, keep_record=False):
    target_id = target.record.id
    structure = target.structure

    structure = Structure(
        atoms=structure.atoms,
        bonds=structure.bonds,
        residues=structure.residues,
        chains=structure.chains,
        connections=structure.connections.astype(Connection),
        interfaces=structure.interfaces,
        mask=structure.mask,
    )

    msas = {}
    for chain in target.record.chains:
        msa_id = chain.msa_id
        if msa_id != -1:
            msa = np.load(msa_id)
            msas[chain.chain_id] = MSA(**msa)

    input = Input(structure, msas)

    tokenizer = BoltzTokenizer()
    tokenized = tokenizer.tokenize(input)
    featurizer = BoltzFeaturizer()

    batch = featurizer.process(
        tokenized,
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=max_seqs,
        pad_to_max_seqs=False,
        symmetries={},
        compute_symmetries=False,
        inference_binder=None,
        inference_pocket=None,
    )

    if keep_record:
        batch["record"] = target.record

    return batch, structure


def get_mid_points(pdistogram):
    boundaries = torch.linspace(2, 22.0, 63)
    lower = torch.tensor([1.0])
    upper = torch.tensor([22.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(pdistogram.device)

    return mid_points


def get_con_loss(
    dgram,
    dgram_bins,
    num=None,
    seqsep=None,
    num_pos=float("inf"),
    cutoff=None,
    binary=False,
    mask_1d=None,
    mask_1b=None,
):
    con_loss = _get_con_loss(dgram, dgram_bins, cutoff, binary)
    idx = torch.arange(dgram.shape[1])
    offset = idx[:, None] - idx[None, :]
    # Add mask for position separation > 3
    m = (torch.abs(offset) >= seqsep).to(dgram.device)
    if mask_1d is None:
        mask_1d = torch.ones(m.shape[0])
    if mask_1b is None:
        mask_1b = torch.ones(m.shape[0])

    m = torch.logical_and(m, mask_1b)
    p = min_k(con_loss, num, m).to(dgram.device)
    p = min_k(p, num_pos, mask_1d).to(dgram.device)
    return p


def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=False):
    """dgram to contacts"""
    if cutoff is None:
        cutoff = dgram_bins[-1]
    bins = dgram_bins < cutoff
    px = torch.softmax(dgram, dim=-1)
    px_ = torch.softmax(dgram - 1e7 * (~bins), dim=-1)
    # binary/categorical cross-entropy
    con_loss_cat_ent = -(px_ * torch.log_softmax(dgram, dim=-1)).sum(-1)
    con_loss_bin_ent = -torch.log((bins * px + 1e-8).sum(-1))

    return binary * con_loss_bin_ent + (1 - binary) * con_loss_cat_ent


def mask_loss(x, mask=None, mask_grad=False):
    if mask is None:
        return x.mean()
    else:
        x_masked = (x * mask).sum() / (1e-8 + mask.sum())
        if mask_grad:
            return (x.mean() - x_masked).detach() + x_masked
        else:
            return x_masked


def get_plddt_loss(plddt, mask_1d=None):
    p = 1 - plddt
    return mask_loss(p, mask_1d)


def get_pae_loss(pae, mask_1d=None, mask_1b=None, mask_2d=None):
    pae = pae / 31.0
    L = pae.shape[1]
    if mask_1d is None:
        mask_1d = torch.ones(L).to(pae.device)
    if mask_1b is None:
        mask_1b = torch.ones(L).to(pae.device)
    if mask_2d is None:
        mask_2d = torch.ones((L, L)).to(pae.device)
    mask_2d = mask_2d * mask_1d[:, :, None] * mask_1b[:, None, :]
    return mask_loss(pae, mask_2d)


def _get_helix_loss(
    dgram, dgram_bins, offset=None, mask_2d=None, binary=False, **kwargs
):
    """helix bias loss"""
    x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=binary)
    if offset is None:
        if mask_2d is None:
            return x.diagonal(offset=3).mean()
        else:
            mask_2d = mask_2d.float()
            return (x * mask_2d).diagonal(offset=3, dim1=-2, dim2=-1).sum() / (
                torch.diagonal(mask_2d, offset=3, dim1=-2, dim2=-1).sum() + 1e-8
            )

    else:
        mask = (offset == 3).float()
        if mask_2d is not None:
            mask = mask * mask_2d.float()
        return (x * mask).sum() / (mask.sum() + 1e-8)


def get_ca_coords(sample_atom_coords, batch, binder_chain="A"):
    atom_to_token = batch["atom_to_token"] * (
        batch["entity_id"] == chain_to_number[binder_chain]
    )
    atom_order = torch.cumsum(atom_to_token, dim=1)
    ca_mask = torch.sum((atom_order == 2).to(atom_to_token.dtype), dim=-1)[0]
    ca_coords = sample_atom_coords[:, ca_mask == 1, :]
    return ca_coords


def add_rg_loss(sample_atom_coords, batch, length, binder_chain="A"):
    ca_coords = get_ca_coords(sample_atom_coords, batch, binder_chain)
    center_of_mass = ca_coords.mean(1, keepdim=True)  # keepdim for proper broadcasting
    squared_distances = torch.sum(torch.square(ca_coords - center_of_mass), dim=-1)
    rg = torch.sqrt(squared_distances.mean() + 1e-8)
    rg_th = 2.38 * ca_coords.shape[1] ** 0.365
    loss = torch.nn.functional.elu(rg - rg_th)
    return loss, rg


def min_k(x, k=1, mask=None):
    # Convert mask to boolean if it's not None
    if mask is not None:
        mask = mask.bool()  # Convert to boolean tensor

    # Sort the tensor, replacing masked values with Nan
    y = torch.sort(x if mask is None else torch.where(mask, x, float("nan")))[0]

    # Create a mask for the top k value
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    # Compute the mean of the top k values
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)




def get_motif_model_loss(
    batch,
    boltz_model,
    device="cuda",
    chain_mask=None, 
    motif_dmat=None,
    num_intra_contacts=2,
    intra_chain_cutoff=14,
):
    dict_out, s, z, s_inputs = boltz_model.get_distogram(batch)
    pdist = dict_out["pdistogram"]
    mid_pts = get_mid_points(pdist).to(device)

    motif_dmat_mask = (motif_dmat > 1e-3) & (motif_dmat < 22)
    
    edist = (pdist.softmax(-1) * mid_pts).sum(-1)
    motif_edist_loss = (motif_dmat - edist)**2
    motif_edist_loss = (motif_edist_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()

    motif_mse_loss = (pdist.softmax(-1) * (mid_pts - motif_dmat[...,None])**2).sum(-1)
    motif_mse_loss = (motif_mse_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()

    motif_dmat_idx = (motif_dmat[...,None] - mid_pts).abs().argmin(-1)
    motif_ce_loss = torch.nn.functional.cross_entropy(pdist.permute(0,3,1,2), motif_dmat_idx[None], reduction='none')

    motif_ce_loss = (motif_ce_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()
    
    
    
    print('motif_ce_loss', motif_ce_loss)
    print('motif_edist_loss', motif_edist_loss)
    print('motif_mse_loss', motif_mse_loss)
    # Calculate contact losses
    
    con_loss = get_con_loss(
        pdist,
        mid_pts,
        num=num_intra_contacts,
        seqsep=9,
        cutoff=intra_chain_cutoff,
        binary=False,
        mask_1d=chain_mask,
        mask_1b=chain_mask,
    )
    
    return con_loss + motif_mse_loss
    

def get_model_loss(
    batch,
    boltz_model,
    device="cuda",
    chain_mask=None,  # required tho
    length=100,
    pre_run=False,
    mask_ligand=True,
    distogram_only=True,
    predict_args=None,
    loss_scales=None,
    binder_chain="A",
    increasing_contact_over_itr=False,
    optimize_contact_per_binder_pos=False,
    num_inter_contacts=1,
    num_intra_contacts=2,
    num_optimizing_binder_pos=1,
    inter_chain_cutoff=20.0,
    intra_chain_cutoff=14.0,
    save_trajectory=False,
):
    traj_coords = None
    traj_plddt = None

    # Handle masking first if needed
    if pre_run and mask_ligand:
        batch["token_pad_mask"][batch["entity_id"] != chain_to_number[binder_chain]] = 0
        masked_token_to_rep = torch.ones_like(batch["token_to_rep_atom"])
        masked_token_to_rep[batch["entity_id"] == chain_to_number[binder_chain], :] = 0
        masked_token_to_rep_index = torch.nonzero(
            batch["token_to_rep_atom"] * masked_token_to_rep, as_tuple=True
        )[2]
        batch["atom_pad_mask"][:, masked_token_to_rep_index] = 0

    else:
        batch["token_pad_mask"][:] = 1
        batch["atom_pad_mask"][:] = 1

    # Common arguments for get_distogram_confidence
    confidence_args = {
        "recycling_steps": predict_args["recycling_steps"],
        "num_sampling_steps": predict_args["sampling_steps"],
        "multiplicity_diffusion_train": 1,
        "diffusion_samples": predict_args["diffusion_samples"],
        "run_confidence_sequentially": True,
        "disconnect_feats": False,
        "disconnect_pairformer": False,
    }
    import time

    start = time.time()

    if save_trajectory:
        # Get model output with trajectory info
        dict_out = boltz_model.get_distogram_confidence(batch, **confidence_args)
        traj_coords = dict_out["sample_atom_coords"][0].detach().cpu().numpy()
        traj_plddt = dict_out["plddt"][0].detach().cpu().numpy()
    else:
        # Get model output without trajectory
        if pre_run or distogram_only:
            dict_out, s, z, s_inputs = boltz_model.get_distogram(batch)
        else:
            dict_out = boltz_model.get_distogram_confidence(batch, **confidence_args)

    pdist = dict_out["pdistogram"]
    mid_pts = get_mid_points(pdist).to(device)

    # Calculate contact losses
    con_loss = get_con_loss(
        pdist,
        mid_pts,
        num=num_intra_contacts,
        seqsep=9,
        cutoff=intra_chain_cutoff,
        binary=False,
        mask_1d=chain_mask,
        mask_1b=chain_mask,
    )

    if optimize_contact_per_binder_pos:
        if increasing_contact_over_itr:
            num_optimizing_binder_pos = 0 if pre_run else num_optimizing_binder_pos
            i_con_loss = get_con_loss(
                pdist,
                mid_pts,
                num=num_inter_contacts,
                seqsep=0,
                num_pos=num_optimizing_binder_pos,
                cutoff=inter_chain_cutoff,
                binary=False,
                mask_1d=chain_mask,
                mask_1b=1 - chain_mask,
            )
        else:
            i_con_loss = get_con_loss(
                pdist,
                mid_pts,
                num=num_inter_contacts,
                seqsep=0,
                cutoff=inter_chain_cutoff,
                binary=False,
                mask_1d=chain_mask,
                mask_1b=1 - chain_mask,
            )

    else:

        i_con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=num_inter_contacts,
            seqsep=0,
            cutoff=inter_chain_cutoff,
            binary=False,
            mask_1d=1 - chain_mask,
            mask_1b=chain_mask,
        )

    mask_2d = chain_mask[:, :, None] * chain_mask[:, None, :]
    helix_loss = _get_helix_loss(
        pdist, mid_pts, offset=None, mask_2d=mask_2d, binary=True
    )

    if pre_run and mask_ligand:
        losses = {"con_loss": con_loss, "helix_loss": helix_loss}
    else:
        losses = {
            "con_loss": con_loss,
            "i_con_loss": i_con_loss,
            "helix_loss": helix_loss,
        }

    if not pre_run and not distogram_only:
        plddt_loss = get_plddt_loss(dict_out["plddt"], mask_1d=chain_mask)
        pae = (dict_out["pae"] + dict_out["pae"].transpose(-2, -1)) / 2
        i_pae_loss = get_pae_loss(pae, mask_1d=1 - chain_mask, mask_1b=chain_mask)
        pae_loss = get_pae_loss(pae, mask_1d=chain_mask, mask_1b=chain_mask)
        rg_loss, rg = add_rg_loss(
            dict_out["sample_atom_coords"],
            batch,
            length,
            binder_chain=binder_chain,
        )

        losses.update(
            {
                "plddt_loss": plddt_loss,
                "i_pae_loss": i_pae_loss,
                "pae_loss": pae_loss,
                "rg_loss": rg_loss,
            }
        )

    # bins = mid_pts < 8.0
    # px = torch.sum(
    #     torch.softmax(dict_out["pdistogram"], dim=-1)[:, :, :, bins], dim=-1
    # )

    if loss_scales is None:
        loss_scales = {
            "con_loss": 1.0,
            "i_con_loss": 1.0,
            "helix_loss": random.uniform(-0.4, 0.0),
            "plddt_loss": 0.1,
            "pae_loss": 0.4,
            "i_pae_loss": 0.1,
            "rg_loss": 0.0,
        }

    # Calculate total loss and print individual losses
    total_loss = sum(loss * loss_scales[name] for name, loss in losses.items())
    loss_str = [f"{k}:{v.item():.2f}" for k, v in losses.items()]
    # loss_history.append(total_loss.item())
    # i_con_loss_history.append(i_con_loss.item())
    # con_loss_history.append(con_loss.item())
    # # distogram_history.append(torch.softmax(dict_out['pdistogram'], dim=-1)[0].detach().cpu().numpy())
    # distogram_history.append(px[0].detach().cpu().numpy())
    # sequence_history.append(
    #     batch["res_type"][0, :, 2:22].detach().cpu().numpy()
    # )

    print(loss_str)
    return total_loss


def update_sequence(
    opt,
    batch,
    mask,
    alpha=2.0,
    non_protein_target=False,
    binder_chain="A",
    device="cuda",
):

    batch["logits"] = alpha * batch["res_type_logits"]
    X = batch["logits"] - torch.sum(
        torch.eye(batch["logits"].shape[-1])[
            [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        ],
        dim=0,
    ).to(device) * (1e10)
    batch["soft"] = torch.softmax(X / opt["temp"], dim=-1)  # probs
    batch["hard"] = torch.zeros_like(batch["soft"]).scatter_(
        -1, batch["soft"].max(dim=-1, keepdim=True)[1], 1.0
    )  # one hot
    batch["hard"] = (batch["hard"] - batch["soft"]).detach() + batch[
        "soft"
    ]  # carries same grad
    batch["pseudo"] = (
        opt["soft"] * batch["soft"] + (1 - opt["soft"]) * batch["res_type_logits"]
    )  # interp between probs and logits
    batch["pseudo"] = (
        opt["hard"] * batch["hard"] + (1 - opt["hard"]) * batch["pseudo"]
    )  # interp between on hot and the above

    batch["res_type"] = batch["pseudo"] * mask + batch["res_type_logits"] * (1 - mask)

    if non_protein_target:
        batch["msa"] = batch["res_type"].unsqueeze(0).to(device).detach()
        batch["profile"] = batch["msa"].float().mean(dim=0).to(device).detach()
    else:
        batch["msa"][:, 0, :, :] = batch["res_type"].to(device).detach()
        batch["profile"][batch["entity_id"] == chain_to_number[binder_chain], :] = (
            batch["msa"][
                :,
                0,
                (batch["entity_id"] == chain_to_number[binder_chain])[0],
                :,
            ]
            .float()
            .mean(dim=1)
            .to(device)
            .detach()
        )

    return batch


def norm_seq_grad(grad, chain_mask):
    chain_mask = chain_mask.bool()
    masked_grad = grad[:, chain_mask.squeeze(0), :]
    eff_L = (masked_grad.pow(2).sum(-1, keepdim=True) > 0).sum(-2, keepdim=True)
    gn = masked_grad.norm(dim=(-1, -2), keepdim=True)
    return grad * torch.sqrt(torch.tensor(eff_L)) / (gn + 1e-7)


def run_model(boltz_model, batch, predict_args):
    boltz_model.predict_args = predict_args
    return boltz_model.predict_step(batch, batch_idx=0, dataloader_idx=0)


class Annealer:
    def __init__(
        self,
        soft=1,
        e_soft=1,
        temp=1,
        e_temp=1,
        hard=1,
        e_hard=1,
        step=1,
        e_step=1,
        num_optimizing_binder_pos=1,
        e_num_optimizing_binder_pos=1,
        iters=100,
        lr=0.1,
    ):
        m = {
            "soft": [soft, e_soft],
            "temp": [temp, e_temp],
            "hard": [hard, e_hard],
            "step": [step, e_step],
            "num_optimizing_binder_pos": [
                num_optimizing_binder_pos,
                e_num_optimizing_binder_pos,
            ],
        }
        self.m = {k: [s, (s if e is None else e)] for k, (s, e) in m.items()}
        self.lr = lr
        self.iters = iters

    def __iter__(self):
        import tqdm

        for i in tqdm.trange(self.iters):
            opt = {}
            for k, (s, e) in self.m.items():
                if k == "temp":
                    opt[k] = e + (s - e) * (1 - (i) / self.iters) ** 2
                else:
                    v = s + (e - s) * ((i) / self.iters)
                    if k == "step":
                        step = v
                    opt[k] = v

            lr_scale = step * ((1 - opt["soft"]) + (opt["soft"] * opt["temp"]))
            num_optimizing_binder_pos = int(opt["num_optimizing_binder_pos"])

            opt["lr_rate"] = self.lr * lr_scale
            yield opt
