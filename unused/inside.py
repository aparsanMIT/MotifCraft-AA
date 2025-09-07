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

device = 'cuda'
randvec = torch.randn(100, 20, device=device)
def reward(seq):
    return (randvec * seq).sum()

mask_tok = torch.zeros(20, device=device)
def get_masked_proposal(seq, mask, reward):
    masked = torch.where(mask[...,None], seq, mask_tok).requires_grad_(True)
    rr = reward(masked)
    rr.backward()
    return rr, torch.where(mask[...,None], 0.0, masked.grad)

seq = torch.randint(0, 20, (100,), device=device)

seq = torch.nn.functional.one_hot(seq, num_classes=20).float()

K = 1
for i in range(200):
    mask = torch.ones(len(seq), device=device, dtype=bool)
    idx = torch.randint(0, len(seq), (K,))
    mask[idx] = False
    curr_r = reward(seq)
    masked_r, prop_r = get_masked_proposal(seq, mask, reward)
    dist = torch.distributions.Categorical(logits=prop_r[idx])
    samp = dist.sample()
    prop = seq.clone()
    prop[idx] = torch.nn.functional.one_hot(samp, num_classes=20).float()

    new_r = reward(prop)

    prob_forward = dist.log_prob(samp).sum()
    prob_backward = dist.log_prob(seq[idx].argmax(-1)).sum()

    mh = new_r + prob_backward - curr_r - prob_forward
    
    if np.random.rand() < mh.exp(): # accept
        seq = prop
        
    print(curr_r)