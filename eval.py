import pickle
from utils import protein
import numpy as np
from utils.geometry import compute_rmsd
import torch
import os

def motifRMSD(motif_pdb, design_pdb, motif_mask):
    with open(motif_pdb) as f:
        motif = protein.from_pdb_string(f.read())
    with open(design_pdb) as f:
        design = protein.from_pdb_string(f.read())

    true_motif_ca = torch.from_numpy(motif.atom_positions[:,1])
    design_motif_ca = torch.from_numpy(design.atom_positions[:,1][motif_mask])
    
    return compute_rmsd(design_motif_ca, true_motif_ca)
    
    
design_dir = "./out/onemotif_twostates/3ixt/design1/"
with open(os.path.join(design_dir,"3ixt_spec.pkl"), "rb") as f:
    motif_mask = pickle.load(f)["motif_mask"]
print(motifRMSD("/data/cb/mihirb14/projects/BoltzDesign1/motifs/3ixt.pdb",os.path.join(design_dir,"state1.pdb"),motif_mask))