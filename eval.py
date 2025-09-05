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
    
motif = "3ixt"
design_dir =f"./out/onemotif_twostates/{motif}/design3/"
with open(os.path.join(design_dir,f"{motif}_spec.pkl"), "rb") as f:
    motif_mask = pickle.load(f)["motif_mask"]
print("mRMSD: state 0", motifRMSD(f"/data/cb/mihirb14/projects/BoltzDesign1/motifs/{motif}.pdb",os.path.join(design_dir,"state0.pdb"),motif_mask))
print("mRMSD: state 1",motifRMSD(f"/data/cb/mihirb14/projects/BoltzDesign1/motifs/{motif}.pdb",os.path.join(design_dir,"state1.pdb"),motif_mask))