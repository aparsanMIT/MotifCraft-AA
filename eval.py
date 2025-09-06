import pickle
import numpy as np
import torch
import os
import glob
from utils import protein
from utils.geometry import compute_rmsd

def motifRMSD(motif_pdb, design_pdb, motif_mask):
    with open(motif_pdb) as f:
        motif = protein.from_pdb_string(f.read())
    with open(design_pdb) as f:
        design = protein.from_pdb_string(f.read())

    true_motif_ca = torch.from_numpy(motif.atom_positions[:, 1])
    design_motif_ca = torch.from_numpy(design.atom_positions[:, 1][motif_mask])
    
    return compute_rmsd(design_motif_ca, true_motif_ca)


if __name__ == "__main__":

    motif = "3ixt"
    design_dir = f"./out/onemotif_twostates/{motif}/design1/"

    with open(os.path.join(design_dir, f"{motif}_spec.pkl"), "rb") as f:
        motif_mask = pickle.load(f)["motif_mask"]

    motif_pdb = f"/data/cb/mihirb14/projects/BoltzDesign1/motifs/{motif}.pdb"

    pdb_files = sorted(glob.glob(os.path.join(design_dir, "state*_sample*.pdb")))

    from collections import defaultdict
    state_groups = defaultdict(list)
    for pdb in pdb_files:
        base = os.path.basename(pdb)
        state = base.split("_")[0]
        state_groups[state].append(pdb)

    for state, files in state_groups.items():
        print(f"\n{state}:")
        for fpath in sorted(files):
            rmsd = motifRMSD(motif_pdb, fpath, motif_mask)
            print(f"  {os.path.basename(fpath)} → RMSD {rmsd:.3f}")
