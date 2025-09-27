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
    import io
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)
    
    
    
    import torch
    import pandas as pd
    def motif_results(motif_out_dir, motif_pdb, motif):
        rows = []
        for design_dir in tqdm.tqdm(sorted(glob.glob(os.path.join(motif_out_dir, "design*")))):
            design_name = os.path.basename(design_dir)
    
            # load motif mask
            with open(os.path.join(design_dir, f"{motif}_spec.pkl"), "rb") as f:
                motif_mask = pickle.load(f)["motif_mask"]
    
            for state in [0, 1]:
                with open(os.path.join(design_dir, f"state{state}.pkl"), 'rb') as f:
                    outdict = CPU_Unpickler(f).load()
    
                for sample_idx in range(5):  # assume 5 samples per state
                    pdb_file = os.path.join(design_dir, f"state{state}_sample{sample_idx}.pdb")
                    if not os.path.exists(pdb_file):
                        continue
    
                    rmsd = motifRMSD(motif_pdb, pdb_file, motif_mask)
    
                    rows.append({
                        "design": design_name,
                        "state": state,
                        "sample": sample_idx,
                        "motifrmsd": rmsd.item() if hasattr(rmsd, "item") else float(rmsd),
                        "plddt": outdict["plddt"].cpu().numpy()[sample_idx].mean(),
                        "ptm": outdict["ptm"].cpu().numpy()[sample_idx],
                    })
                    
        full = pd.DataFrame(rows)
        
        agg = full.groupby(["design", "state"]).agg(
            motifRMSD_mean=("motifrmsd", "mean"),
            motifRMSD_std=("motifrmsd", "std"),
            plddt=("plddt", "mean"),
            ptm=("ptm", "mean")
        ).reset_index()
    
        agg = agg.pivot(index="design", columns="state").reset_index()
        agg.columns = ["_".join(map(str, col)).rstrip("_") for col in agg.columns.to_flat_index()]
        agg = agg.rename(columns=lambda c: c.replace("_0", "_unbound").replace("_1", "_bound"))
        
        agg.to_csv(os.path.join(motif_out_dir,"_aggresults.csv"),index=False)
        full.to_csv(os.path.join(motif_out_dir,"_fullresults.csv"),index=False)

    import sys, tqdm
    out_dir = sys.argv[1]
    for motif in os.listdir(out_dir):
        print(motif)
        motif_pdb = f"motifs/{motif}.pdb"
        motif_out_dir = os.path.join(out_dir,motif)
        motif_results(motif_out_dir, motif_pdb, motif)
        
    # motif = "3ixt"
    # design_dir = f"./out/onemotif_twostates/{motif}/design1/"

    # with open(os.path.join(design_dir, f"{motif}_spec.pkl"), "rb") as f:
    #     motif_mask = pickle.load(f)["motif_mask"]

    # motif_pdb = f"/data/cb/mihirb14/projects/BoltzDesign1/motifs/{motif}.pdb"

    # pdb_files = sorted(glob.glob(os.path.join(design_dir, "state*_sample*.pdb")))

    # from collections import defaultdict
    # state_groups = defaultdict(list)
    # for pdb in pdb_files:
    #     base = os.path.basename(pdb)
    #     state = base.split("_")[0]
    #     state_groups[state].append(pdb)

    # for state, files in state_groups.items():
    #     print(f"\n{state}:")
    #     for fpath in sorted(files):
    #         rmsd = motifRMSD(motif_pdb, fpath, motif_mask)
    #         print(f"  {os.path.basename(fpath)} → RMSD {rmsd:.3f}")
