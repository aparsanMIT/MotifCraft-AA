import pickle, os
import os
import pickle
from dataclasses import asdict
from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
import numpy as np
from utils import protein, residue_constants
from task import TASK_REGISTRY
import time
import os
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--len', type=int, default=100)
# parser.add_argument('--smiles', type=str,default=None)
parser.add_argument('--num_designs',type= int, default = 1)
parser.add_argument('--motif', type=str, default="3ixt")
parser.add_argument('-o','--outpath', type=str, default = "./out/")
parser.add_argument("--task", required=True, choices=TASK_REGISTRY.keys(), help="task to run")
args = parser.parse_args()

device = "cuda"

def get_motif(path):
    from utils.motif_utils import load_motif_spec, sample_motif_mask
    spec = load_motif_spec(path)
    masks = sample_motif_mask(spec)
    motif_mask = masks['sequence']
    motif_idx = masks['group']
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
        


# boltz setup
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

out_dir = os.path.join(args.outpath, args.task, args.motif)
os.makedirs(out_dir, exist_ok=True)

# design
for design in range(args.num_designs):
    print(f"\nStarting  {args.task} design {design+1}/{args.num_designs} for motif {args.motif}")
    
    # init task
    motif = get_motif(f'motifs/{args.motif}.pdb')
    ligand = 'Fc1c(Cl)ccc(n2cnnn2)c1c1c[n+]([O-])c(cc1)C(CC1CC1)n1cc(cn1)c1ccc(N)nc1C'
    designer = TASK_REGISTRY[args.task](motif, ligand, length=len(motif['motif_mask']))
    
    t0 = time.perf_counter()
    print("Optimizing sequence...")
    designer.optimize(boltz_model)
    t1 = time.perf_counter()
    print(f"Optimization done in {t1 - t0:.1f} sec")

    print("Saving structures...")
    t2 = time.perf_counter()
    structs = designer.get_final_structs(boltz_model)
    t3 = time.perf_counter()
    print(f"Structure generation took {t3 - t2:.1f} sec")
    
    design_dir = os.path.join(out_dir, f"design{design}")
    os.makedirs(design_dir, exist_ok=True)
    
    with open(os.path.join(design_dir,f"{args.motif}_spec.pkl"), "wb") as f:    # also save motifspec for eval
        pickle.dump(motif, f)
        
    for output, struct_list, state_idx in structs:
        with open(os.path.join(design_dir, f"state{state_idx}.pkl"), "wb") as f:
            pickle.dump(output, f)

        for j, struct in enumerate(struct_list):
            base = f"state{state_idx}_sample{j}"
            with open(os.path.join(design_dir, base + ".pdb"), "w") as f:
                f.write(to_pdb(struct))
            with open(os.path.join(design_dir, base + ".cif"), "w") as f:
                f.write(to_mmcif(struct))
                
    print(f"Finished design {motif} {design+1} in {t3 - t0:.1f} sec total")
