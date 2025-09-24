import pickle, os
import os
import pickle
from dataclasses import asdict
from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
from task import TASK_REGISTRY
import time
import os
import argparse
from utils import motif_utils

device = "cuda"

def _init_boltz():
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
    
    return boltz_model

def run(args):
    args.ligands = [(x.split(":", 1)[1], x.split(":", 1)[0]) if ":" in x else (x, "ligand")
           for x in args.ligands]
    print(args.ligands)
    # boltz setup
    boltz_model = _init_boltz()

    out_dir = os.path.join(args.outpath, "_".join(args.motifs))
    os.makedirs(out_dir, exist_ok=True)

    # design
    for design in range(args.num_designs):
        print(f"\nStarting  {args.task} design {design+1}/{args.num_designs} for motifs {args.motifs} and ligands {args.ligands}")
        
        # init task
        # motif = get_motif(f'motifs/{args.motif}.pdb')
        motif_templates = motif_utils.get_motif_scaffold_templates([f"motifs/{m}.pdb" for m in args.motifs])
        # breakpoint()
        # ligand = 'Fc1c(Cl)ccc(n2cnnn2)c1c1c[n+]([O-])c(cc1)C(CC1CC1)n1cc(cn1)c1ccc(N)nc1C'

        designer = TASK_REGISTRY[args.task](motifs=motif_templates, ligands=args.ligands, length=len(motif_templates[0]['motif_mask']))
        
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
        
        for i, motif in enumerate(args.motifs):
            with open(os.path.join(design_dir,f"{motif}_spec.pkl"), "wb") as f:    # also save motifspec for eval
                pickle.dump(motif_templates[i], f)
            
        for output, struct_list, state_idx in structs:
            with open(os.path.join(design_dir, f"state{state_idx}.pkl"), "wb") as f:
                pickle.dump(output, f)

            for j, struct in enumerate(struct_list):
                base = f"state{state_idx}_sample{j}"
                with open(os.path.join(design_dir, base + ".pdb"), "w") as f:
                    f.write(to_pdb(struct))
                with open(os.path.join(design_dir, base + ".cif"), "w") as f:
                    f.write(to_mmcif(struct))
                    
        print(f"Finished design {args.motifs} {design+1} in {t3 - t0:.1f} sec total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--len', type=int, default=100)
    parser.add_argument('--num_designs',type= int, default = 1)
    parser.add_argument("--motifs", nargs="+", required=True, help="motif names (e.g. 4jhw 1ycr)")
    parser.add_argument('--ligands', nargs="+",default=["Fc1c(Cl)ccc(n2cnnn2)c1c1c[n+]([O-])c(cc1)C(CC1CC1)n1cc(cn1)c1ccc(N)nc1C"] ,help="space separated ligand smiles")
    parser.add_argument('-o','--outpath', type=str, default = "./out/")
    parser.add_argument("--task", required=True, choices=TASK_REGISTRY.keys(), help="task to run")
    args = parser.parse_args()

    
    run(args)
