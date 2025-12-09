import pickle, os
import os
import pickle
from dataclasses import asdict
from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb

import time
import os
import argparse
from utils import motif_utils
import torch

device = "cuda"

def _init_boltz(fine_tuned=False):
    predict_args={
        "recycling_steps": args.recycles,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": True,
    }
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = 1.638  # Default value
    boltz_model = Boltz1.load_from_checkpoint(
        os.path.join("boltz/boltz1_conf.ckpt"),
        strict=False,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        structure_prediction_training=True,
        no_msa=False,
        no_atom_encoder=False,
    ).eval().requires_grad_(False)

    if fine_tuned:
        ft = torch.load("boltz/step=1000.ckpt", map_location="cuda", weights_only=False)
        boltz_model.load_state_dict(ft["state_dict"], strict=False)
    
    return boltz_model

def run(args):
    if args.v1:
        from task import TASK_REGISTRY
    else:
        from task2 import TASK_REGISTRY
    ligands = []
    for ligs in args.ligands:
        ligands.append([])
        for lig in ligs.split(','):
            ligands[-1].append(tuple(lig.split(':')[::-1]))
    # args.ligands = [(x.split(":", 1)[1], x.split(":", 1)[0]) if ":" in x else (x, "ligand")
    #        for x in args.ligands]
    print(ligands)
    # boltz setup
    if args.motifs:
        out_dir = os.path.join(args.outpath, "_".join(args.motifs))
    else:
        out_dir = args.outpath
    os.makedirs(out_dir, exist_ok=True)

    # design
    for design in range(args.worker_id, args.num_designs, args.num_workers):
        print(f"\nStarting  {args.task} design {design+1}/{args.num_designs} for motifs {args.motifs} and ligands {ligands}")
        
        boltz_model = _init_boltz(args.fine_tuned)
        
        # init task
        # motif = get_motif(f'motifs/{args.motif}.pdb')
        
        # breakpoint()
        # ligand = 'Fc1c(Cl)ccc(n2cnnn2)c1c1c[n+]([O-])c(cc1)C(CC1CC1)n1cc(cn1)c1ccc(N)nc1C'
        
        if args.motifs:
            motif_templates = motif_utils.get_motif_scaffold_templates(
                [f"motifs/{m}.pdb" for m in args.motifs]
            )
            length = len(motif_templates[0]["motif_mask"])
        else:
            motif_templates = []
            length = args.length
        designer = TASK_REGISTRY[args.task](
            motifs=motif_templates,
            ligands=ligands,
            length=length,
            strength=args.strength,
        )

        # get motif residues for ligandmpnn
        motif_residues = None
        try:
            if hasattr(designer, "fixed_mask") and designer.fixed_mask is not None:
                motif_indices = (designer.fixed_mask.nonzero(as_tuple=True)[0]).tolist()
        except Exception:
            motif_residues = None
        
        t0 = time.perf_counter()
        print("Optimizing sequence...")
        designer.optimize(boltz_model, verbose=args.verbose, debug=args.debug, best_by_loss=args.best_by_loss)
        t1 = time.perf_counter()
        print(f"Optimization done in {t1 - t0:.1f} sec")
        print("Saving structures...")
        t2 = time.perf_counter()
        structs = designer.get_final_structs(boltz_model)
        t3 = time.perf_counter()
        print(f"Structure generation took {t3 - t2:.1f} sec")
        
        design_dir = os.path.join(out_dir, f"design{design}")
        os.makedirs(design_dir, exist_ok=True)
        if args.motifs:
            for i, motif_name in enumerate(args.motifs):
                # Save the sampled motif spec (mask, atom_dmat, etc.) for eval
                with open(os.path.join(design_dir, f"{motif_name}_spec.pkl"), "wb") as f:
                    pickle.dump(motif_templates[i], f)

                # Also save the sampled motif as a PDB so we can inspect the
                # exact motif configuration used in this design.
                motif_mask = motif_templates[i]["motif_mask"]
                spec_path = os.path.join("motifs", f"{motif_name}.pdb")
                pdb_out = os.path.join(design_dir, f"{motif_name}_sampled_motif.pdb")
                motif_utils.save_motif_pdb(spec_path, motif_mask, pdb_out)
            
        for output, struct_list, state_idx in structs:
            with open(os.path.join(design_dir, f"state{state_idx}.pkl"), "wb") as f:
                pickle.dump(output, f)

            for j, struct in enumerate(struct_list):
                base = f"state{state_idx}_sample{j}"
                with open(os.path.join(design_dir, base + ".pdb"), "w") as f:
                    f.write(to_pdb(struct))
                with open(os.path.join(design_dir, base + ".cif"), "w") as f:
                    f.write(to_mmcif(struct))
                    

        # LigandMPNN tied redesign 
        t4 = time.perf_counter()
        if getattr(args, "ligandmpnn_seqs", 0) and args.ligandmpnn_seqs > 0:
            print("Redesigning sequences with LigandMPNN...")
            from boltzdesign.tied_lmpnn import perform_tied_lmpnn_redesign
            lmpnn_seqs, fasta_path, best_sample_idx_by_state = perform_tied_lmpnn_redesign(
                design_dir=design_dir,
                state_results=structs,
                num_seqs=int(args.ligandmpnn_seqs),
                motif_indices=motif_indices
            )
            t5 = time.perf_counter()
            print(f"LigandMPNN redesign took {t5 - t4:.1f} sec")
            print("Saving structures for LigandMPNN redesigns...")

            # read LigandMPNN redesigns and regenerate final boltz structures
            regen_dir = os.path.join(design_dir, "lmpnn", "boltz_regen")
            os.makedirs(regen_dir, exist_ok=True)

            # regenerate structures per sequence using designer.get_final_structs
            for seq_idx, seq in enumerate(lmpnn_seqs):
                #designer.get_seq = (lambda s=seq: s) # monkey patch - not the best way to do this
                print("this is the ligandmpnn seq", seq)
                regen_structs = designer.get_final_structs(boltz_model, samples = 1, set_seq = seq)

                for output, struct_list, state_idx in regen_structs:
                    with open(os.path.join(regen_dir, f"lmpnn_seq{seq_idx}_state{state_idx}.pkl"), "wb") as f:
                        pickle.dump(output, f)

                    for j, struct in enumerate(struct_list):
                        base = f"lmpnn_seq{seq_idx}_state{state_idx}_sample{j}"
                        with open(os.path.join(regen_dir, base + ".pdb"), "w") as f:
                            f.write(to_pdb(struct))
                        with open(os.path.join(regen_dir, base + ".cif"), "w") as f:
                            f.write(to_mmcif(struct))

            t6 = time.perf_counter()
            print(f"Generating structures for LigandMPNN redesigns took {t6 - t5:.1f} sec")
            print(f"Finished design {args.motifs} {design+1} in {t6 - t0:.1f} sec total")
        else:
            print(f"Finished design {args.motifs} {design+1} in {t3 - t0:.1f} sec total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--len', type=int, default=100)
    parser.add_argument('--num_designs',type= int, default = 1)
    parser.add_argument("--motifs", nargs="+", required=False, help="motif names (e.g. 4jhw 1ycr)")
    parser.add_argument('--ligands', nargs="+",default=["ligand:Fc1c(Cl)ccc(n2cnnn2)c1c1c[n+]([O-])c(cc1)C(CC1CC1)n1cc(cn1)c1ccc(N)nc1C"] ,help="space separated ligand smiles")
    parser.add_argument('-o','--outpath', type=str, default = "./out/")
    parser.add_argument("--task", required=True, help="task to run")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--v1", action='store_true')
    parser.add_argument("--recycles", default=0, type=int)
    parser.add_argument("--length", default=None, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--worker_id", default=0, type=int)
    parser.add_argument("--strength", default=0, type=float)
    parser.add_argument("--ligandmpnn_seqs", default=0, type=int, help="If >0, run tied LigandMPNN once producing N sequences")
    parser.add_argument("--best_by_loss", action='store_true', help="Save the structure with the lowest loss instead of the last iteration")
    parser.add_argument("--fine_tuned", action='store_true', help="Use the fine-tuned model")
    args = parser.parse_args()

    
    run(args)
