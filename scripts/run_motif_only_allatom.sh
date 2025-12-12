#!/bin/bash
# run_motif_only_allatom.sh
# Usage: ./run_motif_only_allatom.sh
#
# NOTE: This script assumes that, upstream, each motif template is augmented
# with:
#   - motif['atom_dmat']          # all-atom distance matrix [N_atoms, N_atoms]
#   - motif['atom_token_indices'] # token indices for those atoms in Boltz
# so that AllAtomMotifLoss can be evaluated.

num_designs=100
task=motif_only_allatom

visible_devices_list=(4 5 6 7)   # GPUs available on this machine

# Motifs to design (PDB IDs without path, expected under ./motifs/)
motifs=(7mrx_85 7mrx_60 7mrx_128 6exz_short 6exz_med 1qjg 4zyp 4jhw 6e6r_short 6e6r_med 6e6r_long 5yui 5wn9 2kl8 1ycr 3ixt 5trv_short 5trv_med 5trv_long 5tpn 5ius 1prw 6exz_long 1bcf)
#motifs=(1prw)
# Ligands are unused for motif_only_allatom, but mydesign.py expects the flag
ligands='ligand:CC'  # dummy ligand; ignored by motif_only_allatom task

outdir=/data/cb/scratch/aparsan/BoltzDesign1/outputs/full_all_atom_finetuned
logdir=/data/cb/scratch/aparsan/BoltzDesign1/logs/full_all_atom_finetuned

conda_env=boltz_design
home_override=/data/cb/scratch/aparsan/BoltzDesign1

mkdir -p "$logdir"

echo "task: $task"
echo "motifs: [${motifs[@]}]"
echo "ligands (ignored by task): [\"$ligands\"]"
echo "outdir: $outdir"
echo "logdir: $logdir"

for i in "${!motifs[@]}"; do
    motif="${motifs[$i]}"
    gpu="${visible_devices_list[$((i % ${#visible_devices_list[@]}))]}"

    session="aa_finetuned_${motif}"
    echo "Launching motif_only_residue for $motif on GPU $gpu in tmux session $session"

    logfile="$logdir/${motif}.log"
    tmux new-session -d -s "$session" "
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate $conda_env;
        CUDA_VISIBLE_DEVICES=$gpu HOME=$home_override \
        python /data/cb/scratch/aparsan/BoltzDesign1/mydesign.py -o $outdir --num_designs $num_designs --task $task --motifs $motif --ligands \"$ligands\" --fine_tuned --best_by_loss 2>&1 | tee $logfile;
        exec bash
    "
done
