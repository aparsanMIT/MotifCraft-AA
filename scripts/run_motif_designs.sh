#!/bin/bash
# run_motif_designs.sh
# Usage: ./run_motif_designs.sh

visible_devices_list=(4)   # gpus available on this machine
motifs=(4jhw)     # motifs to design
num_designs=100
outdir=./out/onemotif_twostates

conda_env=boltz_design
home_override=/data/cb/mihirb14/bin

for i in "${!motifs[@]}"; do
    motif="${motifs[$i]}"
    gpu="${visible_devices_list[$((i % ${#visible_devices_list[@]}))]}"

    session="design_${motif}"
    echo "Launching motif $motif on GPU $gpu in tmux session $session"

    tmux new-session -d -s "$session" "
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $conda_env;
        CUDA_VISIBLE_DEVICES=$gpu HOME=$home_override \
        python mydesign.py -o $outdir --num_designs $num_designs --motif $motif;
        exec bash
    "
done
