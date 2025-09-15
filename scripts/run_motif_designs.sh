#!/bin/bash
# run_motif_designs.sh
# Usage: ./run_motif_designs.sh


num_designs=100
task=onemotif_twostates_neg

visible_devices_list=(0 1 2 3 4 5 6 7)   # gpus available on thes machine

 # motifs to design
# motifs=(1bcf 1prw 1qjg 1ycr 2kl8 3ixt 4jhw 4zyp)                      
# motifs=(5ius 5tpn 5trv_long 5trv_med 5trv_short 5wn9 5yui 6e6r_long)     
motifs=(6e6r_med 6e6r_short 6exz_long 6exz_med 6exz_short 7mrx_60 7mrx_85 7mrx_128)

# ligands for allostery
ligands='[Zn+2]'

outdir=./out/onemotif_twostates_neg_zn

conda_env=boltz_design
home_override=/data/cb/mihirb14/bin

echo "task: $task"
echo "motifs: [${motifs[@]}]"
echo "ligands: [\"$ligands\"]"
echo "outdir: $outdir"

for i in "${!motifs[@]}"; do
    motif="${motifs[$i]}"
    gpu="${visible_devices_list[$((i % ${#visible_devices_list[@]}))]}"

    session="design_${motif}"
    echo "Launching motif $motif on GPU $gpu in tmux session $session"

    tmux new-session -d -s "$session" "
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $conda_env;
        CUDA_VISIBLE_DEVICES=$gpu HOME=$home_override \
        python mydesign.py -o $outdir --num_designs $num_designs --task $task --motifs $motif --ligands \"$ligands\";
        exec bash
    "
done
