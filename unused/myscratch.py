cfg = dict(
    length=100,
    binder_chain="A",
    design_algorithm="3stages",
    recycling_steps=0,
    pre_iteration=30,
    soft_iteration=75,
    soft_iteration_1=50,
    soft_iteration_2=50,
    temp_iteration=45,
    hard_iteration=5,
    semi_greedy_steps=2,
    learning_rate=0.1,
    learning_rate_pre=0.2,
    inter_chain_cutoff=20.0,
    intra_chain_cutoff=14.0,
    num_inter_contacts=1,
    num_intra_contacts=2,
    e_soft=0.8,
    e_soft_1=0.8,
    e_soft_2=1.0,
    alpha=2.0,
    pre_run=False,
    set_train=True,
    use_temp=True,
    disconnect_feats=True,
    disconnect_pairformer=False,
    mask_ligand=True,
    distogram_only=True,
    input_res_type=False,
    non_protein_target=True,
    increasing_contact_over_itr=False,
    loss_scales={'con_loss': 1.0, 'i_con_loss': 1.0, 'plddt_loss': 0.1, 'pae_loss': 0.4, 'i_pae_loss': 0.1, 'rg_loss': 0.0, 'helix_loss': -0.07880484458590675},
    optimize_contact_per_binder_pos=False,
    pocket_conditioning=False,
    chain_to_number={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9},
    msa_max_seqs=4096,
    optimizer_type="SGD",
    save_trajectory=False,
)

def boltz_hallucination(
    # Required arguments
    boltz_model,
    yaml_path,
    ccd_lib,
    cfg,
):
    
    predict_args = {
        "recycling_steps": 1,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    data["sequences"][chain_to_number[binder_chain]]["protein"]["sequence"] = (
        "X" * length
    )

    target = parse_boltz_schema(name, data, ccd_lib)


    batch, structure = get_batch(
        target,
        max_seqs=msa_max_seqs,
        length=length,
        pocket_conditioning=pocket_conditioning,
    )
    batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}

    ## initialize res_type_logits
    if pre_run:
        term1 = torch.distributions.Gumbel(0, 1).sample(
                batch["res_type"][
                    batch["entity_id"] == chain_to_number[binder_chain], :
                ].shape
            ).to(device)

        term2 = torch.sum(
            torch.eye(batch["res_type"].shape[-1])[
                [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            ],
            dim=0,
        ).to(device) * (1e10)
        
        batch["res_type_logits"] = batch["res_type"].clone().detach().to(device).float()
        batch["res_type_logits"][
            batch["entity_id"] == chain_to_number[binder_chain], :
        ] = torch.softmax(term1 - term2)

    else:
        batch["res_type_logits"] = torch.from_numpy(input_res_type).to(device)

    if non_protein_target:
        batch["msa"] = batch["res_type_logits"].unsqueeze(0).to(device)
        batch["msa_paired"] = torch.ones(
            batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
        ).to(device)
        batch["deletion_value"] = torch.zeros(
            batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
        ).to(device)
        batch["has_deletion"] = torch.full(
            (batch["res_type"].shape[0], 1, batch["res_type"].shape[1]), False
        ).to(device)
        batch["msa_mask"] = torch.ones(
            batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
        ).to(device)
        batch["profile"] = batch["msa"].float().mean(dim=0).to(device)
        batch["deletion_mean"] = torch.zeros(batch["deletion_mean"].shape).to(device)
        batch["res_type"] = batch["res_type"].float()

    batch["res_type_logits"].requires_grad = True
    optimizer = torch.optim.SGD([batch["res_type_logits"]], lr=learning_rate)
    
    if pre_run:
        (
            batch,
            plots,
            loss_history,
            i_con_loss_history,
            con_loss_history,
            plddt_loss_history,
            distogram_history,
            sequence_history,
            traj_coords_list,
            traj_plddt_list,
        ) = design(
            batch,
            iters=pre_iteration,
            soft=1.0,
            mask=mask,
            chain_mask=chain_mask,
            cfg,
        )
    else:
        if design_algorithm == "3stages":
            print("-" * 100)
            print(f"logits to softmax(T={e_soft})")
            print("-" * 100)
            (
                batch,
                plots,
                loss_history,
                i_con_loss_history,
                con_loss_history,
                plddt_loss_history,
                distogram_history,
                sequence_history,
                traj_coords_list1,
                traj_plddt_list1,
            ) = design(
                batch,
                iters=soft_iteration,
                e_soft=e_soft,
                num_optimizing_binder_pos=1,
                e_num_optimizing_binder_pos=8,
                cfg,
            )
            print("-" * 100)
            print("softmax(T=1) to softmax(T=0.01)")
            print("-" * 100)
            print("set res_type_logits to logits")
            new_logits = (
                (alpha * batch["res_type_logits"]).clone().detach().requires_grad_(True)
            )
            batch["res_type_logits"] = new_logits
            optimizer = torch.optim.SGD([batch["res_type_logits"]], lr=learning_rate)
            (
                batch,
                plots,
                loss_history,
                i_con_loss_history,
                con_loss_history,
                plddt_loss_history,
                distogram_history,
                sequence_history,
                traj_coords_list2,
                traj_plddt_list2,
            ) = design(
                batch,
                iters=temp_iteration,
                soft=1.0,
                temp=1.0,
                e_temp=0.01,
                num_optimizing_binder_pos=8,
                e_num_optimizing_binder_pos=12,
                mask=mask,
                chain_mask=chain_mask,
                cfg,
            )
            print("-" * 100)
            print("hard")
            print("-" * 100)
            (
                batch,
                plots,
                loss_history,
                i_con_loss_history,
                con_loss_history,
                plddt_loss_history,
                distogram_history,
                sequence_history,
                traj_coords_list3,
                traj_plddt_list3,
            ) = design(
                batch,
                iters=hard_iteration,
                soft=1.0,
                hard=1.0,
                temp=0.01,
                num_optimizing_binder_pos=12,
                e_num_optimizing_binder_pos=16,
                cfg,
            )
            traj_coords_list = (
                traj_coords_list1 + traj_coords_list2 + traj_coords_list3
                if save_trajectory
                else []
            )
            traj_plddt_list = (
                traj_plddt_list1 + traj_plddt_list2 + traj_plddt_list3
                if save_trajectory
                else []
            )

       
    

    if pre_run:
        predict_args = {
            "recycling_steps": 3,  # Default value
            "sampling_steps": 200,  # Default value
            "diffusion_samples": 1,  # Default value
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": False,
        }

        best_logits = batch["res_type_logits"]
        best_seq = "".join(
            [
                alphabet[i]
                for i in torch.argmax(
                    batch["res_type"][
                        batch["entity_id"] == chain_to_number[binder_chain], :
                    ],
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            ]
        )
        data["sequences"][chain_to_number[binder_chain]]["protein"][
            "sequence"
        ] = best_seq
        return (
            batch["res_type"].detach().cpu().numpy(),
            plots,
            loss_history,
            distogram_history,
            sequence_history,
            traj_coords_list,
            traj_plddt_list,
        )

    boltz_model.eval()

    if best_batch is None:
        if first_step_best_batch is not None:
            best_batch = first_step_best_batch
        else:
            best_batch = batch

    predict_args = {
        "recycling_steps": 3,  # Default value
        "sampling_steps": 200,  # Default value
        "diffusion_samples": 1,  # Default value
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": False,
    }

    def _mutate(sequence, best_logits, i_prob):
        mutated_sequence = list(sequence)  # Create a copy of the input tensor
        i = np.random.choice(np.arange(length), p=i_prob / i_prob.sum())
        i_logits = best_logits[:, i]
        i_logits = i_logits - torch.max(i_logits)
        i_X = i_logits - (
            torch.sum(
                torch.eye(i_logits.shape[-1])[
                    [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                ],
                dim=0,
            )
            * (1e10)
        ).to(device)
        i_aa = torch.multinomial(torch.softmax(i_X, dim=-1), 1).item()
        mutated_sequence[i] = alphabet[i_aa]
        return "".join(mutated_sequence)

    best_logits = best_batch["res_type_logits"]
    best_seq = "".join(
        [
            alphabet[i]
            for i in torch.argmax(
                best_batch["res_type"][
                    best_batch["entity_id"] == chain_to_number[binder_chain], :
                ],
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        ]
    )
    data["sequences"][chain_to_number[binder_chain]]["protein"]["sequence"] = best_seq

    data_apo = copy.deepcopy(data)  # This handles all types of values correctly
    data_apo.pop("constraints", None)  # Remove constraints if they exist
    data_apo["sequences"] = [
        data_apo["sequences"][chain_to_number[binder_chain]]
    ]  # Keep only chain B



    best_batch, best_batch_apo, best_structure, best_structure_apo = _update_batches(
        data, data_apo
    )
    output = _run_model(boltz_model, best_batch, predict_args)
    output_apo = _run_model(boltz_model, best_batch_apo, predict_args)

    prev_sequence = "".join(
        [
            alphabet[i]
            for i in torch.argmax(
                best_batch["res_type"][
                    best_batch["entity_id"] == chain_to_number[binder_chain], :
                ],
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        ]
    )
    prev_iptm = output["iptm"].detach().cpu().numpy()
    print("best design iptm", prev_iptm)
    print("Semi-greedy steps", semi_greedy_steps)
    for step in range(semi_greedy_steps):
        confidence_score = []
        mutated_sequence_ls = []

        for t in range(10):
            plddt = output["plddt"][
                best_batch["entity_id"] == chain_to_number[binder_chain]
            ]
            i_prob = (
                np.ones(length)
                if plddt is None
                else torch.maximum(1 - plddt, torch.tensor(0))
            )
            i_prob = (
                i_prob.detach().cpu().numpy() if torch.is_tensor(i_prob) else i_prob
            )
            sequence = "".join(
                [
                    alphabet[i]
                    for i in torch.argmax(
                        best_batch["res_type"][
                            best_batch["entity_id"] == chain_to_number[binder_chain], :
                        ],
                        dim=-1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                ]
            )
            mutated_sequence = _mutate(sequence, best_logits, i_prob)
            data["sequences"][chain_to_number[binder_chain]]["protein"][
                "sequence"
            ] = mutated_sequence
            best_batch, _, _, _ = _update_batches(data, data_apo)
            output = _run_model(boltz_model, best_batch, predict_args)

            iptm = output["iptm"].detach().cpu().numpy()
            confidence_score.append(iptm)
            mutated_sequence_ls.append(mutated_sequence)
            print(f"Step {step}, Epoch {t}, iptm {iptm[0]:.3f}")

        best_id = np.argmax(confidence_score)
        best_iptm = confidence_score[best_id]



######################### END MAIN DESIGN FUNC #################
def get_batch(
    target, max_seqs=0, length=100, pocket_conditioning=False, keep_record=False
):
    target_id = target.record.id
    structure = target.structure

    structure = Structure(
        atoms=structure.atoms,
        bonds=structure.bonds,
        residues=structure.residues,
        chains=structure.chains,
        connections=structure.connections.astype(Connection),
        interfaces=structure.interfaces,
        mask=structure.mask,
    )

    msas = {}
    for chain in target.record.chains:
        msa_id = chain.msa_id
        if msa_id != -1:
            msa = np.load(msa_id)
            msas[chain.chain_id] = MSA(**msa)

    input = Input(structure, msas)

    tokenizer = BoltzTokenizer()
    tokenized = tokenizer.tokenize(input)
    featurizer = BoltzFeaturizer()

    batch = featurizer.process(
        tokenized,
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=max_seqs,
        pad_to_max_seqs=False,
        symmetries={},
        compute_symmetries=False,
        inference_binder=None,
        inference_pocket=None,
    )

    if keep_record:
        batch["record"] = target.record

    return batch, structure


def get_model_loss(
    batch,
    plots,
    loss_history,
    i_con_loss_history,
    con_loss_history,
    plddt_loss_history,
    distogram_history,
    sequence_history,
    pre_run=False,
    mask_ligand=False,
    distogram_only=False,
    predict_args=None,
    loss_scales=None,
    binder_chain="A",
    increasing_contact_over_itr=False,
    optimize_contact_per_binder_pos=False,
    num_inter_contacts=2,
    num_intra_contacts=4,
    num_optimizing_binder_pos=1,
    inter_chain_cutoff=21.0,
    intra_chain_cutoff=14.0,
    save_trajectory=False,
):
    traj_coords = None
    traj_plddt = None

    # Handle masking first if needed
    if pre_run and mask_ligand:
        batch["token_pad_mask"][
            batch["entity_id"] != chain_to_number[binder_chain]
        ] = 0
        masked_token_to_rep = torch.ones_like(batch["token_to_rep_atom"])
        masked_token_to_rep[
            batch["entity_id"] == chain_to_number[binder_chain], :
        ] = 0
        masked_token_to_rep_index = torch.nonzero(
            batch["token_to_rep_atom"] * masked_token_to_rep, as_tuple=True
        )[2]
        batch["atom_pad_mask"][:, masked_token_to_rep_index] = 0

    # Common arguments for get_distogram_confidence
    confidence_args = {
        "recycling_steps": predict_args["recycling_steps"],
        "num_sampling_steps": predict_args["sampling_steps"],
        "multiplicity_diffusion_train": 1,
        "diffusion_samples": predict_args["diffusion_samples"],
        "run_confidence_sequentially": True,
        "disconnect_feats": disconnect_feats,
        "disconnect_pairformer": disconnect_pairformer,
    }

    if save_trajectory:
        # Get model output with trajectory info
        dict_out = boltz_model.get_distogram_confidence(
            batch, **confidence_args
        )
        traj_coords = dict_out["sample_atom_coords"][0].detach().cpu().numpy()
        traj_plddt = dict_out["plddt"][0].detach().cpu().numpy()
    else:
        # Get model output without trajectory
        if pre_run or distogram_only:
            dict_out, s, z, s_inputs = boltz_model.get_distogram(batch)
        else:
            dict_out = boltz_model.get_distogram_confidence(
                batch, **confidence_args
            )

    pdist = dict_out["pdistogram"]
    mid_pts = get_mid_points(pdist).to(device)

    # Calculate contact losses
    con_loss = get_con_loss(
        pdist,
        mid_pts,
        num=num_intra_contacts,
        seqsep=9,
        cutoff=intra_chain_cutoff,
        binary=False,
        mask_1d=chain_mask,
        mask_1b=chain_mask,
    )

    if optimize_contact_per_binder_pos:
        if increasing_contact_over_itr:
            num_optimizing_binder_pos = (
                0 if pre_run else num_optimizing_binder_pos
            )
            i_con_loss = get_con_loss(
                pdist,
                mid_pts,
                num=num_inter_contacts,
                seqsep=0,
                num_pos=num_optimizing_binder_pos,
                cutoff=inter_chain_cutoff,
                binary=False,
                mask_1d=chain_mask,
                mask_1b=1 - chain_mask,
            )
        else:
            i_con_loss = get_con_loss(
                pdist,
                mid_pts,
                num=num_inter_contacts,
                seqsep=0,
                cutoff=inter_chain_cutoff,
                binary=False,
                mask_1d=chain_mask,
                mask_1b=1 - chain_mask,
            )

    else:

        i_con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=num_inter_contacts,
            seqsep=0,
            cutoff=inter_chain_cutoff,
            binary=False,
            mask_1d=1 - chain_mask,
            mask_1b=chain_mask,
        )

    mask_2d = chain_mask[:, :, None] * chain_mask[:, None, :]
    helix_loss = _get_helix_loss(
        pdist, mid_pts, offset=None, mask_2d=mask_2d, binary=True
    )

    if pre_run and mask_ligand:
        losses = {"con_loss": con_loss, "helix_loss": helix_loss}
    else:
        losses = {
            "con_loss": con_loss,
            "i_con_loss": i_con_loss,
            "helix_loss": helix_loss,
        }

    if not pre_run and not distogram_only:
        plddt_loss = get_plddt_loss(dict_out["plddt"], mask_1d=chain_mask)
        pae = (dict_out["pae"] + dict_out["pae"].transpose(-2, -1)) / 2
        i_pae_loss = get_pae_loss(
            pae, mask_1d=1 - chain_mask, mask_1b=chain_mask
        )
        pae_loss = get_pae_loss(pae, mask_1d=chain_mask, mask_1b=chain_mask)
        rg_loss, rg = add_rg_loss(
            dict_out["sample_atom_coords"],
            batch,
            length,
            binder_chain=binder_chain,
        )

        losses.update(
            {
                "plddt_loss": plddt_loss,
                "i_pae_loss": i_pae_loss,
                "pae_loss": pae_loss,
                "rg_loss": rg_loss,
            }
        )

        plddt_loss_history.append(plddt_loss.item())

    bins = mid_points < 8.0
    px = torch.sum(
        torch.softmax(dict_out["pdistogram"], dim=-1)[:, :, :, bins], dim=-1
    )

    if loss_scales is None:
        loss_scales = {
            "con_loss": 1.0,
            "i_con_loss": 1.0,
            "helix_loss": random.uniform(-0.4, 0.0),
            "plddt_loss": 0.1,
            "pae_loss": 0.4,
            "i_pae_loss": 0.1,
            "rg_loss": 0.0,
        }

    # Calculate total loss and print individual losses
    total_loss = sum(loss * loss_scales[name] for name, loss in losses.items())
    loss_str = [f"{k}:{v.item():.2f}" for k, v in losses.items()]
    plots.append(px[0].detach().cpu().numpy())
    loss_history.append(total_loss.item())
    i_con_loss_history.append(i_con_loss.item())
    con_loss_history.append(con_loss.item())
    # distogram_history.append(torch.softmax(dict_out['pdistogram'], dim=-1)[0].detach().cpu().numpy())
    distogram_history.append(px[0].detach().cpu().numpy())
    sequence_history.append(
        batch["res_type"][0, :, 2:22].detach().cpu().numpy()
    )

    return (
        total_loss,
        plots,
        loss_history,
        i_con_loss_history,
        con_loss_history,
        distogram_history,
        sequence_history,
        plddt_loss_history,
        loss_str,
        traj_coords,
        traj_plddt,
    )

def update_sequence(
    opt, batch, mask, alpha=2.0, non_protein_target=False, binder_chain="A"
):
    batch["logits"] = alpha * batch["res_type_logits"]
    X = batch["logits"] - torch.sum(
        torch.eye(batch["logits"].shape[-1])[
            [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        ],
        dim=0,
    ).to(device) * (1e10)
    # banish invalid toks
    
    batch["soft"] = torch.softmax(X / opt["temp"], dim=-1) # probs
    
    batch["hard"] = torch.zeros_like(batch["soft"]).scatter_(
        -1, batch["soft"].max(dim=-1, keepdim=True)[1], 1.0
    ) # one hot
    
    batch["hard"] = (batch["hard"] - batch["soft"]).detach() + batch["soft"] # batch hard carries the same grad as batch soft!

    
    batch["pseudo"] = (
        opt["soft"] * batch["soft"]
        + (1 - opt["soft"]) * batch["res_type_logits"]
    ) # interp between res type logits and soft

    
    batch["pseudo"] = (
        opt["hard"] * batch["hard"] + (1 - opt["hard"]) * batch["pseudo"]
    ) # interp between above and one hot

    
    batch["res_type"] = batch["pseudo"] * mask + batch["res_type_logits"] * (
        1 - mask
    )

    if non_protein_target:
        batch["msa"] = batch["res_type"].unsqueeze(0).to(device).detach()
        batch["profile"] = batch["msa"].float().mean(dim=0).to(device).detach()
    else:
        batch["msa"][:, 0, :, :] = batch["res_type"].to(device).detach()
        batch["profile"][
            batch["entity_id"] == chain_to_number[binder_chain], :
        ] = (
            batch["msa"][
                :,
                0,
                (batch["entity_id"] == chain_to_number[binder_chain])[0],
                :,
            ]
            .float()
            .mean(dim=1)
            .to(device)
            .detach()
        )

    return batch

def design(
    batch,
    iters=None,
    soft=0.0,
    e_soft=None,
    step=1.0,
    e_step=None,
    temp=1.0,
    e_temp=None,
    hard=0.0,
    e_hard=None,
    num_optimizing_binder_pos=1,
    e_num_optimizing_binder_pos=1,
    learning_rate=1.0,
    inter_chain_cutoff=21.0,
    intra_chain_cutoff=14.0,
    mask=None,
    chain_mask=None,
    length=100,
    plots=None,
    loss_history=None,
    i_con_loss_history=None,
    con_loss_history=None,
    plddt_loss_history=None,
    distogram_history=None,
    sequence_history=None,
    pre_run=False,
    mask_ligand=False,
    distogram_only=False,
    predict_args=None,
    alpha=2.0,
    loss_scales=None,
    binder_chain="A",
    non_protein_target=False,
    increasing_contact_over_itr=False,
    optimize_contact_per_binder_pos=False,
    num_inter_contacts=2,
    num_intra_contacts=4,
    save_trajectory=False,
):

    prev_sequence = ""
    m = {
        "soft": [soft, e_soft],
        "temp": [temp, e_temp],
        "hard": [hard, e_hard],
        "step": [step, e_step],
        "num_optimizing_binder_pos": [
            num_optimizing_binder_pos,
            e_num_optimizing_binder_pos,
        ],
    }
    m = {k: [s, (s if e is None else e)] for k, (s, e) in m.items()}

    opt = {}
    traj_coords_list = []
    traj_plddt_list = []
    for i in range(iters):
        for k, (s, e) in m.items():
            if k == "temp":
                opt[k] = e + (s - e) * (1 - (i) / iters) ** 2
            else:
                v = s + (e - s) * ((i) / iters)
                if k == "step":
                    step = v
                opt[k] = v

        lr_scale = step * ((1 - opt["soft"]) + (opt["soft"] * opt["temp"]))
        num_optimizing_binder_pos = int(opt["num_optimizing_binder_pos"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * lr_scale

        opt["lr_rate"] = learning_rate * lr_scale

        batch = update_sequence(
            opt,
            batch,
            mask,
            non_protein_target=non_protein_target,
            binder_chain=binder_chain,
        )
        (
            total_loss,
            plots,
            loss_history,
            i_con_loss_history,
            con_loss_history,
            distogram_history,
            sequence_history,
            plddt_loss_history,
            loss_str,
            traj_coords,
            traj_plddt,
        ) = get_model_loss(
            batch,
            plots,
            loss_history,
            i_con_loss_history,
            con_loss_history,
            plddt_loss_history,
            distogram_history,
            sequence_history,
            pre_run,
            mask_ligand,
            distogram_only,
            predict_args,
            loss_scales,
            binder_chain,
            increasing_contact_over_itr,
            optimize_contact_per_binder_pos=optimize_contact_per_binder_pos,
            num_inter_contacts=num_inter_contacts,
            num_intra_contacts=num_intra_contacts,
            num_optimizing_binder_pos=num_optimizing_binder_pos,
            inter_chain_cutoff=inter_chain_cutoff,
            intra_chain_cutoff=intra_chain_cutoff,
            save_trajectory=save_trajectory,
        )
        traj_coords_list.append(traj_coords)
        traj_plddt_list.append(traj_plddt)
        current_sequence = "".join(
            [
                alphabet[i]
                for i in torch.argmax(
                    batch["res_type"][
                        batch["entity_id"] == chain_to_number[binder_chain], :
                    ],
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            ]
        )
        if prev_sequence is not None:
            diff_count = sum(
                1 for a, b in zip(current_sequence, prev_sequence) if a != b
            )
            diff_percentage = (diff_count / length) * 100
        prev_sequence = current_sequence
        total_loss.backward()
        if batch["res_type_logits"].grad is not None:
            batch["res_type_logits"].grad[
                batch["entity_id"] != chain_to_number[binder_chain], :
            ] = 0
            batch["res_type_logits"].grad[
                ..., [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            ] = 0
            batch["res_type_logits"].grad = norm_seq_grad(
                batch["res_type_logits"].grad, chain_mask
            )
            optimizer.step()
            optimizer.zero_grad()
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {i}: lr: {current_lr:.3f}, soft: {opt['soft']:.2f}, hard: {opt['hard']:.2f}, temp: {opt['temp']:.2f}, total loss: {total_loss.item():.2f}, {loss_str}"
            )

    return (
        batch,
        plots,
        loss_history,
        i_con_loss_history,
        con_loss_history,
        plddt_loss_history,
        distogram_history,
        sequence_history,
        traj_coords_list,
        traj_plddt_list,
    )

def _update_batches(data, data_apo):
    target = parse_boltz_schema(name, data, ccd_lib)
    target_apo = parse_boltz_schema(name, data_apo, ccd_lib)
    best_batch, best_structure = get_batch(
        target, msa_max_seqs, length, keep_record=True
    )
    best_batch_apo, best_structure_apo = get_batch(
        target_apo, msa_max_seqs, length, keep_record=True
    )
    best_batch = {
        key: value.unsqueeze(0).to(device) if key != "record" else value
        for key, value in best_batch.items()
    }
    best_batch_apo = {
        key: value.unsqueeze(0).to(device) if key != "record" else value
        for key, value in best_batch_apo.items()
    }
    return best_batch, best_batch_apo, best_structure, best_structure_apo


def _run_model(boltz_model, batch, predict_args):
    boltz_model.predict_args = predict_args
    return boltz_model.predict_step(batch, batch_idx=0, dataloader_idx=0)

def visualize_results(plots):
    # Plot distogram predictions
    if plots:
        num_plots = len(plots)
        num_rows = (num_plots + 5) // 6
        fig, axs = plt.subplots(num_rows, 6, figsize=(15, num_rows * 2.5))

        if num_rows == 1:
            axs = axs.reshape(1, -1)

        for i, plot_data in enumerate(plots):
            row, col = i // 6, i % 6
            axs[row, col].imshow(plot_data)
            axs[row, col].set_title(f"Epoch {i + 1}")
            axs[row, col].axis("off")

        # Hide unused subplots
        for j in range(num_plots, num_rows * 6):
            axs[j // 6, j % 6].axis("off")

        plt.tight_layout()
        plt.show()
        plots.clear()

# visualize_results(plots)