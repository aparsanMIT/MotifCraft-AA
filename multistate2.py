from boltz.data.parse.schema import parse_boltz_schema
import torch
import copy
import numpy as np
import pickle
from utils.mydesign_utils import get_batch, run_model, Annealer, get_mid_points, get_con_loss, norm_seq_grad
import os

torch.set_float32_matmul_precision("highest")

with open(os.path.expanduser(os.path.join(os.environ["HOME"], ".boltz/ccd.pkl")), "rb") as f:
    CCD_LIB = pickle.load(f)


def revcomp(seq: str) -> str:
    complement = str.maketrans("ACGT", "TGCA")
    return seq.upper().translate(complement)[::-1]
    
def get_batch_with_ligands(seq, ligands=None, device="cuda"):
    data = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": seq,
                    "msa": "empty",
                }
            },
        ],
    }
    ALPHABET = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    for ligand in ligands:
        if ligand is not None:
            assert isinstance(ligand, tuple) and len(ligand) == 2, "ligand must be a (value, mol_type) tuple"
            ligand, mol_type = ligand
            if mol_type == "ligand":
                data["sequences"].append({
                    "ligand": {
                        "id": [ALPHABET[0]],
                        "smiles": ligand,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type in "rna":
                data["sequences"].append({
                    mol_type: {
                        "id": [ALPHABET[0]],
                        "sequence": ligand,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type in "dna":
                data["sequences"].append({
                    "dna": {
                        "id": [ALPHABET[0]],
                        "sequence": ligand,
                    }
                })
                data["sequences"].append({
                    "dna": {
                        "id": [ALPHABET[1]],
                        "sequence": revcomp(ligand),
                    }
                })
                ALPHABET = ALPHABET[2:]
        else:
            raise ValueError(f"Unsupported mol_type: {mol_type}")

    target = parse_boltz_schema(None, data, CCD_LIB)
    batch, structure = get_batch(target)
    batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
    # batch["msa"] = batch["res_type_logits"].unsqueeze(0).to(device)
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

    return batch, structure


class MotifLoss:
    def __init__(self, motif):
        self.motif = motif
        cb_pos = motif['cb_pos']
        dmat = np.square(cb_pos[None] - cb_pos[:,None]).sum(-1)**0.5
        motif_mask = motif['motif_mask']
        dmat[~motif_mask,:] = dmat[:,~motif_mask] = 0
        self.dmat = dmat
        
    def evaluate(self, dict_out, device, opt=None):
        pdist = dict_out['pdistogram']
        
        mid_pts = get_mid_points(pdist).to(device)
        motif_dmat = torch.from_numpy(self.dmat).to(device)
        pdist = pdist[:,:len(motif_dmat),:len(motif_dmat)]
        motif_dmat_mask = (motif_dmat > 1e-3) & (motif_dmat < 22)
        
        # edist = (pdist.softmax(-1) * mid_pts).sum(-1)
        # motif_edist_loss = (motif_dmat - edist)**2
        # motif_edist_loss = (motif_edist_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()
    
        motif_mse_loss = (pdist.softmax(-1) * (mid_pts - motif_dmat[...,None])**2).sum(-1)
        motif_mse_loss = (motif_mse_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()
    
        # motif_dmat_idx = (motif_dmat[...,None] - mid_pts).abs().argmin(-1)
        # motif_ce_loss = torch.nn.functional.cross_entropy(pdist.permute(0,3,1,2), motif_dmat_idx[None], reduction='none')
    
        # motif_ce_loss = (motif_ce_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()
    
        return motif_mse_loss


class AntiMotifLoss:
    def __init__(self, motif):
        self.motif = motif
        cb_pos = motif['cb_pos']
        dmat = np.square(cb_pos[None] - cb_pos[:,None]).sum(-1)**0.5
        motif_mask = motif['motif_mask']
        dmat[~motif_mask,:] = dmat[:,~motif_mask] = 0
        self.dmat = dmat
        
    def evaluate(self, dict_out, device, opt=None):
        pdist = dict_out['pdistogram']
        
        mid_pts = get_mid_points(pdist).to(device)
        motif_dmat = torch.from_numpy(self.dmat).to(device)
        pdist = pdist[:,:len(motif_dmat),:len(motif_dmat)]
        motif_dmat_mask = (motif_dmat > 1e-3) & (motif_dmat < 22)
    
        motif_mse_loss = (pdist.softmax(-1) * (mid_pts - motif_dmat[...,None])**2).sum(-1)
        motif_mse_loss = (motif_mse_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()
    
        return -0.5*motif_mse_loss

class ContactLoss:
    def __init__(self):
        pass
    def evaluate(self, dict_out, device, opt=None):
        chain_mask = dict_out['mol_type'] == 0
        pdist = dict_out['pdistogram']
        mid_pts = get_mid_points(pdist).to(device)
        #num_optimizing_binder_pos = 0 if pre_run else num_optimizing_binder_pos
        con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=1,
            seqsep=9,
            cutoff=14.,
            binary=False,
            mask_1d=chain_mask,
            mask_1b=chain_mask,
        )
        return con_loss        


class LigandContactLoss:
    def __init__(self, idx=1):
        self.idx = idx
        
    def evaluate(self, dict_out, device, opt=None):
        chain_mask = dict_out['mol_type'] == 0
        i_chain_mask = dict_out['mol_type'] == self.idx
        pdist = dict_out['pdistogram']
        mid_pts = get_mid_points(pdist).to(device)
        #num_optimizing_binder_pos = 0 if pre_run else num_optimizing_binder_pos
        i_con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=2,
            seqsep=0,
            num_pos=int(opt["num_optimizing_binder_pos"]),
            cutoff=20.,
            binary=False,
            mask_1d=chain_mask,
            mask_1b=i_chain_mask,
        )
        return i_con_loss


class MultistateDesigner:
    def __init__(self, num_states=1):
        self.ligands = [[]]*num_states
        self.motifs = []
        self.losses = []
        
    def add_ligand(self, ligand, state):
        if type(ligand) is list:
            self.ligands[state].extend(ligand)
        else:
            self.ligands[state].append(ligand)
            
    def add_motif(self, motif):
        self.motifs.append(motif)

    def add_loss(self, loss, state):
        self.losses.append((loss, state))
        
    def initialize(self, length, device='cuda'):
        self.device = device
        self.batches = []
        for i, ligs in enumerate(self.ligands):
            self.batches.append(get_batch_with_ligands('X'*length, ligs, device)[0])
            self.add_loss(ContactLoss(), state=i)
        
        z = torch.distributions.Gumbel(0, 1).sample((length, 33)).to(device)
        z[...,:2] = z[...,22:] = -np.inf
        self.logits = z.softmax(-1)
        
        alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
        self.fixed_mask = torch.zeros(length, dtype=bool, device=device)
        self.fixed_aa = torch.zeros_like(self.logits)
        
        for i, motif in enumerate(self.motifs):
            if motif is not None:
                motif_mask = torch.from_numpy(motif['motif_mask']).to(device)
                self.fixed_mask |= motif_mask
                motif_seq = [alphabet.index(c) for c in motif['motif_seq']]                
                motif_seq = torch.nn.functional.one_hot(
                    torch.tensor(motif_seq), num_classes=22
                )
                self.fixed_aa[motif_mask,:22] = motif_seq.to(device)[motif_mask].float()

        self.logits = torch.where(self.fixed_mask[...,None], self.fixed_aa, self.logits)

    def get_seq(self):
        alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
        return ''.join([alphabet[i.item()] for i in self.logits.argmax(-1)])

    def get_final_structs(self, boltz_model):
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 5,
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": True,
        }
        results = []
        for i, ligands in enumerate(self.ligands):
            new_batch, new_struct = get_batch_with_ligands(self.get_seq(), ligands)

            output = run_model(boltz_model, new_batch, predict_args)
            coords_all = output["coords"]

            struct_list = []
            for j in range(coords_all.shape[0]):
                struct_copy = copy.deepcopy(new_struct)
                struct_copy.atoms["coords"] = (
                    coords_all[j, : len(new_struct.atoms)].cpu().numpy()
                )
                struct_list.append(struct_copy)

            results.append((output, struct_list, i))
            
        return results
        
    def get_restype_from_logits(self, res_type_logits, opt, alpha=2.0):
        device = res_type_logits.device
        logits = alpha * res_type_logits
        X = logits - torch.sum(
            torch.eye(logits.shape[-1])[
                [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            ],
            dim=0,
        ).to(device) * (1e10)
        soft = torch.softmax(X / opt["temp"], dim=-1)  # probs
        hard = torch.zeros_like(soft).scatter_(
            -1, soft.max(dim=-1, keepdim=True)[1], 1.0
        )  # one hot
        hard = (hard - soft).detach() + soft # carries same grad
        pseudo = (
            opt["soft"] * soft + (1 - opt["soft"]) * res_type_logits
        )  # interp between probs and logits
        pseudo = (
            opt["hard"] * hard + (1 - opt["hard"]) * pseudo
        )  # interp between on hot and the above
    
        return pseudo

    def get_loss(self, restype, boltz_model, opt, verbose=False):
        total_loss = 0
        loss_dict = []

        boltz_out = []
        for batch in self.batches:
            batch['res_type'] = torch.cat([
                restype[None],
                batch['res_type'][:,len(restype):].detach()
            ], 1)
            batch["msa"] = batch["res_type"].unsqueeze(0).detach()
            batch["profile"] = batch["msa"].float().mean(dim=0).detach()

            dict_out = boltz_model.get_distogram(batch)[0]
            boltz_out.append(batch | dict_out)


        for loss, state in self.losses:
            if type(state) is list:
                readout = [boltz_out[s] for s in state]
            else:
                readout = boltz_out[state]
            this_loss = loss.evaluate(readout, boltz_model.device, opt)
            loss_dict.append((type(loss), state, this_loss.item()))
            total_loss = total_loss + this_loss

        if verbose:
            print(loss_dict)
            print(self.get_seq())
        return total_loss
        
            
    def do_iter(self, boltz_model, opt, pre_run=False, verbose=False):

        self.logits.requires_grad = True
        restype = self.get_restype_from_logits(self.logits, opt)
        restype = torch.where(self.fixed_mask[...,None].clone(), self.fixed_aa.clone(), restype.clone())
       
        loss = self.get_loss(restype, boltz_model, opt, verbose=verbose)
        # loss = restype.sum()
    
        loss.backward()
        if verbose: print('total_loss', loss)
        
        
        with torch.no_grad():
            self.logits.grad[self.fixed_mask] = 0
            self.logits.grad[
                ..., [0, 1, 6, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            ] = 0
            self.logits.grad = norm_seq_grad(
                self.logits.grad[None], 
                torch.ones_like(self.logits[:,0])
            )[0]
        
            self.logits -= opt["lr_rate"] * self.logits.grad
        self.logits.grad = None
        
    def optimize(self, boltz_model, verbose=False, debug=False):
        
        for opt in Annealer(hard=0, e_hard=0, iters=30, lr=0.2):
            self.do_iter(boltz_model, opt, pre_run=True, verbose=verbose)
        if debug: return
        
        with torch.no_grad():
            self.logits = self.get_restype_from_logits(self.logits, opt)
        
        
        for opt in Annealer(
            soft=0, 
            e_soft=1,
            hard=0,
            e_hard=0,
            e_num_optimizing_binder_pos=8,
            iters=100,
        ):
            self.do_iter(boltz_model, opt, verbose=verbose)

        with torch.no_grad():
            self.logits = 2 * self.logits
        
        for opt in Annealer(
            e_temp=0.01,
            hard=0,
            e_hard=0,
            num_optimizing_binder_pos=8,
            e_num_optimizing_binder_pos=12,
            iters=100,
        ):
            self.do_iter(boltz_model, opt, verbose=verbose)
        
        
        for opt in Annealer(
            temp=0.01,
            e_temp=0.01,
            num_optimizing_binder_pos=12,
            e_num_optimizing_binder_pos=16,
            iters=10,
        ):
            self.do_iter(boltz_model, opt, verbose=verbose)