from boltz.data.parse.schema import parse_boltz_schema
import torch
import copy
import numpy as np
import pickle
from typing import Optional
from utils.mydesign_utils import get_batch, run_model, Annealer, get_mid_points, get_con_loss, norm_seq_grad
from boltz.data import const
from utils import residue_constants
import os

torch.set_float32_matmul_precision("highest")
#os.environ["HOME"] = "/data/cb/scratch/aparsan/BoltzDesign1"

with open(os.path.expanduser(os.path.join(os.environ["HOME"], "boltz/ccd.pkl")), "rb") as f:
    CCD_LIB = pickle.load(f)


def revcomp(seq: str) -> str:
    complement = str.maketrans("ACGT", "TGCA")
    return seq.upper().translate(complement)[::-1]
    
def get_batch_with_ligands(seq, ligands=None, device="cuda", atomize_positions=None):
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

    # add the residues we need to atomize in the schema 
    if atomize_positions is not None:
        data["sequences"][0]["protein"]["atomize_positions"] = [
            int(pos) + 1 for pos in atomize_positions
        ]
    
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
            elif mol_type == "ccd":
                data["sequences"].append({
                    "ligand": {
                        "id": [ALPHABET[0]],
                        "ccd": ligand,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type == "protein":
                data["sequences"].append({
                    "protein": {
                        "id": [ALPHABET[0]],
                        "sequence": ligand,
                        "msa": "empty",
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


class AllAtomMotifLoss:
    """
    All-atom analogue of MotifLoss.

    This loss assumes that the motif dictionary contains:
      - 'atom_pos': np.ndarray of shape [N_atoms, 3]
      - 'atom_dmat': np.ndarray of shape [N_atoms, N_atoms]
      - 'atom_token_indices': a 1D sequence of length N_atoms giving the token
        indices (in the Boltz distogram) corresponding to each motif atom.

    Computing 'atom_token_indices' requires mapping atoms in the motif to
    Boltz tokens and is expected to be provided upstream.
    """

    def __init__(self, motif, max_dist: float = 22.0):
        # Store motif by reference and lazily initialize when atom_token_indices
        # have been populated (after Boltz batches are built).
        self.motif = motif
        self.max_dist = max_dist

        self._initialized = False
        self.atom_dmat = None
        self.atom_dmat_mask = None
        self.atom_token_indices = None

    def _ensure_initialized(self):
        if self._initialized:
            return

        atom_dmat = self.motif.get("atom_dmat", None)
        atom_token_indices = self.motif.get("atom_token_indices", None)

        if atom_dmat is None or atom_token_indices is None:
            raise ValueError(
                "AllAtomMotifLoss requires motif['atom_dmat'] and motif['atom_token_indices'] "
                "to be set before evaluation. Make sure atom_token_indices are "
                "computed after building Boltz batches."
            )

        self.atom_dmat = atom_dmat.astype(np.float32)
        self.atom_dmat_mask = (self.atom_dmat > 1e-3) & (self.atom_dmat < self.max_dist)
        self.atom_token_indices = np.asarray(atom_token_indices, dtype=np.int64)
        self._initialized = True

    def evaluate(self, dict_out, device, opt=None):
        # Lazily pull atom_dmat / atom_token_indices from the motif dict
        self._ensure_initialized()
        pdist = dict_out["pdistogram"]  # [B, L, L, num_bins]
        print("shape of pdist", pdist.shape)

        idx = torch.as_tensor(self.atom_token_indices, dtype=torch.long, device=device)
        # Subselect distogram to atom tokens: [B, N_atoms, N_atoms, num_bins]
        pdist_atom = pdist[:, idx][:, :, idx]

        print(f"[all-atom] token idx min/max={idx.min().item()}/{idx.max().item()}, count={idx.numel()}")
        print(f"[all-atom] pdist slice shape={tuple(pdist_atom.shape)} vs full {tuple(pdist.shape)}")

        mid_pts = get_mid_points(pdist_atom).to(device)  # [num_bins]

        atom_dmat = torch.from_numpy(self.atom_dmat).to(device)  # [N_atoms, N_atoms]
        atom_mask = torch.from_numpy(self.atom_dmat_mask).to(device)  # [N_atoms, N_atoms]

        probs = pdist_atom.softmax(-1)
        sq_err = (mid_pts - atom_dmat[..., None]) ** 2
        atom_mse = (probs * sq_err).sum(-1)  # [B, N_atoms, N_atoms]

        atom_mse = atom_mse * atom_mask
        loss_per_batch = atom_mse.sum(dim=(-2, -1)) / (atom_mask.sum() + 1e-8)

        return loss_per_batch.mean()

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


class FilteredContactLoss(ContactLoss):
    """
    Contact loss variant that treats atomized residues as single units by using
    only their CB (or CA for GLY) token. This ensures atomized residues participate
    in contact loss the same way as non-atomized residues (one token per residue).
    
    Requires motif['atom_token_indices'] and motif['residue_cb_token_indices']
    to be populated (computed by _assign_atom_token_indices).
    """

    def __init__(self, motif):
        super().__init__()
        self.motif = motif

    def evaluate(self, dict_out, device, opt=None):
        atom_token_indices = self.motif.get("atom_token_indices")
        cb_token_indices = self.motif.get("residue_cb_token_indices")
        
        if atom_token_indices is None or cb_token_indices is None:
            raise ValueError(
                "FilteredContactLoss requires motif['atom_token_indices'] and "
                "motif['residue_cb_token_indices'] to be set."
            )

        chain_mask = dict_out["mol_type"] == 0
        filtered_mask = chain_mask.clone()
        
        # Step 1: Mask out ALL atomized tokens
        atom_idx = torch.as_tensor(atom_token_indices, dtype=torch.long, device=device)
        atom_idx = atom_idx[(atom_idx >= 0) & (atom_idx < filtered_mask.shape[-1])]
        filtered_mask[..., atom_idx] = False
        
        # Step 2: Re-include CB tokens for atomized residues
        # This treats each atomized residue as a single unit (its CB atom)
        cb_idx = torch.as_tensor(cb_token_indices, dtype=torch.long, device=device)
        cb_idx = cb_idx[(cb_idx >= 0) & (cb_idx < filtered_mask.shape[-1])]
        filtered_mask[..., cb_idx] = True

        pdist = dict_out["pdistogram"]
        mid_pts = get_mid_points(pdist).to(device)
        con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=1,
            seqsep=9,
            cutoff=14.0,
            binary=False,
            mask_1d=filtered_mask,
            mask_1b=filtered_mask,
        )
        return con_loss

class DifferenceLoss:
    def __init__(self, strength):
        self.strength = strength
        
    def evaluate(self, dict_out, device, opt=None):
        
        chain_mask = dict_out[0]['mol_type'] == 0
        pdist0 = dict_out[0]['pdistogram'].softmax(dim=-1)[:,chain_mask[0]][:,:,chain_mask[0]]

        chain_mask = dict_out[1]['mol_type'] == 0
        pdist1 = dict_out[1]['pdistogram'].softmax(dim=-1)[:,chain_mask[0]][:,:,chain_mask[0]]
        
        m = (pdist0+pdist1)/2
        jsd = (pdist0 * (pdist0.log() - m.log())).sum(-1)/2 + (pdist1 * (pdist1.log() - m.log())).sum(-1) / 2
        return -self.strength * jsd.max(-1).values.mean()
        
class LigandContactLoss:
    def __init__(self, idx=None):
        self.idx = idx
        
    def evaluate(self, dict_out, device, opt=None):
        chain_mask = dict_out['asym_id'] == 0
        if self.idx is None:
            i_chain_mask = dict_out['asym_id'] != 0
        else:
            i_chain_mask = dict_out['asym_id'] == self.idx
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

        
class AntiLigandContactLoss:
    def __init__(self, strength=0, idx=None):
        self.idx = idx
        self.strength = strength
        print('AntiLigandContactLoss', self.strength)
    def evaluate(self, dict_out, device, opt=None):
        
        chain_mask = dict_out['asym_id'] == 0
        if self.idx is None:
            i_chain_mask = dict_out['asym_id'] != 0
        else:
            i_chain_mask = dict_out['asym_id'] == self.idx
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
        
        return -self.strength * i_con_loss
class MultistateDesigner:
    def __init__(self, num_states=1):
        self.ligands = [[] for _ in range(num_states)]
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
        
    def initialize(self, length, device='cuda', atomize_motif: bool = False):
        self.device = device
        self.batches = []
        self.structures = []
        alphabet = list("XXARNDCQEGHILKMFPSTWYV-")

        z = torch.distributions.Gumbel(0, 1).sample((length, 33)).to(device)
        z[...,:2] = z[...,22:] = -np.inf
        self.logits = z.softmax(-1)
        

        self.fixed_mask = torch.zeros(length, dtype=bool, device=device)
        self.fixed_aa = torch.zeros_like(self.logits)
        
        # Use a mutable list for building the starting sequence.
        start_seq = ['X'] * length

        for i, motif in enumerate(self.motifs):
            if motif is not None:
                motif_mask = torch.from_numpy(motif['motif_mask']).to(device)
                
                self.fixed_mask |= motif_mask
                motif_seq = [alphabet.index(c) for c in motif['motif_seq']]     
                motif_seq = torch.nn.functional.one_hot(
                    torch.tensor(motif_seq), num_classes=22
                )
                self.fixed_aa[motif_mask,:22] = motif_seq.to(device)[motif_mask].float()
                for j in range(length):
                    if motif['motif_mask'][j]:
                        start_seq[j] = motif['motif_seq'][j]

        # TO DO: get this to work on multiple motifs lol  
        if atomize_motif and self.motifs:
            motif = self.motifs[0]
            motif_mask = motif["motif_mask"]          # boolean array over length
            atomize_positions = np.where(motif_mask)[0].tolist()  # 0-based indices
            print("atomizing positions", atomize_positions)
        else:
            atomize_positions = None

        

        self.logits = torch.where(self.fixed_mask[...,None], self.fixed_aa, self.logits)

        for i, ligs in enumerate(self.ligands):
            batch, structure = get_batch_with_ligands(
                ''.join(start_seq),
                ligs,
                device,
                atomize_positions=atomize_positions,
            )
            self.batches.append(batch)
            self.structures.append(structure)
            if atomize_motif and self.motifs:
                self.add_loss(FilteredContactLoss(self.motifs[0]), state=i)
                #self.add_loss(ContactLoss(), state=i)
            else:
                self.add_loss(ContactLoss(), state=i)

        # Only map motif atoms to distogram tokens when using all-atom motif loss.
        if atomize_motif and self.motifs:
            self._assign_atom_token_indices(self.motifs[0], self.structures[0], self.batches[0])

    def _assign_atom_token_indices(self, motif, structure, batch):
        """
        Populate motif['atom_token_indices'] by mapping each motif atom
        (given by motif['atom_res_index'], motif['atom_name']) to the
        corresponding Boltz token index via structure + batch['atom_to_token'].
        """
        atom_res_index = motif.get("atom_res_index", None)
        atom_names = motif.get("atom_name", None)
        if atom_res_index is None or atom_names is None:
            raise ValueError("Motif must contain 'atom_res_index' and 'atom_name'.")

        # Locate chain A in the Structure to get its residue range.
        chains = structure.chains
        chain_A_idx = None
        for i, chain in enumerate(chains):
            if chain["name"] == "A":
                chain_A_idx = i
                break
        if chain_A_idx is None:
            raise ValueError("Chain 'A' not found in structure for motif mapping.")

        res_start = chains[chain_A_idx]["res_idx"]
        res_num = chains[chain_A_idx]["res_num"]
        res_end = res_start + res_num

        residues = structure.residues[res_start:res_end]

        # Map from sequence index -> (atom_start, res_name)
        res_by_seq_idx = {}
        for res in residues:
            seq_idx = int(res["res_idx"])
            atom_start = int(res["atom_idx"])
            atom_num = int(res["atom_num"])
            res_name = str(res["name"])
            res_by_seq_idx[seq_idx] = (atom_start, atom_num, res_name)

        atom_token_indices = []
        atom_to_token = batch["atom_to_token"][0]  # [num_atoms, num_tokens]

        for seq_idx, a_name in zip(atom_res_index, atom_names):
            seq_idx = int(seq_idx)
            a_name = str(a_name)

            if seq_idx not in res_by_seq_idx:
                raise ValueError(f"Residue index {seq_idx} not found in structure.")

            atom_start, atom_num, res_name = res_by_seq_idx[seq_idx]

            # Use Boltz ref_atoms ordering to locate this atom within the residue.
            if res_name not in const.ref_atoms:
                raise ValueError(f"Residue {res_name} not in const.ref_atoms.")

            try:
                local_atom_idx = const.ref_atoms[res_name].index(a_name)
            except ValueError:
                raise ValueError(
                    f"Atom {a_name} not found in ref_atoms list for residue {res_name}."
                )

            if local_atom_idx >= atom_num:
                raise ValueError(
                    f"Local atom index {local_atom_idx} out of range for residue {res_name}."
                )

            global_atom_idx = atom_start + local_atom_idx

            # Map global atom index to token index via atom_to_token.
            token_vec = atom_to_token[global_atom_idx]  # [num_tokens]
            nonzero = (token_vec > 0).nonzero(as_tuple=True)[0]
            if nonzero.numel() == 0:
                raise ValueError(f"No token mapping found for atom index {global_atom_idx}.")

            token_idx = int(nonzero[0].item())
            atom_token_indices.append(token_idx)

        motif["atom_token_indices"] = np.asarray(atom_token_indices, dtype=np.int64)
        print("shape of atom_token_indices", motif["atom_token_indices"].shape)
        print("atom_token_indices of first 50 atoms", motif["atom_token_indices"][:50])
        print("atom_token_indices of last 50 atoms", motif["atom_token_indices"][-50:])

        # Also compute CB (or CA for GLY) token indices - one per atomized residue.
        # This is used by FilteredContactLoss to treat atomized residues as single units.
        residue_cb_token_indices = []
        unique_res_indices = sorted(set(int(x) for x in atom_res_index))
        
        for res_idx in unique_res_indices:
            # Get the residue name from structure
            if res_idx not in res_by_seq_idx:
                continue
            atom_start, atom_num, res_name = res_by_seq_idx[res_idx]
            
            # Determine the distogram atom (CB for most, CA for GLY)
            disto_atom = const.res_to_disto_atom.get(res_name, "CB")
            
            try:
                local_atom_idx = const.ref_atoms[res_name].index(disto_atom)
            except ValueError:
                # Fallback to CA if CB not found
                local_atom_idx = const.ref_atoms[res_name].index("CA")
            
            if local_atom_idx >= atom_num:
                continue
                
            global_atom_idx = atom_start + local_atom_idx
            token_vec = atom_to_token[global_atom_idx]
            nonzero = (token_vec > 0).nonzero(as_tuple=True)[0]
            if nonzero.numel() > 0:
                residue_cb_token_indices.append(int(nonzero[0].item()))
        
        motif["residue_cb_token_indices"] = np.asarray(residue_cb_token_indices, dtype=np.int64)
        print("shape of residue_cb_token_indices", motif["residue_cb_token_indices"].shape)
        print("residue_cb_token_indices", motif["residue_cb_token_indices"])

    def get_seq(self):
        alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
        return ''.join([alphabet[i.item()] for i in self.logits.argmax(-1)])

    def get_final_structs(self, boltz_model, samples: Optional[int] = 5, set_seq: Optional[str] = None):
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": samples,
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": True,
        }
        results = []
        for i, ligands in enumerate(self.ligands):
            seq = self.get_seq() if set_seq is None else set_seq
            new_batch, new_struct = get_batch_with_ligands(seq, ligands)

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
                [0, 1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
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
                ..., [0, 1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            ] = 0
            self.logits.grad = norm_seq_grad(
                self.logits.grad[None], 
                torch.ones_like(self.logits[:,0])
            )[0]
        
            self.logits -= opt["lr_rate"] * self.logits.grad
        self.logits.grad = None

        return loss.item()
        
    def optimize(self, boltz_model, verbose=False, debug=False, best_by_loss=False):

        best_loss = float('inf')
        best_logits = None

        def update_best(loss):
            nonlocal best_loss, best_logits
            if loss < best_loss:
                best_loss = loss
                best_logits = self.logits.clone().detach()

        for opt in Annealer(hard=0, e_hard=0, iters=30, lr=0.2):
            loss_val = self.do_iter(boltz_model, opt, pre_run=True, verbose=verbose)
            if best_by_loss:
                update_best(loss_val)
        if debug: return
        
        with torch.no_grad():
            self.logits = self.get_restype_from_logits(self.logits, opt)
        
        
        for opt in Annealer(
            soft=0,
            e_soft=1,
            hard=0,
            e_hard=0,
            e_num_optimizing_binder_pos=8,
            iters=100, # 100
        ):
            loss_val = self.do_iter(boltz_model, opt, verbose=verbose)
            if best_by_loss:
                update_best(loss_val)

        with torch.no_grad():
            self.logits = 2 * self.logits
        
        for opt in Annealer(
            e_temp=0.01,
            hard=0,
            e_hard=0,
            num_optimizing_binder_pos=8,
            e_num_optimizing_binder_pos=12,
            iters=100, # 100
        ):
            loss_val = self.do_iter(boltz_model, opt, verbose=verbose)
            if best_by_loss:
                update_best(loss_val)
        
        
        for opt in Annealer(
            temp=0.01,
            e_temp=0.01,
            num_optimizing_binder_pos=12,
            e_num_optimizing_binder_pos=16,
            iters=2, # 10
        ):
            loss_val = self.do_iter(boltz_model, opt, verbose=verbose)
            if best_by_loss:
                update_best(loss_val)

        if best_by_loss and best_logits is not None:
            self.logits = best_logits