import numpy as np
from . import protein, residue_constants
from copy import deepcopy
from boltz.data import const  

def get_motif_scaffold_templates(paths):
    
    specs = [load_motif_spec(path) for path in paths]
    # print(specs[0])
    spec = merge_motif_specs(specs)			# merge all motifs into one spec for easy sampling
    masks = sample_motif_mask(spec)
    full_motif_mask = masks['sequence']	    # does not separate motifs, just boolean mask for scaffold/motif
    motif_groups = masks['group']		    # this separates motifs by group index (0 is scaffold, 1 is first motif, ... so on)
    motif_templates = []
    # length_to_add = 80 - len(full_motif_mask)
    # full_motif_mask = np.append(full_motif_mask,[False]*length_to_add)
    # motif_groups = np.append(motif_groups,[0]*length_to_add)

    for i,path in enumerate(paths):	# for each motif
        motif_idx = i+1
        motif_mask = motif_groups == motif_idx
        with open(path) as f:
            prot = protein.from_pdb_string(f.read())

        # Residue-level  representation used by the original motif loss
        cb_pos = np.zeros((len(motif_mask), 3), dtype=np.float32)
        cb_pos[motif_mask] = prot.atom_positions[:, 2]	# glycine Cβs are infilled by featurizer

        seq = ['X'] * len(motif_mask)
        motif_positions = motif_mask.nonzero()[0]
        for design_idx, aatype in zip(motif_positions, prot.aatype):
            seq[design_idx] = residue_constants.restypes[aatype]

        # all-atom rep for motif residues
        atom_pos = []
        atom_res_index = []
        atom_name = []

        # prot.aatype and prot.atom_positions are ordered only over motif residues.
        # We map them back onto design indices via motif_positions.
        for local_res_idx, (design_idx, aatype) in enumerate(
            zip(motif_positions, prot.aatype)
        ):
            restype_1 = residue_constants.restypes[aatype]
            restype_3 = residue_constants.restype_1to3[restype_1]
            # Use Boltz ref_atoms so motif atoms align with atomized residues
            if restype_3 not in const.ref_atoms:
                continue

            for a_name in const.ref_atoms[restype_3]:
                # Map atom name to canonical atom index in OpenFold ordering
                atom_idx = residue_constants.atom_order.get(a_name)
                if atom_idx is None:
                    continue

                coord = prot.atom_positions[local_res_idx, atom_idx]
                # Skip atoms with no coordinates (all zeros)
                if np.allclose(coord, 0.0):
                    continue
                
                atom_pos.append(coord)
                atom_res_index.append(design_idx)
                atom_name.append(a_name)

        atom_pos = np.asarray(atom_pos, dtype=np.float32)
        atom_res_index = np.asarray(atom_res_index, dtype=np.int32)
        atom_name = np.asarray(atom_name, dtype=object)

        # Precompute all-atom distance matrix for motif atoms (used by all-atom motif losses)
        if atom_pos.size > 0:
            atom_dmat = np.linalg.norm(
                atom_pos[None, ...] - atom_pos[:, None, :],
                axis=-1,
            ).astype(np.float32)
        else:
            atom_dmat = np.zeros((0, 0), dtype=np.float32)

        motif_templates.append(
            {
                'full_motif_mask': full_motif_mask,  # saving for now
                'motif_mask': motif_mask,
                'cb_pos': cb_pos,
                'motif_seq': ''.join(seq),
                # All-atom motif description (used for all-atom motif losses)
                'atom_pos': atom_pos,
                'atom_res_index': atom_res_index,
                'atom_name': atom_name,
                'atom_dmat': atom_dmat,
            }
        )

        print("shape of atom_pos", motif_templates[i]['atom_pos'].shape)
        print("atom_res_index of first 50 atoms", motif_templates[i]['atom_res_index'][:50])
        print("atom_res_index of last 50 atoms", motif_templates[i]['atom_res_index'][-50:])
        print("atom_name of first 50 atoms", motif_templates[i]['atom_name'][:50])
        print("atom_name of last 50 atoms", motif_templates[i]['atom_name'][-50:])
        print("shape of atom_dmat", motif_templates[i]['atom_dmat'].shape)
        
    # breakpoint()
    
    return motif_templates
        

def merge_motif_specs(specs):
    
    def recompute_total_lengths(structures):
        min_total = 0
        max_total = 0
        for s in structures:
            if s["type"] == "motif":
                L = s["end_index"] - s["start_index"] + 1
                min_total += L
                max_total += L
            elif  s["type"] == "scaffold":
                min_total += s["min_length"]
                max_total += s["max_length"]
        
        return min_total, max_total

    def merge_two(spec1, spec2):
        trailing = spec1['structures'][-1]
        leading = spec2['structures'][0]
        
        assert trailing["type"] == "scaffold" and leading["type"] == "scaffold", "only valid merge when leading/trailing are scaffold segments"
        
        min_len = max(trailing["min_length"], leading["min_length"])
        max_len = min(trailing["max_length"], leading["max_length"])
        if min_len > max_len:	# edge case
            max_len = max(trailing["max_length"], leading["max_length"])
   
        new_pad = {'type': 'scaffold','min_length': min_len, 'max_length': max_len}
        
        # getting last motif group identifier
        last_motif_group = None
        for struc in spec1['structures']:
            if struc["type"]=="motif":
                last_motif_group = struc["group"]

        # setting new motif group identifiers to last + 1
        next_motif_group = chr(ord(last_motif_group) + 1)
        spec2_structs = deepcopy(spec2["structures"])
        for struc in spec2_structs[1:]:
            if struc["type"]=="motif":
                struc["group"]=next_motif_group
    
        new_scaffold_template= spec1['structures'][:-1] + [new_pad] + spec2_structs[1:]
        
        min_total,max_total = recompute_total_lengths(new_scaffold_template)
        return {
            'name': spec1['name'] + '+' + spec2['name'],
            'structures': new_scaffold_template,
            'min_total_length': min_total,
            'max_total_length': max_total
        }
        
    merged = specs[0]
    for i in range(len(specs)-1):
        merged = merge_two(merged, specs[i+1])
    return merged


def load_motif_spec(filepath):
    """
    Load motif specification file.

    Args:
        filepath:
            Path to the PDB file for motif specification.

    Returns:
        A dictionary of motif specifications containing
            -	name:
                Name of the motif scaffolding problem
            -	structures:
                A list of dictionaries, each of which defines either 
                    -	a motif segment, containing information on the chain and 
                        residue index range that the motif structure is coming 
                        from, as well as the motif group that this segment belongs
                    -	a scaffold segment, containing information on the maximum 
                        and minimum number of residues for the segment
            -	min_total_length:
                Minimum number of residues for the generated structure
            -	max_total_length:
                Maximum number of residues for the generated structure.
    """
    with open(filepath) as file:
        structures = []
        for line in file:
            if line.startswith('REMARK 999 INPUT'):
                if line[18] == ' ':
                    structures.append({
                        'type': 'scaffold',
                        'min_length': int(line[19:23]),
                        'max_length': int(line[23:27])
                    })
                else:
                    structures.append({
                        'type': 'motif',
                        'chain': line[18],
                        'start_index': int(line[19:23]),
                        'end_index': int(line[23:27]),
                        'group': line[28] if len(line) > 28 and line[28] != ' ' else 'A'
                    })
            if line.startswith('REMARK 999 NAME'):
                name = line[18:]
            if line.startswith('REMARK 999 MINIMUM TOTAL LENGTH'):
                min_total_length = int(line[37:])
            if line.startswith('REMARK 999 MAXIMUM TOTAL LENGTH'):
                max_total_length = int(line[37:])
    return {
        'name': name,
        'structures': structures,
        'min_total_length': min_total_length,
        'max_total_length': max_total_length
    }

def sample_motif_mask(spec):
    """
    Sample a motif configuration from a dictionary of specifications.

    Args:
        spec:
            A dictionary of motif specifications containing
                -	name:
                    Name of the motif scaffolding problem
                -	structures:
                    A list of dictionaries, each of which defines either 
                        -	a motif segment, containing information on the chain and 
                            residue index range that the motif structure is coming 
                            from, as well as the motif group that this segment belongs
                        -	a scaffold segment, containing information on the maximum 
                            and minimum number of residues for the segment
                -	min_total_length:
                    Minimum number of residues for the generated structure
                -	max_total_length:
                    Maximum number of residues for the generated structure.

    Returns:
        A dictionary of masks including
            -	sequence:
                A residue-level mask to indicate which residue contains conditional 
                sequence information
            -	structure: 
                A pair residue-residue mask to indicate which pair of residues contains
                conditional structural information
            -	group:
                Residue-level group indices to indicate which group each residue belongs to 
                (0 indicates scaffold and each positive integer indicates a motif group).
    """
    success = False
    while not success:

        # Define
        total_length = 0
        motif_sequence_mask = []
        motif_groups = []

        # Generate
        for structure in spec['structures']:
            if structure['type'] == 'scaffold':
                scaffold_length = np.random.randint(structure['min_length'], structure['max_length'] + 1)
                motif_sequence_mask.extend([0] * scaffold_length)
                motif_groups.extend([0] * scaffold_length)
                total_length += scaffold_length
            else:
                motif_length = structure['end_index'] - structure['start_index'] + 1
                motif_sequence_mask.extend([1] * motif_length)
                motif_groups.extend([ord(structure['group']) - ord('A') + 1] * motif_length)
                total_length += motif_length

        # Validate
        if total_length >= spec['min_total_length'] and \
            total_length <= spec['max_total_length']:
            success = True

    # Create motif structure mask
    motif_structure_mask = np.zeros((total_length, total_length))
    num_groups = np.max(motif_groups)
    for i in range(1, 1 + num_groups):
        motif_group_sequence_mask = np.equal(motif_groups, i)
        motif_structure_mask += motif_group_sequence_mask[:, np.newaxis] * motif_group_sequence_mask[np.newaxis, :]

    return {
        'sequence': np.array(motif_sequence_mask).astype(bool),
        'structure': np.array(motif_structure_mask).astype(bool),
        'group': np.array(motif_groups).astype(int)
    }

def save_motif_pdb(spec_filepath, mask, pdb_filepath):
    """
    Save motif information as a PDB file.

    Args:
        spec_filepath:
            Path to motif specification file.
        mask:
            A residue-level mask to indicate which residue is a motif residue
        pdb_filepath:
            Output PDB filepath.
    """

    def pad_left(string, length):
        assert len(string) <= length
        return ' ' * (length - len(string)) + string

    # Parse residue index in motif spec file
    spec = load_motif_spec(spec_filepath)
    residue_index_spec = []
    for structure in spec['structures']:
        if structure['type'] == 'motif':
            for i in range(structure['start_index'], structure['end_index'] + 1):
                residue_index_spec.append((
                    structure['chain'],
                    i,
                    structure['group']
                ))

    # Parse residue index in motif pdb file
    residue_index_pdb = [i + 1 for i, elt in enumerate(mask) if elt]
    assert len(residue_index_pdb) == len(residue_index_spec)

    # Create residue index map
    residue_index_map = dict([
        (
            '{}_{}'.format(elt[0], elt[1]),
            (residue_index_pdb[i], elt[2])
        )
        for i, elt in enumerate(residue_index_spec)
    ])

    # Parse records in motif spec file
    with open(spec_filepath) as file:
        lines = [line for line in file if line.startswith('ATOM')]

    # Update residue index
    updated_lines = []
    for i, line in enumerate(lines):
        chain = line[21]
        residue_index = int(line[22:26])
        key = '{}_{}'.format(chain, residue_index)
        updated_residue_index = residue_index_map[key][0]
        updated_group = residue_index_map[key][1]
        updated_line = line[:21] + 'A' + str(updated_residue_index).rjust(4) + line[26:72] + updated_group.ljust(4) + line[76:]
        updated_lines.append(updated_line)

    # Save
    with open(pdb_filepath, 'w') as file:
        file.write(''.join(updated_lines))


