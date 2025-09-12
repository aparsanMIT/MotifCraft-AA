from multistate import MultistateDesigner

# common design tasks

TASK_REGISTRY = {}

def task(name):
    def decorator(fn):
        TASK_REGISTRY[name] = fn
        return fn
    return decorator


@task('onemotif_twostates_neg')
def onemotif_twostates_neg(motifs, ligands, length):
    """
    Sequence folds into two structure states with negative allostery: 
    - State 0: motif active when ligand unbound
    - State 1: motif inactive  when ligand bound
    """
    
    designer = MultistateDesigner(num_states=2)
    designer.add_motif(motifs[0], state=0)
    designer.add_anti_motif(motifs[0], state=1)
    designer.add_ligand(ligands[0], state=1)
    designer.initialize(length=length)
    
    return designer

@task('onemotif_twostates_pos')
def onemotif_twostates_pos(motifs, ligands, length):
    """
    Sequence folds into two structure states with positive allostery: 
    - State 0: motif inactive when ligand unbound
    - State 1: motif active when ligand bound
    """
    
    designer = MultistateDesigner(num_states=2)
    designer.add_anti_motif(motifs[0], state=0)
    designer.add_motif(motifs[0], state=1)
    designer.add_ligand(ligands[0], state=1)
    designer.initialize(length=length)
    
    return designer


@task('twomotif_twostates')
def twomotif_twostates(motifs, ligands, length):
    """
    Sequence folds into two structure states switching between two motifs: 
    - State 0: motif B inactive and motif A active when ligand unbound
    - State 1: motif B active and motif A inactive when ligand bound
    """
    motifA,motifB = motifs[0],motifs[1]

    designer = MultistateDesigner(num_states=2)
    designer.add_anti_motif(motifB, state=0)
    designer.add_motif(motifA, state=0)
    if len(ligands) == 2:
        designer.add_ligand(ligands[1], state=0)
    
    designer.add_anti_motif(motifA, state=1)
    designer.add_motif(motifB, state=1)
    designer.add_ligand(ligands[0], state=1)
    
    designer.initialize(length=length)
    
    return designer