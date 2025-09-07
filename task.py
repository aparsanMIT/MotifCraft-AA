from mydesign import MultistateDesigner

# common design tasks

TASK_REGISTRY = {}

def task(name):
    def decorator(fn):
        TASK_REGISTRY[name] = fn
        return fn
    return decorator


@task('onemotif_twostates_neg')
def onemotif_twostates_neg(motif, ligand, length):
    """
    Sequence folds into two structure states with negative allostery: 
    - State 0: motif active when ligand unbound
    - State 1: motif inactive  when ligand bound
    """
    
    designer = MultistateDesigner(num_states=2)
    designer.add_motif(motif, state=0)
    designer.add_anti_motif(motif, state=1)
    designer.add_ligand(ligand, state=1)
    designer.initialize(length=length)
    
    return designer

@task('onemotif_twostates_pos')
def onemotif_twostates_pos(motif, ligand, length):
    """
    Sequence folds into two structure states with positive allostery: 
    - State 0: motif inactive when ligand unbound
    - State 1: motif active when ligand bound
    """
    
    designer = MultistateDesigner(num_states=2)
    designer.add_anti_motif(motif, state=0)
    designer.add_motif(motif, state=1)
    designer.add_ligand(ligand, state=1)
    designer.initialize(length=length)
    
    return designer