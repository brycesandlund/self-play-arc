import torch
from arcplay.catalogue_experiments import pool,episode
from arcplay.catalogue import Rule,OPS,probabilities,execute,CODE_SIZE
from arcplay.unified_solver import DemoEncoder,adapt,hard_code,grid_loss

def test_episode_queries_are_not_demonstrations():
    rep=pool(20,12,193,'cpu');ep=episode(rep,[Rule(1),Rule(8)],4,8)
    assert all(len(set(row.tolist()))==12 for row in ep['indices'])

def test_demo_order_invariance_and_number_of_examples():
    torch.manual_seed(92)
    rep=pool(20,12,193,'cpu');ep=episode(rep,[Rule(1),Rule(8)],4,1)
    model=DemoEncoder(True,16)
    a=model(ep['x'][:,:4],ep['y'][:,:4],ep['selected'][:,:4])
    ix=torch.tensor([3,0,2,1])
    b=model(ep['x'][:,ix],ep['y'][:,ix],ep['selected'][:,ix])
    assert torch.allclose(a,b,atol=1e-6)
    assert model(ep['x'][:,:1],ep['y'][:,:1],ep['selected'][:,:1]).shape==a.shape

def test_latent_adaptation_preserves_perfect_initial_support_fit():
    rep=pool(16,12,91,'cpu');rules=[Rule(OPS.index('recolor'),color=2),Rule(OPS.index('crop'))]
    ep=episode(rep,rules,3,2);initial=ep['codes']*8
    fitted,loss=adapt(initial,ep['support'],ep['y'][:,:3],steps=3,restarts=2)
    assert torch.equal(hard_code(fitted),hard_code(initial))
    assert loss.max()<1e-5

def test_direct_grid_solver_gradients_and_demo_order():
    from arcplay.grid_solver import GridSolver
    torch.manual_seed(11);rep=pool(16,12,100,'cpu');ep=episode(rep,[Rule(1),Rule(8)],3,1)
    model=GridSolver(width=16,depth=2)
    x,y,m=ep['x'],ep['y'],ep['selected']
    pred=model(x[:,:3],y[:,:3],m[:,:3],x[:,3],m[:,3])
    swapped=model(x[:,:3].flip(1),y[:,:3].flip(1),m[:,:3].flip(1),x[:,3],m[:,3])
    assert pred.shape==x[:,3].shape and torch.allclose(pred,swapped,atol=1e-6)
    loss=torch.nn.functional.cross_entropy(pred,y[:,3].argmax(1));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_two_stage_execution_and_gradient_path():
    from arcplay.composed_solver import render,StageEncoder
    rep=pool(8,12,209,'cpu')
    small={k:v[:2] for k,v in rep.items()}
    codes=torch.stack([torch.stack([Rule(1,src=0,color=2).code(),Rule(8,src=0).code()])]*2)
    target=render(small,codes)
    assert target.shape==small['x'].shape
    # Concentrated but not discrete routing permits both-stage gradients.
    logits=(codes*3).requires_grad_(True)
    pred=render(small,probabilities(logits),soft=True)
    loss=(pred-target).square().mean()
    gradient=torch.autograd.grad(loss,logits)[0]
    assert torch.isfinite(gradient).all()
    assert (gradient.abs().sum((0,2))>0).all()

def test_public_solver_without_training_or_query_answers():
    from arcplay.solve_catalogue import solve
    torch.manual_seed(9);rep=pool(16,12,321,'cpu');ep=episode(rep,[Rule(OPS.index('recolor'),src=0,color=2)],4,2)
    output,info=solve(ep['x'][0,:4],ep['y'][0,:4],ep['x'][0,4:],steps=100,restarts=2,seed=10)
    assert info['support_pixel_error']==0
    assert torch.equal(output.argmax(1),ep['y'][0,4:].argmax(1))
