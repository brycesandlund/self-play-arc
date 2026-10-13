import pytest
import torch
from torch.nn import functional as F

from arcplay.object_lab import (CommunicationNet, make_object, region_masks,
                                representation, scenes)


def two_objects():
    labels=torch.zeros(1,8,8,dtype=torch.long)
    labels[:,1:3,1:3]=2
    labels[:,5,4:7]=2
    return F.one_hot(labels,11).movedim(-1,1).double()


def test_separates_same_color_and_corrects_seed_multiplicity():
    x=two_objects()
    rep=representation(x,steps=16)
    model=make_object(1,steps=16).double()
    with torch.no_grad(): model.score.zero_()
    mask,_=model.selection(rep)
    assert torch.allclose(mask[:,0][x[:,2]>0],torch.full((7,),.5,dtype=x.dtype))
    with torch.no_grad(): model.score[0]=2
    mask,_=model.selection(rep)
    assert mask[0,0,1:3,1:3].min()>.999
    assert mask[0,0,5,4:7].max()<.001
    with torch.no_grad(): model.score[0]=-2
    mask,_=model.selection(rep)
    assert mask[0,0,5,4:7].min()>.999
    assert mask[0,0,1:3,1:3].max()<.001
    color,_=region_masks(x,"color")
    assert color[0,1].sum()==7


def test_empty_board_stays_valid_with_extreme_scores():
    x=torch.zeros(2,11,6,6);x[:,0]=1
    model=make_object(0)
    with torch.no_grad(): model.score.mul_(100)
    assert torch.allclose(model(x),x)


def test_ray_scan_leaks_across_a_gap_but_local_does_not():
    labels=torch.zeros(1,5,5,dtype=torch.long)
    labels[0,2,1]=labels[0,2,3]=2
    x=F.one_hot(labels,11).movedim(-1,1).float()
    local,_=region_masks(x,"local",10)
    ray,_=region_masks(x,"ray")
    assert local[0,11,2,3]==0
    assert ray[0,11,2,3]==1


@pytest.mark.parametrize("action",["paint","move"])
def test_soft_board_probabilities_and_higher_derivatives(action):
    torch.set_num_threads(2)
    model=make_object(7,temperature=.4,action=action,steps=4).double()
    x=two_objects()*.97+.03/11
    x.requires_grad_()
    y=model(x)
    assert y.min()>=0
    assert torch.allclose(y.sum(1),torch.ones_like(y[:,0]),atol=1e-10)
    loss=(y*torch.randn_like(y)).sum()
    grads=torch.autograd.grad(loss,(x,*model.parameters()),create_graph=True,allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert grads[0].abs().sum()>0
    second=torch.autograd.grad(sum(g.square().sum() for g in grads if g is not None),
                               (x,*model.parameters()),allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in second)


def test_selection_gradient_matches_finite_difference():
    x,_=scenes(3,seed=9)
    model=make_object(2,temperature=.4).double()
    rep=model.encode(x.double())
    probe=torch.randn_like(x,dtype=torch.double)
    g,=torch.autograd.grad((model.decode(rep)*probe).sum(),(model.score,))
    direction=torch.randn_like(g);direction/=direction.norm()
    original=model.score.detach().clone();vals=[]
    for delta in (1e-5,-1e-5):
        with torch.no_grad(): model.score.copy_(original+delta*direction)
        vals.append((model.decode(rep)*probe).sum().item())
    assert abs((g*direction).sum().item()-(vals[0]-vals[1])/2e-5)<1e-5


def test_communication_ablation_matches_parameter_count_and_gradients():
    x,_=scenes(2,size=10,seed=9,objects=3)
    sizes=[]
    for variant in ("local","pool","scan","both"):
        model=CommunicationNet(variant)
        sizes.append(sum(p.numel() for p in model.parameters()))
        y=model(x)
        assert y.shape==x.shape
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert len(set(sizes))==1


def test_sharpened_regions_preserve_smallest_object_under_soft_noise():
    x=two_objects()
    model=make_object(1,soft_mode="sharp",temperature=.03,steps=16).double()
    with torch.no_grad():
        model.score.zero_();model.score[0]=-1
    clean=model.selection(model.encode(x))[0]
    noisy=x*.95+.05/11
    assert torch.equal(clean>.5,model.selection(model.encode(noisy))[0]>.5)
    noisy.requires_grad_()
    g,=torch.autograd.grad(model(noisy).square().sum(),noisy,create_graph=True)
    h,=torch.autograd.grad(g.square().sum(),noisy)
    assert torch.isfinite(g).all() and torch.isfinite(h).all()


def test_foreground_connectivity_groups_multicolored_shape():
    labels=torch.zeros(1,6,6,dtype=torch.long)
    labels[0,2,2]=1;labels[0,2,3]=2
    x=F.one_hot(labels,11).movedim(-1,1).float()
    color,_=region_masks(x,"local",12)
    fore,_=region_masks(x,"foreground",12)
    assert color[0,14].sum()==1
    assert fore[0,14].sum()==2


def test_object_meta_gradient_matches_finite_difference():
    from arcplay.object_followups import adapted,score_prediction
    x,_=scenes(6,size=10,objects=3,seed=77)
    support=representation(x[:3].double())
    query=representation(x[3:].double())
    score=torch.tensor([.2,-.3,.1,.05,-.1],dtype=torch.double,requires_grad=True)
    def objective(s,graph):
        pred=adapted(s,support,query,inner_steps=2,create_graph=graph)
        target=score_prediction(s,query,.1)
        return (pred-target).square().mean()
    g,=torch.autograd.grad(objective(score,True),score)
    direction=torch.randn_like(score);direction/=direction.norm()
    plus=objective((score.detach()+1e-5*direction),False).item()
    minus=objective((score.detach()-1e-5*direction),False).item()
    assert abs((g*direction).sum().item()-(plus-minus)/2e-5)<1e-6


def test_named_controls_use_correct_sign():
    from arcplay.object_experiments import named_teacher
    x=two_objects().float()
    expected={"largest":(1,1),"smallest":(5,4),"leftmost":(1,1),"topmost":(1,1),"widest":(5,4)}
    for task,(y,z) in expected.items():
        model=named_teacher(task)
        mask,_=model.selection(model.encode(x))
        assert mask[0,0,y,z]>.999,task


def test_vectorized_population_matches_independent_inner_updates():
    from arcplay.object_followups import adapted,population_adapted
    x,_=scenes(6,size=10,objects=3,seed=88)
    support=representation(x[:3].double());query=representation(x[3:].double())
    scores=torch.randn(3,5,dtype=torch.double,requires_grad=True)*.2
    batched=population_adapted(scores,support,query)
    single=torch.stack([adapted(s,support,query) for s in scores])
    assert torch.allclose(batched,single,atol=1e-10,rtol=1e-8)
    gb,=torch.autograd.grad(batched.square().mean(),scores,retain_graph=True)
    gs,=torch.autograd.grad(single.square().mean(),scores)
    assert torch.allclose(gb,gs,atol=1e-10,rtol=1e-8)
