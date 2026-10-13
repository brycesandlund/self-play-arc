import torch
from torch.nn import functional as F
from arcplay.catalogue import (Rule,OPS,SIZES,CODE_SIZE,encode_board,execute,
    transform,sample_pixels,probabilities,SharedExecutor)

def board():
    z=torch.zeros(10,10,dtype=torch.long)
    z[1:3,1:4]=2
    z[6,6]=3
    return F.one_hot(z,11).permute(2,0,1).float()[None]

def output(name, **kwargs):
    return transform(board(),Rule(OPS.index(name),**kwargs).code()[None]).argmax(1)[0]

def test_catalogue_count_and_crop():
    count=output('count',color=2)
    assert (count[0,:2]==7).all() and (count[1:]==10).all() and (count[0,2:]==10).all()
    crop=output('crop')
    assert (crop[:2,:3]==2).all() and (crop[2:]==10).all() and (crop[:,3:]==10).all()

def test_catalogue_crop_rotations_and_scale():
    rotate=output('rotate')
    assert (rotate[:3,:2]==2).all() and (rotate[3:]==10).all() and (rotate[:,2:]==10).all()
    scale=output('scale')
    assert (scale[:4,:6]==2).all() and (scale[4:]==10).all() and (scale[:,6:]==10).all()
    repeat=output('repeat_h',scale=1)
    assert (repeat[:2,:9]==2).all() and (repeat[2:]==10).all()

def test_catalogue_recolor_delete_move_and_copy():
    source=board().argmax(1)[0]
    recolor=output('recolor',src=1,color=3)
    expect=source.clone();expect[6,6]=8
    assert torch.equal(recolor,expect)
    delete=output('delete'); expect=source.clone();expect[1:3,1:4]=0
    assert torch.equal(delete,expect)
    move=output('move');expect=source.clone();expect[1:3,1]=0;expect[1:3,2:5]=2
    assert torch.equal(move,expect)
    copy=output('copy');expect=source.clone();expect[1:3,4]=2
    assert torch.equal(copy,expect)

def test_catalogue_alignment_and_link():
    align=output('align',src=0,dst=1)
    assert (align[6:8,8:10]==2).all() and (align[1:3,1:4]==0).all()
    link=output('link',src=0,dst=1,color=0)
    assert (link[1,4:7]==5).all() and (link[2:6,6]==5).all()

def test_all_branches_conserve_probabilities():
    from arcplay.object_lab import scenes
    x,_=scenes(4,size=12,seed=34)
    rep=encode_board(x)
    code=probabilities(torch.randn(4,CODE_SIZE))
    branches=execute(rep,code,True)
    assert torch.isfinite(branches).all()
    assert branches.min()>=-1e-6
    assert torch.allclose(branches.sum(2),torch.ones_like(branches[:,:,0]),atol=2e-5)

def test_catalogue_task_gradient_and_second_derivative():
    torch.manual_seed(31)
    from arcplay.catalogue import SLICES
    logits=torch.zeros(1,CODE_SIZE)
    logits[:,SLICES['src'].start]=5
    logits[:,SLICES['dst'].start+1]=5
    logits.requires_grad_(True)
    pred=execute(encode_board(board()),probabilities(logits))
    loss=(pred*torch.randn_like(pred)).sum()
    grad=torch.autograd.grad(loss,logits,create_graph=True)[0]
    second=torch.autograd.grad(grad.square().sum(),logits)[0]
    assert torch.isfinite(grad).all() and torch.isfinite(second).all()
    for k in ('op','src','dst','color','direction','turn','scale'):
        from arcplay.catalogue import SLICES
        assert grad[:,SLICES[k]].abs().sum()>0

def test_resampling_mixed_second_derivative():
    x=torch.randn(2,3,5,5,requires_grad=True)
    shift=torch.tensor(.27,requires_grad=True)
    yy,xx=torch.meshgrid(torch.arange(5.),torch.arange(5.),indexing='ij')
    out=sample_pixels(x,yy[None]+shift,xx[None]+.34)
    grad=torch.autograd.grad(out.square().sum(),shift,create_graph=True)[0]
    mixed=torch.autograd.grad(grad,x)[0]
    assert torch.isfinite(mixed).all() and mixed.abs().sum()>0

def test_late_routing_same_hard_function_class():
    from arcplay.catalogue import execute_late
    from arcplay.object_lab import scenes
    torch.manual_seed(27)
    x,_=scenes(18,size=12,seed=81,objects=3)
    codes=torch.stack([Rule(i,i%8,(i+3)%8,i%5,i%4,i%3,i%2).code() for i in range(18)])
    rep=encode_board(x)
    early,late=execute(rep,codes),execute_late(rep,codes)
    assert torch.allclose(early,late,atol=1e-5)
    assert torch.equal(early.argmax(1),late.argmax(1))

def test_late_routing_gradient_through_all_heads():
    from arcplay.catalogue import execute_late
    from arcplay.object_lab import scenes
    x,_=scenes(4,size=12,seed=119,regime='irregular',objects=3)
    logits=torch.zeros(4,CODE_SIZE,requires_grad=True)
    y=execute_late(encode_board(x),probabilities(logits))
    g=torch.autograd.grad((y*torch.randn_like(y)).sum(),logits,create_graph=True)[0]
    gg=torch.autograd.grad(g.square().sum(),logits)[0]
    from arcplay.catalogue import SLICES
    assert torch.isfinite(gg).all()
    assert all(g[:,s].abs().sum()>0 for s in SLICES.values())

def test_task_code_gradient_matches_finite_difference():
    from arcplay.catalogue import execute_late
    from arcplay.object_lab import scenes
    torch.manual_seed(713)
    x,_=scenes(2,size=12,seed=131,regime='irregular',objects=3)
    rep={k:v.double() for k,v in encode_board(x).items()}
    logits=torch.randn(2,CODE_SIZE,dtype=torch.double,requires_grad=True)
    direction=torch.randn_like(logits);direction=direction/direction.norm()
    objective=lambda z:execute_late(rep,probabilities(z)).square().sum()
    gradient=torch.autograd.grad(objective(logits),logits)[0]
    epsilon=1e-5
    numeric=(objective(logits.detach()+epsilon*direction)-objective(logits.detach()-epsilon*direction))/(2*epsilon)
    analytic=(gradient*direction).sum()
    assert torch.allclose(numeric,analytic,atol=1e-6,rtol=1e-5)

def test_cached_categorical_marginal_matches_late_renderer_and_gradient():
    from arcplay.catalogue import execute_late
    from arcplay.catalogue_followups import all_rules,bank
    from arcplay.cached_solver import fast_weights,weight_spec
    from arcplay.object_lab import scenes
    torch.manual_seed(113)
    x,_=scenes(2,size=12,seed=911,regime='irregular',objects=3);rep=encode_board(x)
    rs=all_rules();cached=bank(rep,rs)
    renders=torch.nn.functional.one_hot(cached.long(),11).permute(1,0,4,2,3).float()
    logits=torch.randn(2,CODE_SIZE,requires_grad=True)
    weights=fast_weights(logits,weight_spec(rs,'cpu'))
    via_cache=(renders*weights[:,:,None,None,None]).sum(1)
    direct=execute_late(rep,probabilities(logits))
    assert torch.allclose(weights.sum(1),torch.ones(2),atol=1e-6)
    assert torch.allclose(via_cache,direct,atol=1e-5)
    probe=torch.randn_like(direct)
    a=torch.autograd.grad((via_cache*probe).sum(),logits,retain_graph=True)[0]
    b=torch.autograd.grad((direct*probe).sum(),logits)[0]
    assert torch.allclose(a,b,atol=2e-5,rtol=2e-4)
