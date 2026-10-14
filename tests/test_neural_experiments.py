import torch
from torch.nn import functional as F

from arcplay.neural_experiments import grid_loss, load_bank, score_labels


def test_metrics_do_not_count_padding_as_foreground():
    target=torch.tensor([[[10,10],[0,3]]])
    pred=torch.tensor([[[10,10],[0,0]]])
    scores=score_labels(pred,target,torch.zeros_like(target))
    assert scores['cell']==.75
    assert scores['foreground']==0.
    assert scores['extent']==1.
    assert scores['exact']==0.


def test_balanced_loss_has_finite_gradient_for_identity_and_empty_foreground():
    x=F.one_hot(torch.zeros(2,3,3,dtype=torch.long),11).movedim(-1,1).float()
    logits=torch.randn_like(x,requires_grad=True)
    loss=grid_loss(logits,x,x)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(loss,F.cross_entropy(logits,x.argmax(1)))


def test_compact_data_roundtrip_is_lossless(tmp_path):
    x=torch.randint(0,11,(4,5,6),dtype=torch.uint8)
    task={split:{'x':x,'y':x.flip(-1)} for split in ('train','val','test')}
    path=tmp_path/'bank.pt';torch.save({'encoding':'labels11','tasks':[task]},path)
    result=load_bank(path)[0]
    assert result['train']['x'].shape==(4,11,5,6)
    assert torch.equal(result['train']['x'].argmax(1),x.long())
    assert torch.equal(result['test']['y'].argmax(1),x.flip(-1).long())


def test_neural_inference_adaptation_does_not_depend_on_query_inputs(tmp_path):
    from arcplay.neural_transformer import InContextNeuralTransform
    from arcplay.solve_neural import solve
    torch.manual_seed(2)
    model=InContextNeuralTransform(width=8,depth=1,heads=2)
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save(dict(model=model.state_dict(),config=dict(width=8,depth=1,heads=2)),checkpoint)
    labels=torch.randint(0,3,(3,4,4))
    x=F.one_hot(labels,11).movedim(-1,1).float()
    y=F.one_hot((labels+1)%3,11).movedim(-1,1).float()
    pred,a=solve(x,y,x[:1],checkpoint,steps=2)
    _,b=solve(x,y,x[:1].flip(-1),checkpoint,steps=2)
    assert a['support_loss']==b['support_loss']
    assert a['best_step']==b['best_step']
    assert torch.allclose(pred.sum(1),torch.ones_like(pred[:,0]),atol=1e-6)
