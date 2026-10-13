import torch
from torch.nn import functional as F
from arcplay.advanced_catalogue import fill_holes,shape_match,pack_sorted

def onehot(a):return F.one_hot(a,11).permute(2,0,1).float()[None]
def color(c):return F.one_hot(torch.tensor([c]),11).float()

def test_closed_and_open_holes():
    a=torch.zeros(12,12,dtype=torch.long);a[2:7,2:7]=2;a[3:6,3:6]=0
    y=fill_holes(onehot(a),color(8)).argmax(1)[0]
    expected=a.clone();expected[3:6,3:6]=8
    assert torch.equal(y,expected)
    a[2,4]=0
    assert torch.equal(fill_holes(onehot(a),color(8)).argmax(1)[0],a)

def test_translated_shape_matching_ignores_color():
    a=torch.zeros(12,12,dtype=torch.long)
    a[1:3,1:3]=2;a[6:8,6:8]=3;a[1:3,8]=4
    y=shape_match(onehot(a),color(8)).argmax(1)[0]
    expected=a.clone();expected[1:3,1:3]=8;expected[6:8,6:8]=8
    assert torch.equal(y,expected)

def test_sort_and_pack_independent_of_color_and_location():
    a=torch.zeros(12,12,dtype=torch.long);a[2:4,1:4]=3;a[7,7]=2;a[7:9,1]=4
    y=pack_sorted(onehot(a)).argmax(1)[0]
    expected=torch.full_like(a,10);expected[:2,:7]=0
    expected[0,0]=2;expected[:2,2]=4;expected[:2,4:7]=3
    assert torch.equal(y,expected)
    z=pack_sorted(onehot(a),ascending=False).argmax(1)[0]
    expected[:2,:7]=0;expected[:2,:3]=3;expected[:2,4]=4;expected[0,6]=2
    assert torch.equal(z,expected)

def test_advanced_color_gradients_and_soft_input_finiteness():
    a=torch.zeros(8,8,dtype=torch.long);a[1:5,1:5]=2;a[2:4,2:4]=0
    logits=(onehot(a)*3).requires_grad_(True)
    paint=torch.randn(1,11,requires_grad=True)
    y=fill_holes(logits.softmax(1),paint.softmax(1),steps=20)
    grads=torch.autograd.grad(y.square().sum(),(logits,paint),create_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
    g2=torch.autograd.grad(sum(g.square().sum() for g in grads),(logits,paint))
    assert all(torch.isfinite(g).all() for g in g2)
