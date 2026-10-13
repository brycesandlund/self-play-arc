"""XP004 expressive extensions, independently audited before solver integration.

These use shared tensor routing and finite flood propagation. They are not yet
branches in the trained 18-operation solver; keep that distinction explicit.
"""
import torch
from torch.nn import functional as F
from .object_lab import region_masks,neighbors
from .catalogue import encode_board,geometry,mesh,sample_pixels,paint_over,rectangle


def fill_holes(x,color,steps=144):
    """Paint background components disconnected from the board boundary."""
    bg=x[:,:1]
    boundary=torch.zeros_like(bg);boundary[...,0,:]=1;boundary[...,-1,:]=1
    boundary[...,:,0]=1;boundary[...,:,-1]=1
    seed=bg*boundary;reach=seed
    for _ in range(steps):reach=torch.maximum(seed,bg*torch.stack([reach,*neighbors(reach)]).amax(0))
    holes=(bg-reach).clamp(0,1)
    return paint_over(x,holes,color)


def shape_match(x,color,selector=2,grouping='local',steps=24):
    """Recolor every component with the selected object's exact translated shape.

    Colors are ignored for matching. Rotation/scale invariance is deliberately
    absent. Reference and candidate shapes are compared in a canonical frame.
    """
    masks,prior=region_masks(x,grouping,steps)
    b,k,h,w=masks.shape;yy,xx=mesh(x)
    top,left,_,_=geometry(masks)
    aligned=sample_pixels(masks.reshape(b*k,1,h,w),
                          yy[None]+top.flatten()[:,None,None],xx[None]+left.flatten()[:,None,None])
    ref=encode_board(x,steps,grouping)['selected'][:,selector:selector+1]
    ry,rx,_,_=geometry(ref)
    canonical=sample_pixels(ref,yy[None]+ry[:,:,None],xx[None]+rx[:,:,None])
    distance=(aligned.reshape(b,k,h,w)-canonical).square().sum((-2,-1))
    similarity=(1-distance).clamp(0,1)*prior
    mask=(masks*similarity[:,:,None,None]).amax(1,keepdim=True)
    return paint_over(x,mask,color)


def pack_sorted(x,ascending=True,grouping='local',steps=24,sharpness=1e6):
    """Sort connected objects by area, pack left to right with one-cell gaps.

    Pairwise comparisons determine total preceding width. Each foreground seed
    carries inverse-component-area weight, so duplicates contribute once. Output
    is cropped to the packed extent and maximum object height, bounded by canvas.
    """
    masks,prior=region_masks(x,grouping,steps);b,k,h,w=masks.shape
    mass=masks.sum((-2,-1)).clamp_min(1e-8);measure=prior/mass
    top,left,bh,bw=geometry(masks);yy,xx=mesh(x)
    cx=(masks*xx).sum((-2,-1))/mass;cy=(masks*yy).sum((-2,-1))/mass
    score=(mass if ascending else -mass)+.01*cx/w+.001*cy/h
    earlier=((score[:,:,None]-score[:,None,:])*sharpness+.5).clamp(0,1)
    # Equal-score seeds all describe the same component; remove its half vote.
    destination=(earlier*((bw+1)*measure)[:,None]).sum(-1)-.5*(bw+1)
    pieces=(x[:,None]*masks[:,:,None]).reshape(b*k,x.shape[1],h,w)
    shifted=sample_pixels(pieces,yy[None]+top.flatten()[:,None,None],
                          xx[None]+(left-destination).flatten()[:,None,None])
    canvas=(shifted.reshape(b,k,x.shape[1],h,w)*measure[:,:,None,None,None]).sum(1)
    height=(bh*prior).amax(1).clamp(max=h)
    width=(((bw+1)*measure).sum(1)-1).clamp(min=0,max=w)
    valid=rectangle(yy[None],xx[None],x.new_zeros(b),x.new_zeros(b),height,width)
    canvas=canvas*valid;canvas=canvas/canvas.sum(1,keepdim=True).clamp_min(1)
    bg=torch.zeros_like(x);bg[:,0]=1;pad=torch.zeros_like(x);pad[:,-1]=1
    return canvas+bg*(valid-canvas.sum(1,keepdim=True)).clamp_min(0)+pad*(1-valid)
