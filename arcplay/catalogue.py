"""XP004: one tensor executor for procedurally parameterized ARC-like rules.

No program strings, instance labels, discrete component finder, or learned
pretraining. Fixed architectural branches accept a shared continuous task code.
Region extraction is bounded differentiable flood propagation from XP003.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from .object_lab import region_masks, cardinal_scans

OPS = ('identity','recolor','delete','move','copy','bbox','outline','ray',
       'crop','rotate','reflect','scale','repeat_h','repeat_v','count','link','align','stamp')
SELECTORS = ('largest','smallest','leftmost','rightmost','topmost','bottommost','widest','tallest')
SIZES = dict(op=len(OPS), src=8, dst=8, color=5, direction=4, turn=3, scale=2)
SLICES = {}
_n = 0
for _k, _v in SIZES.items():
    SLICES[_k] = slice(_n, _n+_v)
    _n += _v
CODE_SIZE = _n

@dataclass(frozen=True)
class Rule:
    op: int
    src: int = 0
    dst: int = 2
    color: int = 0
    direction: int = 0
    turn: int = 0
    scale: int = 0

    def name(self):
        return OPS[self.op]

    def code(self, device='cpu'):
        return torch.cat([F.one_hot(torch.tensor(asdict(self)[k],device=device),n)
                          for k,n in SIZES.items()]).float()


def probabilities(logits, temperature=1.):
    return torch.cat([(logits[...,s]/temperature).softmax(-1) for s in SLICES.values()],-1)


def active_heads(op):
    name = OPS[op]
    heads = ['op']
    if name not in ('identity','count'): heads += ['src']
    if name in ('link','align','stamp'): heads += ['dst']
    if name in ('recolor','bbox','outline','ray','count','link'): heads += ['color']
    if name in ('move','copy','ray'): heads += ['direction']
    if name=='rotate': heads += ['turn']
    if name in ('scale','repeat_h','repeat_v'): heads += ['scale']
    return heads


def sample_pixels(x, yy, xx):
    """Bilinear tensor gather, zero exterior; admits second derivatives.

    floor chooses neighboring pixels; fractional coordinates retain their
    gradient. Unlike grid_sample, mixed second derivatives are implemented.
    """
    b,c,h,w=x.shape
    yy,xx=torch.broadcast_tensors(yy,xx)
    yy,xx=yy.expand(b,-1,-1),xx.expand(b,-1,-1)
    y0,x0=yy.floor(),xx.floor()
    fy,fx=yy-y0,xx-x0
    out=x.new_zeros(b,c,*yy.shape[-2:])
    for dy,dx,weight in ((0,0,(1-fy)*(1-fx)),(0,1,(1-fy)*fx),
                         (1,0,fy*(1-fx)),(1,1,fy*fx)):
        y,xp=y0+dy,x0+dx
        valid=(y>=0)&(y<h)&(xp>=0)&(xp<w)
        index=(y.clamp(0,h-1)*w+xp.clamp(0,w-1)).long().flatten(1)
        value=x.flatten(2).gather(2,index[:,None].expand(-1,c,-1)).reshape(b,c,*yy.shape[-2:])
        out=out+value*(weight*valid)[:,None]
    return out


def geometry(m):
    """Soft bounding box edges, exact for nonempty binary masks."""
    rows=m.amax(-1).clamp(0,1)
    cols=m.amax(-2).clamp(0,1)
    top=(1-rows.cummax(-1).values).sum(-1)
    left=(1-cols.cummax(-1).values).sum(-1)
    bottom=m.shape[-2]-(1-rows.flip(-1).cummax(-1).values).sum(-1)
    right=m.shape[-1]-(1-cols.flip(-1).cummax(-1).values).sum(-1)
    return top,left,(bottom-top).clamp_min(1),(right-left).clamp_min(1)


def mesh(x):
    h,w=x.shape[-2:]
    return torch.meshgrid(torch.arange(h,device=x.device,dtype=x.dtype),
                          torch.arange(w,device=x.device,dtype=x.dtype),indexing='ij')


def rectangle(yy,xx,top,left,h,w):
    def col(v): return v[:,None,None]
    return ((yy-col(top)+1).clamp(0,1)*(col(top+h)-yy).clamp(0,1)*
            (xx-col(left)+1).clamp(0,1)*(col(left+w)-xx).clamp(0,1))[:,None]


def encode_board(x, steps=24, grouping='local', temperature=1e-6):
    masks,prior=region_masks(x,mode=grouping,steps=steps)
    b,k,h,w=masks.shape
    area=masks.sum((-2,-1)).clamp_min(1e-8)
    yy,xx=mesh(x)
    cy=(masks*yy).sum((-2,-1))/area
    cx=(masks*xx).sum((-2,-1))/area
    top,left,bh,bw=geometry(masks)
    # Lexicographic tie break is intentionally part of this fixed family.
    # Exact-equal candidates within the same component remain equal.
    tie=1e-2*cx/w+1e-3*cy/h
    scores=torch.stack((area+tie,-area+tie,-cx+1e-3*cy/h,cx+1e-3*cy/h,
                        -cy+1e-3*cx/w,cy+1e-3*cx/w,bw+tie,bh+tie),1)
    logits=scores/temperature+prior.clamp_min(1e-30).log()[:,None]-area.log()[:,None]
    logits=logits.masked_fill(prior[:,None]<=0,-1e30)
    weights=logits.softmax(-1)
    selected=torch.einsum('bsk,bkhw->bshw',weights,masks)
    count=(prior/area).sum(1)
    return {'x':x,'selected':selected,'count':count}


def paint_over(x,mask,color):
    return x*(1-mask)+color[:,:,None,None]*mask


def over(x,patch):
    alpha=patch.sum(1,keepdim=True).clamp(0,1)
    return x*(1-alpha)+patch


def execute(rep, code, return_branches=False):
    """Task code can be exact categorical weights or differentiable soft mixtures.

    Code batch and board batch must match. Spatial routing and object rendering
    are shared by all operations. Geometry is re-extracted for composed stages.
    """
    x=rep['x']; b,c,h,w=x.shape
    p={k:code[...,s] for k,s in SLICES.items()}
    src=(rep['selected']*p['src'][:,:,None,None]).sum(1,keepdim=True)
    dst=(rep['selected']*p['dst'][:,:,None,None]).sum(1,keepdim=True)
    color=F.pad(p['color'],(5,1))
    bg=torch.zeros_like(color); bg[:,0]=1
    pad=torch.zeros_like(color); pad[:,-1]=1
    yy,xx=mesh(x); yy=yy[None];xx=xx[None]
    sy,sx,sh,sw=[a[:,0] for a in geometry(src)]
    ty,tx,th,tw=[a[:,0] for a in geometry(dst)]
    col=lambda a:a[:,None,None]
    selected=x*src
    cleared=paint_over(x,src,bg)
    branches=[x,paint_over(x,src,color),cleared]
    moves=[]
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
        moves.append(sample_pixels(selected,yy-dy,xx-dx))
    moved=(torch.stack(moves,1)*p['direction'][:,:,None,None,None]).sum(1)
    branches += [over(cleared,moved),over(x,moved)]
    box=rectangle(yy,xx,sy,sx,sh,sw)
    branches += [paint_over(x,box,color)]
    padded=F.pad(src,(1,1,1,1))
    adjacent=torch.stack((padded[...,1:1+h,:w],padded[...,1:1+h,2:2+w],
                          padded[...,:h,1:1+w],padded[...,2:2+h,1:1+w])).amax(0)
    outline=(adjacent-src).clamp(0,1)
    branches += [paint_over(x,outline,color)]
    rays=torch.stack(cardinal_scans(src),1)
    ray=(rays*p['direction'][:,:,None,None,None]).sum(1)
    branches += [paint_over(x,(ray-src).clamp(0,1),color)]
    def crop_render(patch,oh,ow):
        valid=rectangle(yy,xx,torch.zeros_like(oh),torch.zeros_like(ow),oh,ow)
        patch=patch*valid
        patch=patch/patch.sum(1,keepdim=True).clamp_min(1)
        return patch+bg[:,:,None,None]*(valid-patch.sum(1,keepdim=True)).clamp_min(0)+pad[:,:,None,None]*(1-valid)
    cropped=sample_pixels(selected,yy+col(sy),xx+col(sx))
    branches += [crop_render(cropped,sh,sw)]
    rotations=[]
    for turn in range(1,4):
        if turn==1: ry,rx=xx, col(sw-1)-yy; oh,ow=sw,sh
        elif turn==2: ry,rx=col(sh-1)-yy,col(sw-1)-xx;oh,ow=sh,sw
        else: ry,rx=col(sh-1)-xx,yy;oh,ow=sw,sh
        patch=sample_pixels(selected,ry+col(sy),rx+col(sx))
        rotations.append(crop_render(patch,oh,ow))
    branches += [(torch.stack(rotations,1)*p['turn'][:,:,None,None,None]).sum(1)]
    reflected=sample_pixels(selected,yy+col(sy),col(sx+sw-1)-xx)
    branches += [crop_render(reflected,sh,sw)]
    scales=[]; repeats_h=[];repeats_v=[]
    for factor in (2,3):
        # Fixed nearest-cell expansion is linear in pixel values. Coordinate
        # floor here depends on constants only, never on learned task parameters.
        patch=sample_pixels(selected,(yy/factor).floor()+col(sy),(xx/factor).floor()+col(sx))
        scales.append(crop_render(patch,(sh*factor).clamp(max=h),(sw*factor).clamp(max=w)))
        # Periodic copy uses a sum of bounded translated patches (no modulo).
        hh=torch.zeros_like(x); vv=torch.zeros_like(x)
        for k in range(factor):
            hh=hh+sample_pixels(selected,yy+col(sy),xx+col(sx-k*sw))
            vv=vv+sample_pixels(selected,yy+col(sy-k*sh),xx+col(sx))
        repeats_h.append(crop_render(hh,sh,(sw*factor).clamp(max=w)))
        repeats_v.append(crop_render(vv,(sh*factor).clamp(max=h),sw))
    mix=lambda values:(torch.stack(values,1)*p['scale'][:,:,None,None,None]).sum(1)
    branches += [mix(scales),mix(repeats_h),mix(repeats_v)]
    countvalid=rectangle(yy,xx,x.new_zeros(b),x.new_zeros(b),x.new_ones(b),rep['count'].clamp(max=w))
    branches += [color[:,:,None,None]*countvalid+pad[:,:,None,None]*(1-countvalid)]
    # L connector joins the top-left corners, horizontal leg first.
    ymin,ymax=torch.minimum(sy,ty),torch.maximum(sy,ty)
    xmin,xmax=torch.minimum(sx,tx),torch.maximum(sx,tx)
    horiz=rectangle(yy,xx,sy,xmin,torch.ones_like(sy),xmax-xmin+1)
    vert=rectangle(yy,xx,ymin,tx,ymax-ymin+1,torch.ones_like(sx))
    line=torch.maximum(horiz,vert)*(1-src)*(1-dst)
    branches += [paint_over(x,line,color)]
    # Align/copy source immediately to the right of destination (one-cell gap).
    placed=sample_pixels(selected,yy+col(sy-ty),xx+col(sx-tx-tw-1))
    branches += [over(cleared,placed),over(x,placed)]
    stack=torch.stack(branches,1)
    if return_branches: return stack
    return (stack*p['op'][:,:,None,None,None]).sum(1)


def transform(x, code, **kwargs):
    return execute(encode_board(x,**kwargs),code)


class SharedExecutor(nn.Module):
    """Repeat the identical catalogue executor with a separate inferred stage code."""
    def forward(self,x,codes):
        if codes.ndim==2: codes=codes[:,None]
        for stage in range(codes.shape[1]):
            x=transform(x,codes[:,stage])
        return x


def execute_late(rep,code):
    """Mix rendered alternatives, preserving crisp geometry inside each branch.

    Same hard-code function class as execute. The relaxation changes: mixtures
    over object choices occur after rendering, avoiding fictitious soft bounding
    boxes between unrelated objects. Parameter and board gradients remain live.
    """
    x=rep['x'];b,c,h,w=x.shape;s=8
    expanded={k:v.repeat_interleave(s,0) for k,v in rep.items()}
    routed=code.repeat_interleave(s,0).clone()
    routed[:,SLICES['src']]=torch.eye(s,device=x.device,dtype=x.dtype).repeat(b,1)
    branches=execute(expanded,routed,True).reshape(b,s,len(OPS),c,h,w)
    # Relational branches also postpone the destination mixture until rendering.
    masks=rep['selected'];y,left,height,width=geometry(masks)
    src=masks[:,:,None].expand(-1,-1,s,-1,-1).reshape(b*s*s,1,h,w)
    dst=masks[:,None].expand(-1,s,-1,-1,-1).reshape(b*s*s,1,h,w)
    repeat_src=lambda v:v[:,:,None].expand(-1,-1,s).flatten()
    repeat_dst=lambda v:v[:,None].expand(-1,s,-1).flatten()
    sy,sx=repeat_src(y),repeat_src(left);ty,tx,tw=repeat_dst(y),repeat_dst(left),repeat_dst(width)
    xxboard=x[:,None,None].expand(-1,s,s,-1,-1,-1).reshape(b*s*s,c,h,w)
    yy,xx=mesh(x);yy=yy[None];xx=xx[None]
    color=F.pad(code[:,SLICES['color']],(5,1)).repeat_interleave(s*s,0)
    bg=torch.zeros_like(color);bg[:,0]=1
    horiz=rectangle(yy,xx,sy,torch.minimum(sx,tx),torch.ones_like(sy),(sx-tx).abs()+1)
    vert=rectangle(yy,xx,torch.minimum(sy,ty),tx,(sy-ty).abs()+1,torch.ones_like(sx))
    line=torch.maximum(horiz,vert)*(1-src)*(1-dst)
    linked=paint_over(xxboard,line,color)
    placed=sample_pixels(xxboard*src,yy+(sy-ty)[:,None,None],xx+(sx-tx-tw-1)[:,None,None])
    aligned=over(paint_over(xxboard,src,bg),placed);stamped=over(xxboard,placed)
    relation=torch.stack((linked,aligned,stamped),1).reshape(b,s,s,3,c,h,w)
    relation=(relation*code[:,SLICES['dst']][:,None,:,None,None,None,None]).sum(2)
    branches=torch.cat((branches[:,:,:15],relation),2)
    return (branches*code[:,SLICES['src']][:,:,None,None,None,None]*code[:,SLICES['op']][:,None,:,None,None,None]).sum((1,2))
