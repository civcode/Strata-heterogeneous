"""Independent FP64 algebra, not a GPU bit emulator or admission gate."""
import numpy as np

def finite(x):
    a = np.asarray(x, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("empty/nonfinite operand")
    return a

def sigmoid(x):
    x = finite(x)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1/(1+e), e/(1+e))

def image(x, mode):
    x = finite(x)
    if mode == "fp64":
        return x
    with np.errstate(over="raise", invalid="raise"):
        f = x.astype("<f4")
        if mode == "fp32":
            return finite(f)
        if mode == "fp16":
            return finite(f.astype("<f2"))
        def bf(v):
            b = np.asarray(v, dtype="<f4").view("<u4")
            return finite(((b + np.uint32(0x7fff) + ((b >> 16) & 1)) & np.uint32(0xffff0000)).view("<f4"))
        hi = bf(f)
        if mode == "bf16":
            return hi
        if mode == "bf16x2":
            return hi + bf(f.astype(np.float64)-hi)
    raise ValueError("unknown image")

def hc_read(r, gamma, down, up, inject, norm_mode="fp64", silu_mode="fp64", eps=float(np.float32(1e-6))):
    r, gamma, down, up, inject = map(finite, (r,gamma,down,up,inject))
    if r.ndim != 2 or gamma.shape != r.shape or eps <= 0 or not np.isfinite(eps):
        raise ValueError("HC dimensions")
    h,n = r.shape
    d = h*n
    if down.ndim != 2 or down.shape[1] != d or up.shape != (d,down.shape[0]) or inject.shape != (h,d):
        raise ValueError("HC projection dimensions")
    scale = 1/np.sqrt(np.mean(r*r,axis=1)+eps)
    x = (r*scale[:,None]*gamma).ravel()
    operand = image(x,norm_mode)
    low = down@operand/h
    low = low*sigmoid(low)
    up_operand = image(low,silu_mode)
    gate = up@up_operand
    return {"normalized":x,"operand":operand,"silu":low,"up_operand":up_operand,
            "mixed":finite(np.mean((x*sigmoid(gate)).reshape(h,n),axis=0)),
            "inject":finite(inject@operand)}

def hc_write(r, block, injection):
    r,block,injection = map(finite,(r,block,injection))
    if r.ndim != 2 or block.shape != (r.shape[1],) or injection.shape != (r.shape[0],):
        raise ValueError("HC write dimensions")
    return finite(r+2*sigmoid(injection/r.shape[0])[:,None]*block)

def gdn_step(state, q, k, v, log_decay, beta):
    """One head, already normalized q/k, before output RMS and z gate."""
    state,q,k,v = map(finite,(state,q,k,v))
    if q.ndim != 1 or k.shape != q.shape or v.ndim != 1 or state.shape != (q.size,v.size):
        raise ValueError("GDN dimensions")
    if not np.isfinite(log_decay) or log_decay > 0 or not 0 <= beta <= 1:
        raise ValueError("GDN gate")
    g = np.exp(log_decay)
    delta = beta*(v-g*(k@state))
    s = finite(g*state+np.outer(k,delta))
    return s, finite((q@s)/np.sqrt(q.size))

def causal_conv(history, rows, weights):
    """Four-tap causal convolution + SiLU; history oldest first, channels last."""
    history,rows,weights = map(finite,(history,rows,weights))
    if rows.ndim != 2 or history.shape != (3,rows.shape[1]) or weights.shape != (4,rows.shape[1]):
        raise ValueError("conv dimensions")
    all_rows = np.concatenate((history,rows))
    pre = np.stack([np.sum(all_rows[t:t+4]*weights,axis=0) for t in range(len(rows))])
    return finite(pre*sigmoid(pre)), all_rows[-3:].copy()

def selected_attention(q, keys, values, scale):
    """One query/head over already selected, decoded valid K/V only."""
    q,keys,values = map(finite,(q,keys,values))
    if q.ndim != 1 or keys.ndim != 2 or keys.shape[1] != q.size or values.ndim != 2 or values.shape[0] != keys.shape[0] or not np.isfinite(scale):
        raise ValueError("attention dimensions")
    logits = finite((keys@q)*scale)
    p = np.exp(logits-max(logits)); p /= p.sum()
    return finite(p@values)

def route(logits,k):
    logits = finite(logits)
    if logits.ndim != 1 or type(k) is not int or not 0 < k < logits.size:
        raise ValueError("router dimensions")
    ids = np.argsort(-logits,kind="stable")[:k]
    p = np.exp(logits-max(logits)); p /= p.sum()
    return ids,p[ids]/max(p[ids].sum(),2**-14)

def router_sensitivity(w,x,y,k,mode="bf16"):
    """Same stored weights; separate upstream and operand rounding contributions."""
    w,x,y = map(finite,(w,x,y))
    if w.ndim != 2 or x.ndim != 1 or x.shape != y.shape or w.shape[1] != x.size:
        raise ValueError("router dimensions")
    a,b,c = map(finite,(w@x,w@y,w@image(y,mode)))
    ids,wa = route(a,k); ids_b,_ = route(b,k); ids_c,wc = route(c,k)
    ranked = np.sort(a)[::-1]
    margin = float(ranked[k-1]-ranked[k])
    delta = c-a
    bound = float(np.max(np.abs(delta)))
    return {"margin":margin,"error_linf":bound,"sufficient_set_stability":margin>2*bound,
            "upstream_linf":float(np.max(np.abs(b-a))),"rounding_linf":float(np.max(np.abs(c-b))),
            "upstream_swaps":len(set(ids)-set(ids_b)),"total_swaps":len(set(ids)-set(ids_c)),
            "order_equal":bool(np.array_equal(ids,ids_c)),
            "selected_weight_l1":float(sum(abs(dict(zip(ids,wa)).get(i,0)-dict(zip(ids_c,wc)).get(i,0)) for i in set(ids)|set(ids_c)))}

def hc_cost(n=2560,h=4,rank=320,layers=48,tokens=1):
    """Analytical projection work only; excludes head, norms, nonlinearities and traffic."""
    if any(type(i) is not int or i<=0 for i in (n,h,rank,layers,tokens)):
        raise ValueError("positive integer dimensions required")
    d=n*h
    macs=2*d*rank+h*d
    reads=2*layers*tokens
    return {"macs_per_read":macs,"projection_flops":2*macs*reads,
            "bf16x2_projection_flops":4*macs*reads,"fp32_projection_flops":2*macs*reads,
            "extra_low_image_bytes_per_chunk":2*tokens*(d+rank),
            "scope":"attention and FFN HC reads only; no measured latency"}
