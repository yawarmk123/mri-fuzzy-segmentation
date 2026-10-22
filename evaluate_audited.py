import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndi
from scipy.cluster.vq import kmeans2
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description="Auditable real-case BraTS FLAIR evaluation")
p.add_argument("-n", "--num_cases", type=int, default=100)
p.add_argument("--input-dir", default="/kaggle/input")
p.add_argument("--output-dir", default="/kaggle/working")
p.add_argument("--out-name", default="results_audit")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--bootstrap", type=int, default=2000)
args, _ = p.parse_known_args()
SELECT = {"brightest": 0, "total": 0}

def normalise(v, b):
    x = v[b]
    if not x.size: raise ValueError("Empty brain mask")
    lo, hi = np.percentile(x, [1, 99])
    if not np.isfinite(lo+hi) or hi <= lo: return np.zeros(v.shape, np.float32)
    return np.clip((v-lo)/(hi-lo+1e-8), 0, 1).astype(np.float32)

def memberships(x, c, m=2.):
    d = np.abs(x[:, None]-c[None, :])+1e-8
    u = d**(-2/(m-1))
    return u/(u.sum(axis=1, keepdims=True)+1e-12)

def fcm_fit(x, c=4, m=2., seed=0):
    rng = np.random.default_rng(seed)
    s = x if x.size <= 25000 else x[rng.choice(x.size, 25000, replace=False)]
    centers = np.percentile(s, np.linspace(5,95,c))
    for _ in range(100):
        u = memberships(s, centers, m); w = u**m
        new = (w*s[:,None]).sum(0)/(w.sum(0)+1e-12)
        if np.max(np.abs(new-centers)) < 1e-5:
            centers = new; break
        centers = new
    return np.sort(centers)

def entropy_weights(R):
    m,n = R.shape
    if m <= 1: return np.ones(n)/max(n,1)
    E = -(1/np.log(m))*np.sum(R*np.log(R+1e-12),axis=0)
    div = np.maximum(1-E,0)
    return div/div.sum() if div.sum()>1e-12 else np.ones(n)/n

def copras_rank(X, beneficial):
    X = np.maximum(np.asarray(X,float),1e-8)
    R = X/(X.sum(axis=0,keepdims=True)+1e-12)
    w = entropy_weights(R); D = R*w[None,:]
    ben = np.asarray(beneficial,bool); cost = ~ben
    sp = D[:,ben].sum(1) if ben.any() else np.zeros(X.shape[0])
    sm = D[:,cost].sum(1) if cost.any() else np.zeros(X.shape[0])
    if cost.any() and sm.sum()>1e-12:
        safe=np.maximum(sm,1e-8); inv=1/safe
        q=sp+(safe.sum()*inv)/(inv.sum()+1e-12)
    else: q=sp.copy()
    u=100*q/(q.max()+1e-12)
    return int(np.argmax(u)),u,w

def largest(mask):
    lab,n=ndi.label(mask)
    if n==0:return mask.astype(bool)
    cnt=np.bincount(lab.ravel());cnt[0]=0
    return lab==int(np.argmax(cnt))

def otsu_thr(x):
    h,e=np.histogram(x,bins=256,range=(0,1)); prob=h/max(h.sum(),1)
    w=np.cumsum(prob); mu=np.cumsum(prob*(e[:-1]+e[1:])/2)
    score=(mu[-1]*w-mu)**2/(w*(1-w)+1e-12)
    return e[np.argmax(score)+1]

def otsu(v,b):
    n=normalise(v,b);return b&(n>otsu_thr(n[b]))

def kmeans(v,b):
    n=normalise(v,b);x=n[b].astype(float)
    s=x if x.size<=25000 else x[np.random.default_rng(args.seed).choice(x.size,25000,replace=False)]
    centers,_=kmeans2(s[:,None],4,minit="++",seed=args.seed)
    out=np.zeros(v.shape,bool)
    out[b]=np.abs(x[:,None]-centers.ravel()[None,:]).argmin(1)==int(centers.argmax())
    return out

def masked_fcm(v,b):
    x=normalise(v,b)[b].astype(float); c=fcm_fit(x,4,2.,args.seed)
    out=np.zeros(v.shape,bool);out[b]=memberships(x,c).argmax(1)==len(c)-1
    return out

def smooth_fcm(v,b): return masked_fcm(ndi.gaussian_filter(v,1.),b)
def smooth_fcm_cc(v,b): return ndi.binary_fill_holes(largest(smooth_fcm(v,b)))

def copras_fcm(v,b,diag=False):
    n=normalise(ndi.gaussian_filter(v,1.),b);x=n[b].astype(float)
    c=fcm_fit(x,4,2.,args.seed); lab=memberships(x,c).argmax(1)
    grid=np.zeros(v.shape,np.int16);grid[b]=lab+1
    bc=ndi.center_of_mass(b); rows=[]; ids=[]
    for k in range(1,len(c)+1):
        mask=grid==k; count=int(mask.sum())
        if count<10:continue
        surface=float((mask^ndi.binary_erosion(mask)).sum())
        compact=count/(surface+1e-3); com=ndi.center_of_mass(mask)
        dist=float(np.sqrt(sum((com[i]-bc[i])**2 for i in range(3))))
        std=float(x[lab==k-1].std())
        rows.append([max(float(c[k-1]),1e-8),max(compact,1e-8),max(dist,1e-8),max(std,1e-8)])
        ids.append(k)
    if not ids:return np.zeros(v.shape,bool),{"selected":None,"brightest":False,"weights":[]}
    idx,utility,w=copras_rank(np.array(rows),[True,True,False,False])
    chosen=ids[idx]; brightest=chosen==max(ids)
    SELECT["total"]+=1; SELECT["brightest"]+=int(brightest)
    pred=ndi.binary_fill_holes(largest(grid==chosen))
    info={"selected":chosen,"brightest":bool(brightest),"weights":w.tolist(),"candidate_ids":ids}
    return (pred,info) if diag else pred

METHODS={"otsu":otsu,"kmeans":kmeans,"masked_fcm":masked_fcm,
         "gaussian_smoothed_fcm":smooth_fcm,
         "gaussian_smoothed_fcm_cc":smooth_fcm_cc,"copras_fcm":copras_fcm}

def dice(a,b):
    den=int(a.sum())+int(b.sum())
    return float(2*np.logical_and(a,b).sum()/den) if den else 1.

def hd95(a,b,spacing):
    if not a.any() or not b.any():return np.nan
    ea=a^ndi.binary_erosion(a);eb=b^ndi.binary_erosion(b)
    da=ndi.distance_transform_edt(~ea,sampling=spacing)
    db=ndi.distance_transform_edt(~eb,sampling=spacing)
    return float(np.percentile(np.r_[db[ea],da[eb]],95))

def diagonal(shape,spacing):
    return float(np.sqrt(sum((s*z)**2 for s,z in zip(shape,spacing))))

def find_cases(root,limit):
    root=Path(root); found=[]
    if not root.exists():return []
    for d in sorted(root.rglob("*")):
        if not d.is_dir():continue
        files=list(d.glob("*.nii"))+list(d.glob("*.nii.gz"))
        fs=sorted(f for f in files if "flair" in f.name.lower() or "t2f" in f.name.lower())
        gs=sorted(f for f in files if "seg" in f.name.lower())
        if fs and gs:found.append((d.name,fs[0],gs[0]))
    seen=set(); unique=[]
    for row in found:
        key=str(row[1].resolve())
        if key not in seen:unique.append(row);seen.add(key)
    return unique[:limit]

def boot_ci(x,rng,n):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return [None,None]
    vals=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)])
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
    cases=find_cases(args.input_dir,args.num_cases)
    if not cases:raise SystemExit("ERROR: no real BraTS FLAIR+SEG cases found; no synthetic fallback.")
    print(f"Found {len(cases)} real cases; requested {args.num_cases}")
    out=Path(args.output_dir)/args.out_name;out.mkdir(parents=True,exist_ok=True)
    rows=[];manifest=[];diags=[]
    for i,(cid,fp,gp) in enumerate(cases,1):
        t=time.time();img=nib.load(str(fp));seg=nib.load(str(gp))
        v=np.asarray(img.get_fdata(dtype=np.float32));gt=np.asarray(seg.get_fdata(dtype=np.float32))>0
        spacing=tuple(float(z) for z in img.header.get_zooms()[:3])
        if v.shape!=gt.shape:print("SKIP shape mismatch",cid);continue
        v=np.nan_to_num(v,nan=0,posinf=0,neginf=0);brain=v>1e-3
        if not brain.any() or not gt.any():print("SKIP empty",cid);continue
        manifest.append({"case":cid,"flair":str(fp),"seg":str(gp),"shape":list(v.shape),
                         "spacing_mm":list(spacing),"gt_voxels":int(gt.sum())})
        for name,fn in METHODS.items():
            if name=="copras_fcm":pred,info=copras_fcm(v,brain,True);diags.append({"case":cid,**info})
            else:pred=fn(v,brain)
            h=hd95(pred,gt,spacing)
            rows.append({"case":cid,"method":name,"dice":dice(pred,gt),"hd95_mm":h,
                         "hd95_penalised_mm":diagonal(v.shape,spacing) if np.isnan(h) else h,
                         "empty_prediction":not bool(pred.any())})
        print(f"[{i}/{len(cases)}] {cid}: {time.time()-t:.2f}s")
    df=pd.DataFrame(rows)
    if df.empty or df.case.nunique()<2:raise SystemExit("ERROR: fewer than two valid cases")
    df.to_csv(out/"per_case_results.csv",index=False)
    pd.DataFrame(manifest).to_csv(out/"case_manifest.csv",index=False)
    pd.DataFrame(diags).to_json(out/"copras_diagnostics.json",orient="records",indent=2)
    rng=np.random.default_rng(args.seed); summaries=[]
    for name,s in df.groupby("method"):
        ci=boot_ci(s.dice.to_numpy(),rng,args.bootstrap);h=s.hd95_mm.to_numpy(float)
        summaries.append({"method":name,"n":s.case.nunique(),"dice_mean":s.dice.mean(),
          "dice_sd":s.dice.std(ddof=1),"dice_ci95_low":ci[0],"dice_ci95_high":ci[1],
          "hd95_nonempty_mean_mm":float(np.nanmean(h)) if np.isfinite(h).any() else None,
          "hd95_valid_n":int(np.isfinite(h).sum()),"hd95_penalised_mean_mm":s.hd95_penalised_mm.mean(),
          "empty_prediction_n":int(s.empty_prediction.sum())})
    summ=pd.DataFrame(summaries).sort_values("dice_mean",ascending=False);summ.to_csv(out/"summary.csv",index=False)
    print("\nSUMMARY\n",summ.to_string(index=False))
    piv=df.pivot(index="case",columns="method",values="dice"); comp={}
    a="copras_fcm";b="gaussian_smoothed_fcm_cc"
    if a in piv and b in piv:
        pair=piv[[a,b]].dropna();delta=(pair[a]-pair[b]).to_numpy()
        comp={"n_pairs":len(delta),"mean_delta":float(delta.mean()),"median_delta":float(np.median(delta)),
              "better_n":int((delta>0).sum()),"ties_n":int((delta==0).sum()),"worse_n":int((delta<0).sum()),
              "delta_bootstrap_ci95":boot_ci(delta,rng,args.bootstrap)}
        try:
            stat,pv=wilcoxon(delta,zero_method="wilcox")
            comp.update({"wilcoxon_statistic":float(stat),"p_value_unadjusted":float(pv)})
        except ValueError as e:comp["wilcoxon_note"]=str(e)
    (out/"paired_comparison.json").write_text(json.dumps(comp,indent=2))
    order=[m for m in METHODS if m in piv];fig,ax=plt.subplots(figsize=(11,6))
    ax.boxplot([piv[m].dropna().values for m in order]);ax.set_xticks(range(1,len(order)+1))
    ax.set_xticklabels(order,rotation=20,ha="right");ax.set_ylabel("Dice score")
    ax.set_title("BraTS FLAIR segmentation comparison (real cases)");fig.tight_layout()
    fig.savefig(out/"boxplot.png",dpi=160);plt.close(fig)
    info={"requested_cases":args.num_cases,"valid_cases":int(df.case.nunique()),"seed":args.seed,
          "methods":list(METHODS),"case_ids":sorted(df.case.unique().tolist()),
          "note":"gaussian_smoothed_fcm is Gaussian-smoothed ordinary FCM, not asserted to be published spatial FCM."}
    (out/"run_info.json").write_text(json.dumps(info,indent=2))
    print(f"\nCOPRAS brightest cluster: {SELECT['brightest']}/{SELECT['total']}")
    print("Saved:",out)
if __name__=="__main__":main()
