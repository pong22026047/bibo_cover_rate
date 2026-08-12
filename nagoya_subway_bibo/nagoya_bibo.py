#!/usr/bin/env python3
"""Nagoya subway, line-contained RFID/BIBO fare-recovery simulation.

The bundled section-load profiles are *relative indices* transcribed from the
published congestion diagrams. They are not passenger counts or observed OD.
"""
from __future__ import annotations

import argparse, csv, json, math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


@dataclass(frozen=True)
class Line:
    key: str
    name: str
    stations: tuple[str, ...]
    edges: tuple[tuple[int, int, float], ...]
    topology: str

    def adjacency(self):
        g = {i: [] for i in range(len(self.stations))}
        for eid, (a, b, km) in enumerate(self.edges):
            g[a].append((b, km, eid)); g[b].append((a, km, eid))
        return g

    def shortest_path(self, origin: int, destination: int):
        import heapq
        g=self.adjacency(); dist={origin:0.0}; parent={}; pq=[(0.0,origin)]
        while pq:
            d,u=heapq.heappop(pq)
            if d != dist[u]: continue
            if u == destination: break
            for v,w,eid in g[u]:
                nd=d+w
                if nd < dist.get(v,float("inf"))-1e-12:
                    dist[v]=nd; parent[v]=(u,eid); heapq.heappush(pq,(nd,v))
        nodes=[destination]; edge_ids=[]; cur=destination
        while cur != origin:
            cur,eid=parent[cur]; nodes.append(cur); edge_ids.append(eid)
        return tuple(reversed(nodes)), tuple(reversed(edge_ids)), dist[destination]


def _scaled(vals, target):
    f=target/sum(vals); out=[round(v*f,3) for v in vals]
    out[-1]=round(out[-1]+target-sum(out),3); return out


def build_lines():
    specs=json.loads((DATA/"lines.json").read_text(encoding="utf-8"))
    ans={}
    for key,s in specs.items():
        stations=tuple(s["stations"]); raw=s["edge_km_raw"]
        kms=_scaled(raw, s["official_total_km"])
        edges=[]
        for e,km in zip(s["edges"],kms): edges.append((e[0],e[1],km))
        ans[key]=Line(key,s["name"],stations,tuple(edges),s["topology"])
    return ans


def fare(distance_km: float) -> int:
    if distance_km <= 3+1e-9: return 210
    if distance_km <= 7+1e-9: return 240
    if distance_km <= 11+1e-9: return 270
    if distance_km <= 15+1e-9: return 310
    return 340


def hypotheses(line: Line):
    hs=[]
    for i in range(len(line.stations)):
        for j in range(len(line.stations)):
            if i==j: continue
            nodes,edges,d=line.shortest_path(i,j)
            hs.append({"o":i,"d":j,"nodes":nodes,"edges":edges,"km":d,"m":len(edges),"fare":fare(d)})
    return hs


def gravity_prior(hs, gamma):
    w=np.array([math.exp(-gamma*h["km"]) for h in hs],float)
    return w/w.sum()


def census_prior(line, hs, csv_path=DATA/"census_od_2021.csv"):
    """Return the official-census station-pair OD weights aligned with ``hs``.

    The bundled table is filtered to station pairs whose two endpoints occur on
    the selected line.  It does not prove that the passenger stayed on that
    line, so the result is deliberately named a station-pair OD prior.
    """
    weights={}
    with Path(csv_path).open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            if row["line"]==line.key:
                weights[(row["origin"],row["destination"])]=float(row["passengers"])
    w=np.array([weights.get((line.stations[h["o"]],line.stations[h["d"]]),0.0) for h in hs],float)
    if not np.isfinite(w).all() or w.sum()<=0:
        raise ValueError(f"{line.key}: no positive census OD weights in {csv_path}")
    return w/w.sum()


def load_profile(line_key, edge_count):
    rows=list(csv.DictReader((DATA/"section_load_relative.csv").open(encoding="utf-8-sig")))
    vals=[float(r["relative_load"]) for r in rows if r["line"]==line_key]
    if len(vals)!=edge_count: raise ValueError(f"{line_key}: section-load length {len(vals)} != {edge_count}")
    a=np.asarray(vals,float); return a/a.mean()


def section_fitted_prior(hs, base, target, edge_count, strength=40.0, iterations=3000, lr=.08):
    """KL-regularised non-negative fit: min ||A*pi-cL||² + KL(pi||base)/strength.

    Multiplicative exponentiated-gradient steps retain pi >= 0 and sum(pi)=1.
    A is edge-path incidence; c is profiled analytically each iteration.
    """
    A=np.zeros((edge_count,len(hs)),float)
    for k,h in enumerate(hs): A[list(h["edges"]),k]=1.0
    pi=base.copy(); target=target/target.mean()
    for _ in range(iterations):
        flow=A@pi; c=float(flow@target/(target@target)); residual=flow-c*target
        grad=2*(A.T@residual)+(np.log(np.maximum(pi,1e-300)/base)+1)/strength
        grad-=float(pi@grad)
        pi*=np.exp(np.clip(-lr*grad,-20,20)); pi/=pi.sum()
    rmse=float(np.sqrt(np.mean(((A@pi)/(A@pi).mean()-target)**2)))
    return pi,rmse


def posterior(observed, hs, prior, p, false_positive=1e-4):
    obs=set(observed); scores=np.empty(len(hs))
    all_edges=max(e for h in hs for e in h["edges"])+1
    for k,h in enumerate(hs):
        path=set(h["edges"]); tp=len(obs&path); fn=len(path-obs); fp=len(obs-path); tn=all_edges-len(path)-fp
        scores[k]=math.log(max(prior[k],1e-300))+tp*math.log(p)+fn*math.log(1-p)+fp*math.log(false_positive)+tn*math.log(1-false_positive)
    z=scores.max(); probs=np.exp(scores-z); probs/=probs.sum()
    return int(np.argmax(probs)),float(probs.max())


def one_seed(hs, prior, p, n, seed, threshold):
    rng=np.random.default_rng(seed); chosen=rng.choice(len(hs),size=n,p=prior)
    by_m={}; true_sum=collected=0.0; cache={}
    for idx in chosen:
        h=hs[int(idx)]; obs=[e for e in h["edges"] if rng.random()<p]
        key=tuple(obs)
        if key not in cache: cache[key]=posterior(obs,hs,prior,p)
        pred,conf=cache[key]
        paid=hs[pred]["fare"] if conf>=threshold else 0
        true_sum+=h["fare"]; collected+=paid
        b=by_m.setdefault(h["m"],[0.0,0.0,0]); b[0]+=h["fare"]; b[1]+=paid; b[2]+=1
    return collected/true_sum,by_m


def summarise(values):
    n=len(values); mu=mean(values); sd=stdev(values) if n>1 else 0.0; se=sd/math.sqrt(n)
    return mu,sd,mu-1.96*se,mu+1.96*se


def run_line(line, model, gamma, p, seeds, sample_size, threshold, out):
    hs=hypotheses(line); base=gravity_prior(hs,gamma); fit_rmse=None
    if model=="section_fit":
        prior,fit_rmse=section_fitted_prior(hs,base,load_profile(line.key,len(line.edges)),len(line.edges))
    elif model=="census_od": prior=census_prior(line,hs)
    else: prior=base
    overall=[]; bucket={}
    for seed in range(seeds):
        r,b=one_seed(hs,prior,p,sample_size,seed,threshold); overall.append(r)
        for m,(t,c,n) in b.items(): bucket.setdefault(m,[]).append((c/t,n))
    mu,sd,lo,hi=summarise(overall)
    detail=[]
    for m,rows in sorted(bucket.items()):
        rs=[x[0] for x in rows]; counts=[x[1] for x in rows]; a,b,c,d=summarise(rs)
        detail.append({"line":line.key,"line_name":line.name,"od_model":model,"p":p,"seed_count":seeds,"trip_links":m,"mean_recovery":a,"sd":b,"ci95_low":c,"ci95_high":d,"mean_sample_count":mean(counts)})
    return {"line":line.key,"line_name":line.name,"topology":line.topology,"od_model":model,"gamma":gamma,"p":p,"sample_size":sample_size,"seed_count":seeds,"mean_recovery":mu,"sd":sd,"ci95_low":lo,"ci95_high":hi,"section_fit_rmse":fit_rmse},detail,prior,hs


def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def plots(summary,detail,out):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV output was created, plots were skipped.")
        return
    plt.rcParams["axes.unicode_minus"]=False
    for line in sorted({r["line"] for r in detail}):
      rows=[r for r in detail if r["line"]==line]
      fig,ax=plt.subplots(figsize=(11,6))
      for model in sorted({r["od_model"] for r in rows}):
       for p in sorted({r["p"] for r in rows}):
        z=[r for r in rows if r["od_model"]==model and r["p"]==p]
        ax.errorbar([r["trip_links"] for r in z],[100*r["mean_recovery"] for r in z],yerr=[100*1.96*r["sd"]/math.sqrt(max(1,r["seed_count"])) for r in z],marker="o",label=f"{model}, p={p}")
      ax.set(xlabel="Trip length (links)",ylabel="Fare recovery (%)",ylim=(0,103),title=f"{line}: recovery by trip length")
      ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/f"{line}_recovery_by_trip_length.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6)); labels=[]; vals=[]; errs=[]
    for r in summary:
      labels.append(f"{r['line']}\n{r['od_model']}\np={r['p']}"); vals.append(100*r["mean_recovery"]); errs.append(100*r["sd"])
    ax.bar(range(len(vals)),vals,yerr=errs,capsize=3); ax.set_xticks(range(len(vals)),labels,rotation=45,ha="right"); ax.set_ylabel("Overall fare recovery (%)"); ax.set_ylim(0,103); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(out/"all_lines_overall_recovery.png",dpi=170); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--lines",nargs="+",default=["higashiyama","tsurumai","sakuradori","meijo_meiko"])
    ap.add_argument("--od-models",nargs="+",choices=["census_od","gravity","section_fit"],default=["census_od","gravity","section_fit"])
    ap.add_argument("--detection-rates",nargs="+",type=float,default=[.65,.70]); ap.add_argument("--gamma",type=float,default=.18)
    ap.add_argument("--seed-count",type=int,default=30); ap.add_argument("--sample-size",type=int,default=3300); ap.add_argument("--confidence-threshold",type=float,default=0.0,help="0 charges the MAP fare; raise to model manual-review/no-charge cases")
    ap.add_argument("--output-dir",default="nagoya_bibo_outputs"); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); lines=build_lines(); summary=[]; detail=[]; odrows=[]
    for key in args.lines:
      for model in args.od_models:
       for p in args.detection_rates:
        s,d,prior,hs=run_line(lines[key],model,args.gamma,p,args.seed_count,args.sample_size,args.confidence_threshold,out); summary.append(s); detail+=d
        if p==args.detection_rates[0]:
         for h,prob in zip(hs,prior): odrows.append({"line":key,"od_model":model,"origin":lines[key].stations[h["o"]],"destination":lines[key].stations[h["d"]],"d_ij_km":h["km"],"m_ij":h["m"],"F_ij_yen":h["fare"],"od_probability":prob})
    write_csv(out/"line_summary.csv",summary); write_csv(out/"trip_length_summary.csv",detail); write_csv(out/"od_candidates.csv",odrows); plots(summary,detail,out)
    (out/"run_config.json").write_text(json.dumps(vars(args),ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Wrote {out.resolve()}")

if __name__=="__main__": main()
