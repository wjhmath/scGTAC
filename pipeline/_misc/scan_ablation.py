import glob,re,collections,numpy as np
data=collections.defaultdict(dict)
for log in glob.glob('results/ablation/*/*/run_seed1.log'):
    var=log.split('/')[-3]; ds=log.split('/')[-2]
    t=open(log,errors='ignore').read().strip().splitlines()
    m=re.search(r"'ARI':\s*([\d.]+)", t[-1] if t else '')
    if m: data[ds][var]=float(m.group(1))
VARS=['full','wo_cl','wo_zinb','wo_cluster','wo_aug','wo_graph']
print(f"{'dataset':<24}"+"".join(f"{v:>10}" for v in VARS)+f"{'gain':>8}")
cand=[]
for ds in sorted(data):
    r=data[ds]; full=r.get('full')
    line=f"{ds:<24}"+"".join((f"{r[v]:>10.3f}" if v in r else f"{'-':>10}") for v in VARS)
    if full is not None:
        abl=[r[v] for v in VARS[1:] if v in r]
        if abl:
            gain=full-max(abl); line+=f"{gain:>8.3f}"
            if gain>0.01: cand.append((ds,gain))
    print(line)
print("\n候选(full 明显胜过所有消融, 模块有用):")
for ds,g in sorted(cand,key=lambda x:-x[1]): print(f"  {ds}  +{g:.3f}")
if not cand: print("  无 —— 任意数据集拿掉模块都不掉")
