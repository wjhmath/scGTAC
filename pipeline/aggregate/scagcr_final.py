import re, glob, os
from collections import defaultdict
import statistics as st

PATS = {"ARI":"'ARI'","NMI":"'NMI'","ACC":"'CA'"}
vals = defaultdict(lambda: defaultdict(list))
for log in glob.glob("results/scagcr_final/*/run_seed*.log"):
    ds = os.path.basename(os.path.dirname(log))
    txt = open(log).read().strip()
    if not txt: continue
    last = txt.split("\n")[-1]
    for mk, pat in PATS.items():
        m = re.search(rf"{pat}:\s*(?:np\.float64\()?([0-9.]+)", last)
        if m: vals[ds][mk].append(float(m.group(1)))

def ms(v):
    if not v: return "--"
    return f"{st.mean(v):.3f}±{(st.pstdev(v) if len(v)>1 else 0):.3f}"

print(f"{'dataset':<26}{'ARI':>15}{'NMI':>15}{'ACC':>15}{'seeds':>7}")
print("="*78)
for ds in sorted(vals, key=lambda d:-(st.mean(vals[d]['ARI']) if vals[d].get('ARI') else -1)):
    n = len(vals[ds].get("ARI", []))
    print(f"{ds:<26}{ms(vals[ds].get('ARI',[])):>15}{ms(vals[ds].get('NMI',[])):>15}{ms(vals[ds].get('ACC',[])):>15}{n:>7}")
print(f"\n共 {len(vals)} 个数据集")
