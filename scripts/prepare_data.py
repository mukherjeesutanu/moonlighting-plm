"""
Curate the raw moonlighting / non-moonlighting spreadsheets into a clean, frozen benchmark.

    python scripts/prepare_data.py

Steps
  A  load data/raw/{Moonlighting,Non-moonlighting}.xlsx
  B  drop accessions listed under BOTH labels, and duplicate sequences
  C  drop sequences < 30 aa; flag > 1022 aa for windowed embedding (never truncate)
  D  all-vs-all Smith-Waterman -> homology clusters (>=30% identity, >=50% coverage)
  E  freeze CV folds: 5 repeats x {random, homology-grouped} stratified 5-fold

Outputs  data/clean.csv, data/clean.fasta, data/folds.json,
         data/pairwise_identity.npy, results/qc_audit.json
"""
import hashlib, json, pathlib, re
import numpy as np, pandas as pd, parasail
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, DATA, RES = ROOT / "data/raw", ROOT / "data", ROOT / "results"
MIN_LEN, MAX_TOK, ID_THR, COV_THR = 30, 1022, 0.30, 0.50

# A -- load
moon = pd.read_excel(RAW / "Moonlighting.xlsx").assign(label=1)
nonm = pd.read_excel(RAW / "Non-moonlighting.xlsx").assign(label=0)
df = pd.concat([moon, nonm], ignore_index=True)
df["Entry"] = df.Entry.astype(str).str.strip()
df["Sequence"] = df.Sequence.astype(str).str.strip().str.upper()
audit = {"loaded_moonlighting": len(moon), "loaded_nonmoonlighting": len(nonm)}

# B -- label contradictions and duplicates
both = sorted(set(moon.Entry.str.strip()) & set(nonm.Entry.str.strip()))
df = df[~df.Entry.isin(both)]
df = df[~df.Sequence.duplicated(keep="first")]
uniprot = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
assert df.Entry.str.match(uniprot).all(), "malformed accession"
audit["removed_label_contradictions"] = 2 * len(both)
print(f"[B] removed {len(both)} accessions present in both classes: {both}")

# C -- length
df["length"] = df.Sequence.str.len()
audit["dropped_too_short"] = int((df.length < MIN_LEN).sum())
df = df[df.length >= MIN_LEN].reset_index(drop=True)
df["needs_windowing"] = df.length > MAX_TOK
df["id"] = df.Entry
audit["needs_windowing"] = int(df.needs_windowing.sum())
audit["prevalence"] = round(float(df.label.mean()), 4)

# D -- homology clusters
seqs, N = df.Sequence.tolist(), len(df)
pid, cov = np.eye(N), np.eye(N)
for i in range(N):
    for j in range(i + 1, N):
        r = parasail.sw_stats_striped_16(seqs[i], seqs[j], 11, 1, parasail.blosum62)
        mn = min(len(seqs[i]), len(seqs[j]))
        pid[i, j] = pid[j, i] = r.matches / mn
        cov[i, j] = cov[j, i] = r.length / mn
A = (pid >= ID_THR) & (cov >= COV_THR)
n_clu, df["cluster30"] = connected_components(A, directed=False)
print(f"[D] {N} proteins -> {n_clu} clusters at {ID_THR:.0%} id / {COV_THR:.0%} cov")

# E -- frozen folds
y, g = df.label.values, df.cluster30.values
folds = {}
for regime in ["random", "grouped"]:
    for rep in range(5):
        cv = (StratifiedKFold(5, shuffle=True, random_state=rep) if regime == "random"
              else StratifiedGroupKFold(5, shuffle=True, random_state=rep))
        assign = np.empty(N, int)
        for k, (_, te) in enumerate(cv.split(df, y, g if regime == "grouped" else None)):
            assign[te] = k
        folds[f"{regime}_rep{rep}"] = assign.tolist()

# write
RES.mkdir(exist_ok=True)
cols = ["Entry", "Sequence", "label", "length", "needs_windowing", "id", "cluster30"]
df[cols].to_csv(DATA / "clean.csv", index=False)
with open(DATA / "clean.fasta", "w") as fh:
    fh.writelines(f">{r.id}|{r.label}\n{r.Sequence}\n" for r in df.itertuples())
json.dump(folds, open(DATA / "folds.json", "w"))
np.save(DATA / "pairwise_identity.npy", np.where(cov >= COV_THR, pid, 0).astype(np.float16))
h = hashlib.sha256("".join(df.id + df.Sequence + df.label.astype(str)).encode()).hexdigest()[:16]
audit.update(final_n=N, n_clusters=int(n_clu), dataset_sha256_16=h)
json.dump(audit, open(RES / "qc_audit.json", "w"), indent=2)
print(json.dumps(audit, indent=2))
