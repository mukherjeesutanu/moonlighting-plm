"""
Non-pLM baselines, saved next to the pLM embeddings so every representation
goes through the identical evaluation code.

    python scripts/baseline_features.py

  length     : log sequence length (1 dim) -- the trivial "size" shortcut
  handcrafted: AAC (20) + dipeptide composition (400) + 8 ProtParam descriptors
"""
import itertools, pathlib
import numpy as np, pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

ROOT = pathlib.Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"
DIPEP = ["".join(p) for p in itertools.product(AA, repeat=2)]


def handcrafted(seq):
    s = "".join(a for a in seq if a in AA)          # ProtParam rejects X/U/B/Z
    pa, L = ProteinAnalysis(s), len(s)
    aac = [s.count(a) / L for a in AA]
    pairs = [s[i:i + 2] for i in range(L - 1)]
    dpc = pd.Series(pairs).value_counts().reindex(DIPEP, fill_value=0).values / (L - 1)
    helix, turn, sheet = pa.secondary_structure_fraction()
    phys = [np.log(L), pa.molecular_weight() / L, pa.gravy(), pa.isoelectric_point(),
            pa.aromaticity(), pa.instability_index(), pa.charge_at_pH(7.0) / L, helix]
    return np.concatenate([aac, dpc, phys])


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data/clean.csv")
    out = ROOT / "embeddings"; out.mkdir(exist_ok=True)
    np.save(out / "length.npy", np.log(df.length.values)[:, None].astype(np.float32))
    np.save(out / "handcrafted.npy", np.stack(df.Sequence.map(handcrafted)).astype(np.float32))
    print("saved length (1) and handcrafted (428) features")
