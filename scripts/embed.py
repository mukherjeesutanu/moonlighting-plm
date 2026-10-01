"""
Extract mean-pooled per-protein embeddings from several protein language models.

    python scripts/embed.py --model esm2_650M --fasta data/clean.fasta --out embeddings/

Long sequences are never truncated: they are split into overlapping windows,
per-residue embeddings are averaged where windows overlap, and the protein
vector is the mean over all residues (special tokens excluded).
"""
import argparse, json, pathlib, time
import numpy as np, torch
from tqdm import tqdm

MODELS = {
    # name        : (family, checkpoint)
    "esm2_8M"     : ("esm2",  "facebook/esm2_t6_8M_UR50D"),
    "esm2_35M"    : ("esm2",  "facebook/esm2_t12_35M_UR50D"),
    "esm2_150M"   : ("esm2",  "facebook/esm2_t30_150M_UR50D"),
    "esm2_650M"   : ("esm2",  "facebook/esm2_t33_650M_UR50D"),
    "esm2_3B"     : ("esm2",  "facebook/esm2_t36_3B_UR50D"),
    "esmc_300M"   : ("esmc",  "esmc_300m"),
    "esmc_600M"   : ("esmc",  "esmc_600m"),
    "prott5_xl"   : ("t5",    "Rostlab/prot_t5_xl_half_uniref50-enc"),
    "ankh_large"  : ("ankh",  "ElnaggarLab/ankh-large"),
}
WINDOW, STRIDE = 1000, 500
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def read_fasta(path):
    ids, seqs = [], []
    for block in open(path).read().split(">")[1:]:
        head, *body = block.strip().split("\n")
        ids.append(head.split("|")[0]); seqs.append("".join(body))
    return ids, seqs


def load(name):
    """Return fwd(seq) -> float32 tensor (n_layers, L, d) of per-residue states."""
    family, ckpt = MODELS[name]

    if family == "esm2":
        from transformers import AutoTokenizer, EsmModel
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = EsmModel.from_pretrained(ckpt, add_pooling_layer=False)
        model = model.half() if DEV == "cuda" else model
        model = model.to(DEV).eval()

        def fwd(seq):
            x = tok(seq, return_tensors="pt").to(DEV)
            hs = model(**x, output_hidden_states=True).hidden_states
            return torch.stack(hs)[:, 0, 1:-1].float()           # drop <cls>, <eos>

    elif family == "esmc":
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        model = ESMC.from_pretrained(ckpt).to(DEV).eval()
        cfg = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)

        def fwd(seq):
            out = model.logits(model.encode(ESMProtein(sequence=seq)), cfg)
            # hidden_states = outputs of blocks 1..n (no input-embedding layer); the last one is
            # pre-final-norm, so swap it for the normed output that `embeddings` returns
            hs = out.hidden_states[:-1, 0, 1:-1].float()
            last = out.embeddings[0, 1:-1].float()[None]
            return torch.cat([hs, last])

    elif family in ("t5", "ankh"):
        from transformers import T5EncoderModel, AutoTokenizer, T5Tokenizer
        if family == "t5":
            tok = T5Tokenizer.from_pretrained(ckpt, do_lower_case=False)
            model = T5EncoderModel.from_pretrained(ckpt, torch_dtype=torch.float16)
        else:  # Ankh overflows in fp16 -> keep fp32
            tok = AutoTokenizer.from_pretrained(ckpt)
            model = T5EncoderModel.from_pretrained(ckpt)
        model = (model.float() if DEV == "cpu" else model).to(DEV).eval()

        def fwd(seq):
            seq = "".join(a if a in "ACDEFGHIKLMNPQRSTVWY" else "X" for a in seq)
            if family == "t5":
                x = tok(" ".join(seq), return_tensors="pt")
            else:
                x = tok([list(seq)], is_split_into_words=True, return_tensors="pt")
            x = x.to(DEV)
            hs = model(**x, output_hidden_states=True).hidden_states
            return torch.stack(hs)[:, 0, :-1].float()            # drop </s>

    fwd.n_params = sum(p.numel() for p in model.parameters())
    return fwd


@torch.no_grad()
def embed(seq, fwd):
    L = len(seq)
    starts = list(range(0, max(L - WINDOW, 0) + 1, STRIDE))
    if starts[-1] + WINDOW < L:
        starts.append(L - WINDOW)
    acc, cnt = None, torch.zeros(L, device=DEV)
    for s in starts:
        h = fwd(seq[s:s + WINDOW])
        assert h.shape[1] == min(WINDOW, L - s), (h.shape, L, s)
        if acc is None:
            acc = torch.zeros(h.shape[0], L, h.shape[2], device=DEV)
        acc[:, s:s + h.shape[1]] += h
        cnt[s:s + h.shape[1]] += 1
    return (acc / cnt[None, :, None]).mean(1).cpu().numpy()     # (n_layers, d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--fasta", default="data/clean.fasta")
    ap.add_argument("--out", default="embeddings")
    a = ap.parse_args()

    ids, seqs = read_fasta(a.fasta)
    fwd, t0 = load(a.model), time.time()
    E = np.stack([embed(s, fwd) for s in tqdm(seqs, desc=a.model)])  # (N, n_layers, d)
    assert np.isfinite(E).all(), f"{a.model}: non-finite embeddings"

    out = pathlib.Path(a.out); out.mkdir(exist_ok=True, parents=True)
    np.save(out / f"{a.model}.npy", E[:, -1].astype(np.float32))           # last layer
    # all layers, z-scored per feature so they fit float16 (T5 activations overflow it);
    # harmless for probing, which standardises features anyway. ~20-65 MB each -> not in git.
    Z = (E - E.mean(0)) / (E.std(0) + 1e-6)
    (out / "layers").mkdir(exist_ok=True)
    np.save(out / "layers" / f"{a.model}.npy", Z.astype(np.float16))
    meta = dict(model=a.model, checkpoint=MODELS[a.model][1], n_params=fwd.n_params,
                dim=E.shape[2], seconds=round(time.time() - t0),
                # index 0 of the layer stack is the input embedding for all but ESM C
                first_layer_index=1 if MODELS[a.model][0] == "esmc" else 0)
    meta["n_layers"] = E.shape[1] - 1 + meta["first_layer_index"]
    json.dump(meta, open(out / f"{a.model}.json", "w"), indent=1)
    print(meta)
