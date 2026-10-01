"""
Fetch UniProt annotations (names, organism, EC, GO-MF, keywords) for every protein.
Used only for *interpreting* the models, never as model input.

    python scripts/fetch_uniprot.py
"""
import io, pathlib
import pandas as pd, requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIELDS = "accession,protein_name,organism_name,ec,go_f,keyword,annotation_score"

ids = pd.read_csv(ROOT / "data/clean.csv").Entry.tolist()
chunks = []
for i in range(0, len(ids), 100):
    query = " OR ".join(f"accession:{a}" for a in ids[i:i + 100])
    r = requests.get("https://rest.uniprot.org/uniprotkb/search", timeout=60,
                     params=dict(query=query, format="tsv", size=500, fields=FIELDS))
    r.raise_for_status()
    chunks.append(pd.read_csv(io.StringIO(r.text), sep="\t"))
ann = pd.concat(chunks)
assert set(ann.Entry) == set(ids), "some accessions not returned"
ann.to_csv(ROOT / "data/uniprot_annotations.csv", index=False)
print(f"saved annotations for {len(ann)} proteins")
