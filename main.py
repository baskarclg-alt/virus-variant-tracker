from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from Bio import SeqIO
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# -------------------------------
def load_reference():
    ref = str(next(SeqIO.parse("data/reference.fasta", "fasta")).seq)
    return ref

# -------------------------------
def find_mutations(ref, seq):
    muts, table = [], []
    for i in range(min(len(ref), len(seq))):
        if ref[i] != seq[i]:
            m = f"{ref[i]}{i+1}{seq[i]}"
            muts.append(m)
            table.append({"position": i+1, "ref": ref[i], "mut": seq[i]})
    return muts, table

# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------------------
@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    sequences = list(SeqIO.parse(path, "fasta"))
    ref = load_reference()

    table = []
    all_muts = set()
    seq_lengths = []
    mut_counts = []
    seq_strings = []

    for s in sequences:
        seq = str(s.seq)
        seq_strings.append(seq)

        seq_lengths.append(len(seq))

        m, t = find_mutations(ref, seq)

        all_muts.update(m)
        table.extend(t)

        mut_counts.append(len(m))

    # -------------------------------
    # 📊 STATS (NEW 🔥)
    total_sequences = len(sequences)
    total_mutations = len(all_muts)
    avg_length = sum(seq_lengths) // len(seq_lengths)
    ref_length = len(ref)

    df = pd.DataFrame(table).head(100)

    # -------------------------------
    # 📊 Chart
    chart = px.bar(
        df["mut"].value_counts().head(10),
        title="Top Mutations"
    )
    chart.write_html("uploads/chart.html")

    # -------------------------------
    # 🔥 Density
    positions = df["position"]
    heat = px.histogram(
        x=positions,
        nbins=50,
        title="Mutation Density"
    )
    heat.write_html("uploads/heatmap.html")

    # -------------------------------
    # 🌳 CLEAN TREE (IMPROVED STYLE)

    tree_path = "uploads/tree.png"

    try:
        def encode(seq):
            return [ord(c) for c in seq[:300]]

        encoded = []
        labels = []

        for i, seq in enumerate(seq_strings):
            encoded.append(encode(seq))
            labels.append(f"S{i+1} | M:{mut_counts[i]}")

        X = np.array(encoded)

        Z = linkage(X, method='ward')

        plt.figure(figsize=(12,5))

        dendrogram(
            Z,
            labels=labels,
            leaf_rotation=45,
            leaf_font_size=9
        )

        plt.title("Sequence Clustering (Phylogenetic View)")
        plt.xlabel("Sequences")
        plt.ylabel("Distance")

        plt.tight_layout()
        plt.savefig(tree_path)
        plt.close()

    except Exception as e:
        print("Tree Error:", e)
        tree_path = None

    # -------------------------------
    df.to_csv("uploads/report.csv", index=False)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "chart": "/uploads/chart.html",
        "heatmap": "/uploads/heatmap.html",
        "tree": "/uploads/tree.png",
        "table": df.to_dict(orient="records"),
        "csv": "/uploads/report.csv",

        # 🔥 NEW DATA
        "total_sequences": total_sequences,
        "total_mutations": total_mutations,
        "avg_length": avg_length,
        "ref_length": ref_length
    })