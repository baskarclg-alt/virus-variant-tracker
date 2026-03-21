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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# -------------------------------
def load_reference():
    ref_path = os.path.join(DATA_FOLDER, "reference.fasta")
    return str(next(SeqIO.parse(ref_path, "fasta")).seq[:10000])  # 🔥 LIMIT

# -------------------------------
def find_mutations(ref, seq):
    muts, table = [], []

    seq = seq[:10000]  # 🔥 LIMIT

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

    try:
        path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(path, "wb") as f:
            f.write(await file.read())

        sequences = list(SeqIO.parse(path, "fasta"))[:30]  # 🔥 LIMIT SEQS

        ref = load_reference()

        table = []
        all_muts = set()
        seq_lengths = []
        seq_strings = []
        mut_counts = []

        for s in sequences:
            seq = str(s.seq)
            seq_strings.append(seq)

            seq_lengths.append(len(seq))

            muts, t = find_mutations(ref, seq)

            all_muts.update(muts)
            table.extend(t)

            mut_counts.append(len(muts))

        df = pd.DataFrame(table)

        if df.empty:
            return HTMLResponse("No mutations found")

        # 🔥 STATS
        total_sequences = len(sequences)
        total_mutations = len(all_muts)
        avg_length = sum(seq_lengths) // len(seq_lengths)
        ref_length = len(ref)

        df_display = df.head(100)

        # -------------------------------
        # 📊 Chart
        chart = px.bar(df["mut"].value_counts().head(10))
        chart.write_html(os.path.join(UPLOAD_FOLDER, "chart.html"))

        # -------------------------------
        # 🔥 Density
        heat = px.histogram(x=df["position"], nbins=40)
        heat.write_html(os.path.join(UPLOAD_FOLDER, "heatmap.html"))

        # -------------------------------
        # 🌳 LIGHT TREE
        tree_path = os.path.join(UPLOAD_FOLDER, "tree.png")

        try:
            X = np.array([list(map(ord, s[:200])) for s in seq_strings])

            Z = linkage(X, method='ward')

            plt.figure(figsize=(10,4))
            dendrogram(Z, leaf_rotation=45)
            plt.tight_layout()
            plt.savefig(tree_path)
            plt.close()

        except:
            tree_path = None

        # -------------------------------
        df.to_csv(os.path.join(UPLOAD_FOLDER, "report.csv"), index=False)

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "chart": "/uploads/chart.html",
            "heatmap": "/uploads/heatmap.html",
            "tree": "/uploads/tree.png",
            "table": df_display.to_dict(orient="records"),

            "total_sequences": total_sequences,
            "total_mutations": total_mutations,
            "avg_length": avg_length,
            "ref_length": ref_length
        })

    except Exception as e:
        return HTMLResponse(f"Error: {str(e)}")