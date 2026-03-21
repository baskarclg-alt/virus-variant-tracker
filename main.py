from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os
import shutil
from Bio import SeqIO
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# -------------------------------
def load_reference():
    try:
        ref = next(SeqIO.parse(os.path.join(DATA_FOLDER, "reference.fasta"), "fasta"))
        return str(ref.seq)
    except:
        return ""

# -------------------------------
def find_mutations(ref, seq, limit=3000):
    ref = ref[:limit]
    seq = seq[:limit]

    muts = []
    table = []

    for i in range(min(len(ref), len(seq))):
        r = ref[i]
        s = seq[i]

        if r not in "ATGC" or s not in "ATGC":
            continue

        if r != s:
            mut_type = f"{r}>{s}"

            muts.append((i+1, mut_type))

            table.append({
                "position": i+1,
                "ref": r,
                "mut": s,
                "type": mut_type
            })

    return muts, table

# -------------------------------
def safe_lineage(mut_counts):
    try:
        df = pd.DataFrame({"mut": mut_counts})

        if df["mut"].nunique() < 4:
            return ["Lineage-A"] * len(mut_counts)

        df["Lineage"] = pd.qcut(
            df["mut"],
            4,
            labels=["Lineage-A", "Lineage-B", "Lineage-C", "Lineage-D"],
            duplicates="drop"
        )

        return df["Lineage"].astype(str).tolist()

    except:
        return ["Lineage-A"] * len(mut_counts)

# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------------------
@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):

    try:
        # 🔥 SAFE FILE SAVE (RENDER FRIENDLY)
        path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 🔥 READ ONLY USER FILE
        sequences = list(SeqIO.parse(path, "fasta"))

        if len(sequences) == 0:
            return HTMLResponse("❌ No sequences found in file")

        ref = load_reference()

        mut_counts = []
        all_muts = []
        table = []
        seq_vectors = []
        labels = []

        for i, s in enumerate(sequences):
            seq = str(s.seq).upper()

            if len(seq) == 0:
                continue

            muts, t = find_mutations(ref, seq)

            mut_counts.append(len(muts))
            all_muts.extend(muts)
            table.extend(t)

            seq_vectors.append([ord(c) for c in seq[:300]])
            labels.append(f"Seq{i+1}")

        if len(table) == 0:
            return HTMLResponse("⚠️ No mutations detected")

        lineage_list = safe_lineage(mut_counts)

        df = pd.DataFrame(table)

        # -------------------------------
        total_sequences = len(sequences)
        total_mutations = len(all_muts)
        unique_mutations = len(set(all_muts))
        avg_mutations = total_mutations // max(total_sequences, 1)

        # -------------------------------
        # 📊 Chart
        chart_path = os.path.join(UPLOAD_FOLDER, "chart.html")
        fig1 = px.bar(df["type"].value_counts().head(10), title="Top Mutation Types")
        fig1.write_html(chart_path)

        # -------------------------------
        # 🔥 Density
        heat_path = os.path.join(UPLOAD_FOLDER, "heatmap.html")
        fig2 = px.histogram(df, x="position", nbins=50, title="Mutation Density")
        fig2.write_html(heat_path)

        # -------------------------------
        # 🌳 Tree
        tree_path = os.path.join(UPLOAD_FOLDER, "tree.png")

        try:
            if len(seq_vectors) > 2:
                X = np.array(seq_vectors[:10])
                Z = linkage(X, method='ward')

                plt.figure(figsize=(12, 5))
                dendrogram(Z, labels=labels[:10])
                plt.title("Phylogenetic Tree")
                plt.tight_layout()
                plt.savefig(tree_path)
                plt.close()
        except:
            pass

        # -------------------------------
        lineage_df = pd.DataFrame({"Lineage": lineage_list})
        lineage_summary = lineage_df.value_counts().reset_index(name="Count")

        top_positions = df["position"].value_counts().head(10).reset_index()
        top_positions.columns = ["Position", "Count"]

        top_types = df["type"].value_counts().head(10).reset_index()
        top_types.columns = ["Mutation", "Count"]

        # -------------------------------
        # SAVE CSV (NO AUTO DOWNLOAD)
        df.to_csv(os.path.join(UPLOAD_FOLDER, "report.csv"), index=False)

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "chart": "/uploads/chart.html",
            "heatmap": "/uploads/heatmap.html",
            "tree": "/uploads/tree.png",
            "csv": "/download/csv",

            "total_sequences": total_sequences,
            "total_mutations": total_mutations,
            "unique_mutations": unique_mutations,
            "avg_mutations": avg_mutations,

            "top_positions": top_positions.to_dict(orient="records"),
            "top_types": top_types.to_dict(orient="records"),
            "lineage_summary": lineage_summary.to_dict(orient="records")
        })

    except Exception as e:
        return HTMLResponse(f"❌ ERROR: {str(e)}")

# -------------------------------
@app.get("/download/csv")
def download_csv():
    return FileResponse("uploads/report.csv", filename="report.csv")