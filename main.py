from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os, shutil, uuid
from Bio import SeqIO
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")


# -------------------------------
# ✅ HOME ROUTE (FIX NOT FOUND)
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------------------------------
def load_reference():
    ref_path = os.path.join(DATA_FOLDER, "reference.fasta")
    ref = next(SeqIO.parse(ref_path, "fasta"))
    return str(ref.seq).upper()


# -------------------------------
def get_mutations(ref, seq):
    muts = []
    pos = []

    for i in range(min(len(ref), len(seq))):
        r = ref[i]
        s = seq[i]

        if r in "ATGC" and s in "ATGC" and r != s:
            muts.append(f"{r}>{s}_{i}")
            pos.append(i)

    return muts, pos


# -------------------------------
def classify_variants(mutation_counts):
    df = pd.DataFrame({"count": mutation_counts})

    q1 = df["count"].quantile(0.25)
    q2 = df["count"].quantile(0.50)
    q3 = df["count"].quantile(0.75)

    variants = []

    for c in mutation_counts:
        if c <= q1:
            variants.append("Alpha-like")
        elif c <= q2:
            variants.append("Delta-like")
        elif c <= q3:
            variants.append("Omicron-like")
        else:
            variants.append("Other Variant")

    return variants


# -------------------------------
@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):

    uid = str(uuid.uuid4())[:8]
    path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ref = load_reference()

    all_muts = []
    all_pos = []
    mutation_counts = []
    labels = []

    for i, rec in enumerate(SeqIO.parse(path, "fasta")):
        seq = str(rec.seq).upper()

        muts, pos = get_mutations(ref, seq)

        mutation_counts.append(len(muts))
        all_muts.extend(muts)
        all_pos.extend(pos)

        labels.append(f"S{i+1}")

    total_seq = len(labels)
    total_mut = sum(mutation_counts)
    unique_mut = len(set(all_muts))
    avg_mut = int(np.mean(mutation_counts)) if total_seq > 0 else 0

    # ---------------- VARIANT
    variants = classify_variants(mutation_counts)

    variant_df = pd.DataFrame({
        "Sequence": labels,
        "Variant": variants
    })

    summary = variant_df["Variant"].value_counts().reset_index()
    summary.columns = ["Variant", "Count"]

    # ---------------- TREE
    tree_file = f"{uid}_tree.html"
    tree_path = os.path.join(UPLOAD_FOLDER, tree_file)

    vectors = []

    for rec in SeqIO.parse(path, "fasta"):
        seq = str(rec.seq).upper()

        vec = []
        for i in range(min(len(ref), len(seq))):
            r = ref[i]
            s = seq[i]

            if r in "ATGC" and s in "ATGC" and r != s:
                vec.append(1)
            else:
                vec.append(0)

        vectors.append(vec[:1000])

    if len(vectors) > 2:
        X = np.array(vectors[:100])

        dist = pdist(X, metric="hamming")
        Z = linkage(dist, method='ward')

        d = dendrogram(Z, labels=labels[:100], no_plot=True)

        import plotly.graph_objects as go
        fig = go.Figure()

        for i in range(len(d['icoord'])):
            fig.add_trace(go.Scatter(
                x=d['dcoord'][i],
                y=d['icoord'][i],
                mode='lines',
                line=dict(color="lime"),
                showlegend=False
            ))

        fig.update_layout(template="plotly_dark", height=800)
        fig.write_html(tree_path)

    # ---------------- CHART
    chart_file = f"{uid}_chart.html"

    px.bar(pd.Series(all_muts).value_counts().head(10),
           template="plotly_dark").write_html(os.path.join(UPLOAD_FOLDER, chart_file))

    # ---------------- HEATMAP
    heat_file = f"{uid}_heatmap.html"

    heat_df = pd.DataFrame({"pos": all_pos})
    heat_df["bin"] = heat_df["pos"] // 500

    heat_counts = heat_df["bin"].value_counts().sort_index()

    px.area(x=heat_counts.index,
            y=heat_counts.values,
            template="plotly_dark").write_html(os.path.join(UPLOAD_FOLDER, heat_file))

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "tree": f"/uploads/{tree_file}",
        "chart": f"/uploads/{chart_file}",
        "heatmap": f"/uploads/{heat_file}",

        "total_sequences": total_seq,
        "total_mutations": total_mut,
        "unique_mutations": unique_mut,
        "avg_mutations": avg_mut,

        "variant_summary": summary.to_dict(orient="records"),
        "variant_details": variant_df.to_dict(orient="records")
    })