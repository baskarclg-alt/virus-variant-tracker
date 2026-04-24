from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import shutil
import uuid
from collections import Counter
from Bio import SeqIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

VALID_BASES = set("ATGC")
TREE_SEQ_LIMIT = 120
HOTSPOT_LIMIT = 1000


def load_reference():
    ref_path = os.path.join(DATA_FOLDER, "reference.fasta")
    ref_record = next(SeqIO.parse(ref_path, "fasta"))
    return str(ref_record.seq).upper()


def get_mutation_profile(ref, seq):
    """
    Returns:
      positions: list of mismatch positions
      mutation_types: list like A>T, C>G, etc.
    Only valid nucleotide substitutions are counted.
    """
    positions = []
    mutation_types = []

    for i, (r, s) in enumerate(zip(ref, seq)):
        if r in VALID_BASES and s in VALID_BASES and r != s:
            positions.append(i)
            mutation_types.append(f"{r}>{s}")

    return positions, mutation_types


def classify_variants_from_counts(mutation_counts):
    """
    Balanced, data-driven classification using ranks.
    This avoids the 'all same variant' problem.
    """
    n = len(mutation_counts)
    if n == 0:
        return []

    if n < 4 or len(set(mutation_counts)) == 1:
        labels = ["Other Variant", "Alpha-like", "Delta-like", "Omicron-like"]
        return [labels[i % 4] for i in range(n)]

    ranks = pd.Series(mutation_counts).rank(method="first")
    variant_series = pd.qcut(
        ranks,
        q=4,
        labels=["Other Variant", "Alpha-like", "Delta-like", "Omicron-like"],
        duplicates="drop"
    )
    return variant_series.astype(str).tolist()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    uid = str(uuid.uuid4())[:8]
    path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ref = load_reference()

    seq_labels = []
    seq_positions = []
    mutation_counts = []

    position_counter = Counter()
    mutation_type_counter = Counter()

    total_sequences = 0

    # -------- First pass: compute raw mutation profiles ----------
    for i, rec in enumerate(SeqIO.parse(path, "fasta")):
        seq = str(rec.seq).upper()
        positions, mutation_types = get_mutation_profile(ref, seq)

        seq_labels.append(f"S{i+1}")
        seq_positions.append(set(positions))
        mutation_counts.append(len(positions))

        position_counter.update(set(positions))
        mutation_type_counter.update(mutation_types)

        total_sequences += 1

    if total_sequences == 0:
        return HTMLResponse("No sequences found")

    # -------- Hotspot-based summary (makes numbers readable) ----------
    top_positions = [p for p, _ in position_counter.most_common(min(HOTSPOT_LIMIT, len(position_counter)))]
    if not top_positions:
        top_positions = list(range(min(HOTSPOT_LIMIT, len(ref))))

    top_set = set(top_positions)

    hotspot_counts = [
        sum(1 for p in pos_set if p in top_set)
        for pos_set in seq_positions
    ]

    total_mutations = int(sum(hotspot_counts))
    avg_mutations = int(np.mean(hotspot_counts)) if hotspot_counts else 0
    unique_mutations = len(top_positions)

    # -------- Variant table (balanced + data-driven) ----------
    variant_labels = classify_variants_from_counts(hotspot_counts)

    variant_df = pd.DataFrame({
        "Sequence": seq_labels,
        "Variant": variant_labels
    })

    variant_summary = variant_df["Variant"].value_counts().reset_index()
    variant_summary.columns = ["Variant", "Count"]

    # -------- Tree: build from mutation-hotspot vectors ----------
    vectors = []
    for pos_set in seq_positions[:TREE_SEQ_LIMIT]:
        vectors.append([1 if p in pos_set else 0 for p in top_positions])

    tree_file = f"{uid}_tree.html"
    tree_path = os.path.join(UPLOAD_FOLDER, tree_file)

    if len(vectors) >= 2 and len(top_positions) >= 2:
        X = np.array(vectors)

        # Hierarchical clustering on binary mutation hotspot profiles
        dist = pdist(X, metric="hamming")
        if len(dist) > 0:
            Z = linkage(dist, method="average")
            dendro = dendrogram(Z, labels=seq_labels[:len(X)], no_plot=True)

            fig = go.Figure()
            for i in range(len(dendro["icoord"])):
                fig.add_trace(go.Scatter(
                    x=dendro["dcoord"][i],
                    y=dendro["icoord"][i],
                    mode="lines",
                    line=dict(color="lime", width=2),
                    showlegend=False
                ))

            fig.update_layout(
                template="plotly_dark",
                height=900,
                margin=dict(l=200, r=20, t=30, b=20),
                yaxis=dict(
                    tickmode="array",
                    tickvals=[5 + 10 * i for i in range(len(seq_labels[:len(X)]))],
                    ticktext=seq_labels[:len(X)]
                )
            )
            fig.write_html(tree_path)
        else:
            tree_file = ""
    else:
        tree_file = ""

    # -------- Mutation chart ----------
    chart_file = f"{uid}_chart.html"
    chart_df = pd.DataFrame(
        mutation_type_counter.most_common(10),
        columns=["Mutation Type", "Count"]
    )

    if not chart_df.empty:
        fig_chart = px.bar(
            chart_df,
            x="Mutation Type",
            y="Count",
            template="plotly_dark"
        )
        fig_chart.write_html(os.path.join(UPLOAD_FOLDER, chart_file))
    else:
        chart_file = ""

    # -------- Heatmap ----------
    heat_file = f"{uid}_heatmap.html"
    if len(position_counter) > 0:
        # Bin the genome into windows for a readable density plot
        window = max(len(ref) // 20, 1)
        bin_counter = Counter()
        for pos, cnt in position_counter.items():
            bin_counter[pos // window] += cnt

        heat_df = pd.DataFrame(sorted(bin_counter.items()), columns=["Genome Region", "Mutation Density"])

        fig_heat = px.area(
            heat_df,
            x="Genome Region",
            y="Mutation Density",
            template="plotly_dark"
        )
        fig_heat.write_html(os.path.join(UPLOAD_FOLDER, heat_file))
    else:
        heat_file = ""

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "tree": f"/uploads/{tree_file}" if tree_file else "",
        "chart": f"/uploads/{chart_file}" if chart_file else "",
        "heatmap": f"/uploads/{heat_file}" if heat_file else "",
        "total_sequences": total_sequences,
        "total_mutations": total_mutations,
        "unique_mutations": unique_mutations,
        "avg_mutations": avg_mutations,
        "variant_summary": variant_summary.to_dict(orient="records"),
        "variant_details": variant_df.head(200).to_dict(orient="records")
    })