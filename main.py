from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):

    file_location = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ref_record = next(SeqIO.parse("data/reference.fasta", "fasta"))
    reference = str(ref_record.seq)

    sequences = list(SeqIO.parse(file_location, "fasta"))

    if len(sequences) == 0:
        return HTMLResponse("No sequences found")

    uploaded_length = len(sequences[0].seq)

    mutation_positions = set()
    mutation_table = []
    mutation_freq = {}

    for record in sequences:
        seq = str(record.seq)
        max_len = min(len(reference), len(seq))

        for i in range(max_len):
            ref_base = reference[i]
            seq_base = seq[i]

            if seq_base not in ["A","T","G","C"]:
                continue

            if ref_base != seq_base:
                mutation_positions.add(i+1)
                mutation_freq[i+1] = mutation_freq.get(i+1, 0) + 1

    for pos in sorted(mutation_positions):
        ref_base = reference[pos-1]
        mut_set = set()

        for record in sequences:
            seq = str(record.seq)
            if pos-1 < len(seq):
                base = seq[pos-1]
                if base in ["A","T","G","C"] and base != ref_base:
                    mut_set.add(base)

        mutation_table.append({
            "position": pos,
            "ref": ref_base,
            "mut": list(mut_set)[0] if len(mut_set) == 1 else "varies"
        })

    mutation_count = len(mutation_positions)

    if mutation_freq:
        plt.figure()
        plt.bar(list(mutation_freq.keys()), list(mutation_freq.values()))
        plt.xlabel("Genome Position")
        plt.ylabel("Mutation Frequency")
        plt.title("Mutation Frequency Chart")

        chart_path = f"{STATIC_FOLDER}/chart.png"
        plt.savefig(chart_path)
        plt.close()
    else:
        chart_path = None

    try:
        heatmap_sequences = sequences[:10]
        heatmap_data = []

        important_positions = list(mutation_positions)[:100]

        for record in heatmap_sequences:
            seq = str(record.seq)
            row = []

            for pos in important_positions:
                i = pos - 1
                if i < len(seq) and i < len(reference):
                    row.append(1 if seq[i] != reference[i] else 0)
                else:
                    row.append(0)

            heatmap_data.append(row)

        if len(heatmap_data) > 0:
            heatmap_array = np.array(heatmap_data)

            plt.figure(figsize=(12,5))
            plt.imshow(heatmap_array, aspect='auto')
            plt.colorbar(label="Mutation")

            heatmap_path = f"{STATIC_FOLDER}/heatmap.png"
            plt.savefig(heatmap_path)
            plt.close()
        else:
            heatmap_path = None

    except Exception as e:
        print("HEATMAP ERROR:", e)
        heatmap_path = None

    try:
        import plotly.graph_objects as go

        tree_sequences = sequences[:8]

        labels = []
        distances = []

        for rec in tree_sequences:
            seq = str(rec.seq)
            max_len = min(len(reference), len(seq))

            diff = sum(1 for i in range(max_len) if seq[i] != reference[i])

            labels.append(rec.id[:10])
            distances.append(diff)

        if len(labels) >= 2:
            fig = go.Figure()

            fig.add_trace(go.Bar(x=labels, y=distances))

            fig.update_layout(
                title="Phylogenetic Distance",
                xaxis_title="Sequences",
                yaxis_title="Mutation Distance"
            )

            tree_html_path = f"{STATIC_FOLDER}/tree.html"
            fig.write_html(tree_html_path)
        else:
            tree_html_path = None

    except Exception as e:
        print("TREE ERROR:", e)
        tree_html_path = None

    results = {
        "sequence_name": file.filename,
        "sequence_length": uploaded_length,
        "reference_length": len(reference),
        "mutation_count": mutation_count,
        "mutation_table": mutation_table[:50],
        "chart": "/static/chart.png" if chart_path else None,
        "heatmap": "/static/heatmap.png" if heatmap_path else None,
        "tree_html": "/static/tree.html" if tree_html_path else None
    }

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "results": results}
    )