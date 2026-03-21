from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from Bio import SeqIO
import pandas as pd
import plotly.express as px

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 🔥 SAFE FOLDERS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# -------------------------------
def load_reference():
    try:
        ref_path = os.path.join(DATA_FOLDER, "reference.fasta")
        return str(next(SeqIO.parse(ref_path, "fasta")).seq)
    except Exception as e:
        print("Reference Error:", e)
        return None

# -------------------------------
def find_mutations(ref, seq):
    muts, table = [], []

    for i in range(min(len(ref), len(seq))):
        if ref[i] != seq[i]:
            m = f"{ref[i]}{i+1}{seq[i]}"
            muts.append(m)

            table.append({
                "position": i+1,
                "ref": ref[i],
                "mut": seq[i]
            })

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

        sequences = list(SeqIO.parse(path, "fasta"))

        if not sequences:
            return HTMLResponse("❌ No sequences found in file")

        ref = load_reference()

        if ref is None:
            return HTMLResponse("❌ reference.fasta missing")

        table = []
        all_muts = set()
        seq_lengths = []

        for s in sequences:
            seq = str(s.seq)
            seq_lengths.append(len(seq))

            muts, t = find_mutations(ref, seq)

            all_muts.update(muts)
            table.extend(t)

        if len(table) == 0:
            return HTMLResponse("⚠️ No mutations found (check your reference file)")

        df = pd.DataFrame(table)

        # 🔥 STATS
        total_sequences = len(sequences)
        total_mutations = len(all_muts)
        avg_length = sum(seq_lengths) // len(seq_lengths)
        ref_length = len(ref)

        df_display = df.head(100)

        # -------------------------------
        # 📊 Chart
        chart_path = os.path.join(UPLOAD_FOLDER, "chart.html")

        chart = px.bar(
            df["mut"].value_counts().head(10),
            title="Top Mutations"
        )
        chart.write_html(chart_path)

        # -------------------------------
        # 🔥 Density
        heat_path = os.path.join(UPLOAD_FOLDER, "heatmap.html")

        heat = px.histogram(
            x=df["position"],
            nbins=50,
            title="Mutation Density"
        )
        heat.write_html(heat_path)

        # -------------------------------
        # 📥 CSV
        csv_path = os.path.join(UPLOAD_FOLDER, "report.csv")
        df.to_csv(csv_path, index=False)

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "chart": "/uploads/chart.html",
            "heatmap": "/uploads/heatmap.html",
            "table": df_display.to_dict(orient="records"),
            "csv": "/uploads/report.csv",

            # 🔥 NEW DATA
            "total_sequences": total_sequences,
            "total_mutations": total_mutations,
            "avg_length": avg_length,
            "ref_length": ref_length
        })

    except Exception as e:
        print("ERROR:", e)
        return HTMLResponse(f"❌ Server Error: {str(e)}")