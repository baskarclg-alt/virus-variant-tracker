from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from Bio import SeqIO
import pandas as pd
import shutil
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):

    # Save uploaded FASTA
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read FASTA sequence length
    sequence_length = 0
    for record in SeqIO.parse(file_location, "fasta"):
        sequence_length = len(record.seq)
        break

    # Read Nextclade CSV
    try:
        df = pd.read_csv("data/nextclade_results.csv", sep=";")

        print(df.head())
        print(df.columns)

        mutation_count = int(df["totalSubstitutions"].sum())
        clade = df["clade"].iloc[0]
        qc_score = df["qc.overallStatus"].iloc[0]

    except Exception as e:
        print("CSV ERROR:", e)
        mutation_count = 0
        clade = "Unknown"
        qc_score = "Error"

    mutation_table = [
        {
            "position": "Substitutions",
            "ref": "-",
            "mut": mutation_count
        }
    ]

    results = {
        "sequence_name": file.filename,
        "sequence_length": sequence_length,
        "mutations": mutation_count,
        "clade": clade,
        "qc_score": qc_score,
        "mutation_table": mutation_table
    }

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "results": results}
    )