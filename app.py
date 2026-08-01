from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from database import (
    init_db,
    create_dossier,
    add_piece_jointe,
    list_dossiers,
    get_piece_jointe,
)
from agents import get_agent
from outils import read_file_text


app = FastAPI()

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Création des tables si elles n'existent pas encore
init_db()


# ---------------------------------------------------------------------------
# Page d'accueil (interface)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Endpoint de génération d'un acte
#   - reçoit les pièces jointes + l'utilisateur
#   - sauvegarde les fichiers, extrait le texte
#   - lance l'agent
#   - met à jour la base (dossier + pièces jointes)
#   - renvoie l'acte généré
# ---------------------------------------------------------------------------
@app.post("/generer")
async def generer_acte(
    files: list[UploadFile] = File(...),
    utilisateur_id: int = Form(...),   # TODO: viendra de l'authentification plus tard
):
    try:
        pieces = []      # couples (nom_fichier, chemin) pour la base
        textes = []      # textes extraits, pour l'agent

        # 1. Sauvegarde des fichiers + extraction du texte
        for file in files:
            filename = file.filename or "piece.pdf"
            file_id = str(uuid.uuid4())
            safe_name = filename.replace(" ", "_")
            chemin = f"uploads/{file_id}_{safe_name}"

            with open(chemin, "wb") as f:
                f.write(await file.read())

            texte = read_file_text(chemin)
            textes.append(f"===== FILE: {filename} =====\n{texte}")
            pieces.append((filename, chemin))

        source_text = "\n\n".join(textes)

        # 2. Génération de l'acte par l'agent
        agent = get_agent()
        result = agent.invoke({"messages": source_text})
        status = result.get("status", "ok")

        # 3. Création du dossier en base (1re table)
        if status == "need_more_info":
            dossier = create_dossier(
                utilisateur_id=utilisateur_id,
                type_acte="",
                source_text=source_text,
                acte_genere="",
                statut="need_more_info",
            )
        else:
            dossier = create_dossier(
                utilisateur_id=utilisateur_id,
                type_acte=result.get("act_type", ""),
                source_text=source_text,
                acte_genere=result.get("final_act", ""),
                statut="genere",
            )

        # 4. Enregistrement des pièces jointes (2e table)
        for nom, chemin in pieces:
            add_piece_jointe(dossier.id, nom, chemin)

        # 5. Réponse au front
        if status == "need_more_info":
            return JSONResponse({
                "status": "need_more_info",
                "dossier_id": dossier.id,
                "message": result.get("message", "Informations insuffisantes."),
            })

        return JSONResponse({
            "status": "ok",
            "dossier_id": dossier.id,
            "type_acte": result.get("act_type", ""),
            "acte": result.get("final_act", ""),
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Endpoint d'historique des dossiers d'un utilisateur
# ---------------------------------------------------------------------------
@app.get("/historique/{utilisateur_id}")
async def historique(utilisateur_id: int):
    dossiers = list_dossiers(utilisateur_id)

    return {
        "dossiers": [
            {
                "id": d.id,
                "type_acte": d.type_acte,
                "statut": d.statut,
                "created_at": d.created_at.isoformat(),
                "pieces_jointes": [
                    {"id": p.id, "nom": p.nom_fichier}
                    for p in d.pieces_jointes
                ],
            }
            for d in dossiers
        ]
    }


# ---------------------------------------------------------------------------
# Endpoint de téléchargement d'une pièce jointe
# ---------------------------------------------------------------------------
@app.get("/piece/{piece_id}")
async def download_piece(piece_id: int):
    piece = get_piece_jointe(piece_id)

    if not piece:
        return JSONResponse({"error": "Pièce introuvable."}, status_code=404)

    # TODO: vérifier que la pièce appartient bien à l'utilisateur connecté
    return FileResponse(
        path=piece.chemin,
        filename=piece.nom_fichier,
        media_type="application/pdf",
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )