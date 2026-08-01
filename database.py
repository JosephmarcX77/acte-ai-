from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Dossier où sera rangé le fichier de base
Path("data").mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///data/acte_ai.db"


# Connexion à la base SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Fabrique de sessions
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Socle des modèles
Base = declarative_base()


# ---------------------------------------------------------------------------
# Table utilisateur (pour la connexion)
# ---------------------------------------------------------------------------
class Utilisateur(Base):
    __tablename__ = "utilisateurs"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    mot_de_passe = Column(String, nullable=False)   # toujours HACHÉ, jamais en clair
    created_at = Column(DateTime, default=datetime.utcnow)

    # Lien vers les dossiers de cet utilisateur
    dossiers = relationship("Dossier", back_populates="utilisateur")


# ---------------------------------------------------------------------------
# Table dossier (actes générés + historique)
# ---------------------------------------------------------------------------
class Dossier(Base):
    __tablename__ = "dossiers"

    id = Column(Integer, primary_key=True, index=True)
    utilisateur_id = Column(Integer, ForeignKey("utilisateurs.id"), nullable=False)

    type_acte = Column(String)          # type d'acte détecté par l'agent
    source_text = Column(Text)          # texte extrait des pièces (optionnel)
    acte_genere = Column(Text)          # acte final produit par l'agent
    statut = Column(String, default="genere")   # "genere" ou "need_more_info"

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Liens
    utilisateur = relationship("Utilisateur", back_populates="dossiers")
    pieces_jointes = relationship("PieceJointe", back_populates="dossier")


# ---------------------------------------------------------------------------
# Table pièce jointe (fichiers uploadés, liés à un dossier)
# ---------------------------------------------------------------------------
class PieceJointe(Base):
    __tablename__ = "pieces_jointes"

    id = Column(Integer, primary_key=True, index=True)
    dossier_id = Column(Integer, ForeignKey("dossiers.id"), nullable=False)

    nom_fichier = Column(String, nullable=False)   # nom original, ex: "bulletin.pdf"
    chemin = Column(String, nullable=False)        # chemin sur disque, ex: "uploads/abc_bulletin.pdf"
    created_at = Column(DateTime, default=datetime.utcnow)

    # Lien vers le dossier parent
    dossier = relationship("Dossier", back_populates="pieces_jointes")


# ---------------------------------------------------------------------------
# Initialisation : crée les tables si elles n'existent pas encore
# ---------------------------------------------------------------------------
def init_db():
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Historique des dossiers d'un utilisateur (du plus récent au plus ancien)
# ---------------------------------------------------------------------------
def list_dossiers(utilisateur_id: int):
    db = SessionLocal()
    try:
        dossiers = (
            db.query(Dossier)
            .filter(Dossier.utilisateur_id == utilisateur_id)
            .order_by(Dossier.created_at.desc())
            .all()
        )
        return dossiers
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Enregistrer une pièce jointe liée à un dossier
# ---------------------------------------------------------------------------
def add_piece_jointe(dossier_id: int, nom_fichier: str, chemin: str):
    db = SessionLocal()
    try:
        piece = PieceJointe(
            dossier_id=dossier_id,
            nom_fichier=nom_fichier,
            chemin=chemin,
        )
        db.add(piece)
        db.commit()
        db.refresh(piece)
        return piece
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Récupérer une pièce jointe par son id (pour le téléchargement)
# ---------------------------------------------------------------------------
def get_piece_jointe(piece_id: int):
    db = SessionLocal()
    try:
        return (
            db.query(PieceJointe)
            .filter(PieceJointe.id == piece_id)
            .first()
        )
    finally:
        db.close()




# ---------------------------------------------------------------------------
# Créer un dossier (au moment de la génération d'un acte)
# ---------------------------------------------------------------------------
def create_dossier(utilisateur_id: int, type_acte: str, source_text: str,
                   acte_genere: str, statut: str = "genere"):
    db = SessionLocal()
    try:
        dossier = Dossier(
            utilisateur_id=utilisateur_id,
            type_acte=type_acte,
            source_text=source_text,
            acte_genere=acte_genere,
            statut=statut,
        )
        db.add(dossier)
        db.commit()
        db.refresh(dossier)   # recharge pour récupérer l'id généré par la base
        return dossier
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("Base de données initialisée : data/acte_ai.db")