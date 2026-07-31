from typing import TypedDict, Annotated
import json
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

my_key = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Seuil de confiance sous lequel on refuse de générer (garde-fou anti-hallucination)
CONFIDENCE_THRESHOLD = 0.6


def normalize_model_name(model_name: str | None = None) -> str:
    """
    Return a valid model name.
    Falls back to DEFAULT_MODEL if nothing (or an empty value) is provided.
    """
    if not model_name or not model_name.strip():
        return DEFAULT_MODEL
    return model_name.strip()


def parse_json(raw: str) -> dict:
    """
    Parse une réponse LLM censée être du JSON, en tolérant d'éventuelles
    balises Markdown ```json ... ``` autour.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip().strip("`").strip()
    return json.loads(cleaned)


# Définition du state

class AgentState(TypedDict):
    messages: str         # texte concaténé des pièces jointes (extrait côté FastAPI)
    identification: dict   # résultat du tri des pièces
    act_type: str          # type d'acte détecté
    confidence: float      # confiance de la détection (0.0 à 1.0)
    detection_reason: str  # explication de la détection
    entities: dict         # infos extraites (parties, dates, montants...)
    missing_info: list     # infos essentielles manquantes
    draft: str             # brouillon de l'acte
    final_act: str         # acte final (après relecture)
    status: str            # "ok" ou "need_more_info"
    message: str           # message à l'utilisateur si need_more_info



def build_agent(model_name):
    model_name = normalize_model_name(model_name)
    llm = ChatOpenAI(model=model_name, api_key=my_key)

    # Noeud d'identification de pièce jointe
    def identification(state):
        PROMPT_TRI_PIECES = """Tu es un juriste expérimenté qui reçoit un dossier composé de plusieurs pièces jointes. Ta tâche est la première étape du traitement : la prise de connaissance et le tri des pièces.

    À partir du texte extrait des documents fournis, tu dois :

    1. Identifier et classer chaque pièce selon sa nature. Catégories possibles (liste non exhaustive) :
    - pièce d'identité (CNI, passeport, titre de séjour)
    - justificatif (domicile, revenus, situation)
    - titre de propriété
    - contrat existant
    - correspondance (courrier, e-mail, mise en demeure reçue)
    - mandat ou lettre de mission
    - document officiel (extrait Kbis, acte d'état civil, jugement)
    - autre (à préciser)

    2. Écarter les pièces manifestement non pertinentes pour la constitution d'un dossier juridique, en indiquant lesquelles et pourquoi.

    3. Repérer les pièces manquantes évidentes, c'est-à-dire les documents qu'on s'attendrait normalement à trouver dans un tel dossier mais qui sont absents.

    Règles impératives :
    - N'invente aucune pièce ni aucune information. Fonde-toi uniquement sur le texte fourni.
    - Si la nature d'une pièce est incertaine, classe-la en "autre" et signale l'incertitude plutôt que de deviner.
    - Reste factuel : décris ce que tu observes, ne tire pas de conclusions juridiques à ce stade.

    Réponds UNIQUEMENT en JSON strict, sans texte autour, au format suivant :
    {"pieces_identifiees": [
        {"nom": "<nom ou description de la pièce>", "categorie": "<catégorie>", "text": "<texte de la pièce jointe>"}
    ]}
    """

        system = SystemMessage(content=PROMPT_TRI_PIECES)
        human = HumanMessage(content=(
            f"Voici le texte extrait des documents fournis :\n\n{state['messages']}"
        ))

        message_final = [system, human]

        response = llm.invoke(message_final)

        try:
            data = parse_json(response.content)
        except Exception:
            data = {"pieces_identifiees": []}

        return {"identification": data}

    # Noeud de détection du type d'acte
    def detect_act_type(state):
        PROMPT_DETECTION = """Tu es un expert juridique chargé d'identifier le type d'acte à produire à partir des documents fournis.

    Cherche un document déclencheur (modèle d'acte, acte de référence, mandat, lettre de mission) qui indique le type d'acte attendu.

    Réponds UNIQUEMENT en JSON strict, sans texte autour, au format suivant :
    {"act_type": "<type d'acte ou vide>", "confidence": <nombre entre 0 et 1>, "reason": "<explication courte>"}

    Si aucun document ne permet d'identifier le type d'acte avec certitude, mets une confidence basse (< 0.6) et explique pourquoi. N'invente jamais un type d'acte pour combler un doute.
    """

        system = SystemMessage(content=PROMPT_DETECTION)
        human = HumanMessage(content=(
            f"Voici le texte extrait des documents fournis :\n\n{state['messages']}"
        ))

        message_final = [system, human]

        response = llm.invoke(message_final)

        try:
            data = parse_json(response.content)
            act_type = str(data.get("act_type", "")).strip()
            confidence = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", "")).strip()
        except Exception:
            act_type = ""
            confidence = 0.0
            reason = "Échec de l'analyse du type d'acte."

        return {
            "act_type": act_type,
            "confidence": confidence,
            "detection_reason": reason,
        }

    # Routage : continuer ou demander une précision
    def route_on_confidence(state):
        if state.get("act_type") and state.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD:
            return "extract_entities"
        return "ask_for_precision"

    # Noeud impasse : confiance insuffisante
    def ask_for_precision(state):
        reason = state.get("detection_reason", "")
        message = (
            "Je n'ai pas pu déterminer avec certitude le type d'acte à générer à "
            "partir des documents fournis. Merci de préciser le type d'acte souhaité "
            "ou de fournir un document de référence (modèle, mandat, lettre de mission)."
        )
        if reason:
            message += f"\n\nAnalyse : {reason}"

        return {"status": "need_more_info", "message": message}

    # Noeud d'extraction des informations du dossier
    def extract_entities(state):
        PROMPT_EXTRACTION = f"""Tu es un juriste qui prépare un acte de type « {state.get('act_type', '')} ».

    Extrais des documents fournis toutes les informations utiles à la rédaction de cet acte : identité et coordonnées des parties, dates, durées, montants, adresses, références, et toute clause spécifique.

    Réponds UNIQUEMENT en JSON strict, sans texte autour, au format suivant :
    {{"entities": {{"<clé>": "<valeur>"}}, "missing_info": ["<info essentielle manquante>"]}}

    N'invente aucune information : si une donnée essentielle pour ce type d'acte est absente des documents, liste-la dans missing_info.
    """

        system = SystemMessage(content=PROMPT_EXTRACTION)
        human = HumanMessage(content=(
            f"Voici le texte extrait des documents fournis :\n\n{state['messages']}"
        ))

        message_final = [system, human]

        response = llm.invoke(message_final)

        try:
            data = parse_json(response.content)
            entities = data.get("entities", {}) or {}
            missing = data.get("missing_info", []) or []
        except Exception:
            entities = {}
            missing = []

        return {"entities": entities, "missing_info": missing}

    # Noeud de rédaction de l'acte
    def draft_act(state):
        entities_text = json.dumps(state.get("entities", {}), ensure_ascii=False, indent=2)
        missing = state.get("missing_info", [])
        missing_text = ", ".join(missing) if missing else "aucune"

        PROMPT_REDACTION = f"""Tu es un juriste rédacteur. Rédige un acte de type « {state.get('act_type', '')} » complet, professionnel et structuré, en français.

    Structure attendue : identification des parties, préambule/exposé, articles numérotés, mentions obligatoires, date et emplacements de signature.

    Règles impératives :
    - N'invente jamais d'information (noms, montants, dates, articles de loi). Utilise uniquement les informations fournies.
    - Pour toute information essentielle manquante, insère un champ à compléter clairement visible, du type [À COMPLÉTER : ...].
    - Informations essentielles signalées comme manquantes : {missing_text}.
    - Termine par une mention indiquant que ce document est un projet devant être relu par un professionnel du droit et ne constitue pas un conseil juridique.

    Rends uniquement le texte de l'acte, sans commentaire autour.
    """

        system = SystemMessage(content=PROMPT_REDACTION)
        human = HumanMessage(content=(
            f"Informations extraites du dossier :\n{entities_text}\n\n"
            f"Documents sources (pour contexte) :\n{state['messages']}"
        ))

        message_final = [system, human]

        response = llm.invoke(message_final)

        return {"draft": response.content}

    # Noeud de relecture / contrôle de cohérence
    def review_act(state):
        PROMPT_RELECTURE = f"""Tu es un juriste relecteur. Relis le projet d'acte de type « {state.get('act_type', '')} » ci-dessous et corrige : incohérences de dates ou de montants, mentions obligatoires manquantes, formulations ambiguës, erreurs de structure.

    Règles :
    - Ne supprime aucun champ [À COMPLÉTER : ...] : ils doivent rester visibles.
    - N'ajoute aucune information inventée.
    - Conserve la mention finale sur le caractère de projet à faire relire.

    Rends uniquement la version corrigée et finale de l'acte, sans commentaire.
    """

        system = SystemMessage(content=PROMPT_RELECTURE)
        human = HumanMessage(content=state.get("draft", ""))

        message_final = [system, human]

        response = llm.invoke(message_final)

        return {"final_act": response.content, "status": "ok"}

    # Construction du graphe
    graph = StateGraph(AgentState)

    graph.add_node("identification", identification)
    graph.add_node("detect_act_type", detect_act_type)
    graph.add_node("ask_for_precision", ask_for_precision)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("draft_act", draft_act)
    graph.add_node("review_act", review_act)

    graph.add_edge(START, "identification")
    graph.add_edge("identification", "detect_act_type")

    graph.add_conditional_edges(
        "detect_act_type",
        route_on_confidence,
        {
            "extract_entities": "extract_entities",
            "ask_for_precision": "ask_for_precision",
        },
    )

    graph.add_edge("ask_for_precision", END)
    graph.add_edge("extract_entities", "draft_act")
    graph.add_edge("draft_act", "review_act")
    graph.add_edge("review_act", END)

    return graph.compile()


# Cache des agents par modèle
_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached agent for selected model.
    If not created yet, create it once and reuse it.
    """
    selected_model = normalize_model_name(model_name)
    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)
    return _AGENT_CACHE[selected_model]