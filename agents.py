import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


Path("data").mkdir(exist_ok=True)


# prompt

SYSTEM_PROMPT = """Tu es un assistant juridique spécialisé dans la génération et l'analyse d'actes juridiques (contrats, statuts, procès-verbaux, baux, attestations, mises en demeure, etc.).

Tu peux :

1. Répondre aux questions juridiques générales de manière claire et pédagogique.
2. Utiliser des outils quand c'est nécessaire.
3. Rechercher dans les documents PDF fournis par l'utilisateur (modèles d'actes, pièces, contrats existants) via l'outil de recherche documentaire.
4. Rechercher des informations juridiques à jour sur le web (textes de loi récents, jurisprudence) via la recherche web.
5. Mémoriser les informations importantes du dossier (parties, dates, montants, clauses spécifiques) via l'outil de mémoire.
6. Rappeler les informations mémorisées quand c'est utile.
7. Générer un acte juridique complet et structuré à partir des informations fournies.

Règles :
- Quand l'utilisateur demande de rédiger ou générer un acte, utilise generate_legal_act pour produire un document structuré et complet.
- Quand l'utilisateur pose une question sur un document PDF fourni (modèle, contrat, pièce), utilise search_uploaded_documents.
- Quand l'utilisateur demande une information juridique récente (loi, décret, jurisprudence, barème à jour), utilise web_search.
- Quand l'utilisateur te demande de retenir une information du dossier, utilise remember_this.
- Quand l'utilisateur se réfère à des informations déjà fournies, utilise recall_memory.
- Lorsque tu utilises la recherche web, résume clairement et précise que la réponse s'appuie sur des résultats web.
- N'invente jamais d'articles de loi, de références de jurisprudence, de noms de parties ou de montants : si une information nécessaire manque, demande-la explicitement à l'utilisateur avant de rédiger.
- Structure les actes de manière professionnelle (identification des parties, exposé/préambule, articles numérotés, mentions obligatoires, date et signatures).
- Précise systématiquement que le document généré est un projet à faire relire par un professionnel du droit et ne constitue pas un conseil juridique.
- Sois clair, précis et rigoureux.
"""


def normalize_model_name(model_name: str | None = None) -> str:
    """
    Return a valid model name.
    Falls back to DEFAULT_MODEL if nothing (or an empty value) is provided.
    """
    if not model_name or not model_name.strip():
        return DEFAULT_MODEL
    return model_name.strip()


# fonction pour construire l'agent

def build_agent(model_name: str | None = None):

    selected_model = normalize_model_name(model_name)

    llm = ChatOpenAI(model=selected_model, temperature=0.3, api_key=api_key)
    llm_with_tool = llm.bind_tools(tools=tools)

    # noeud de génération
    def chat_bot_node(state):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tool.invoke(messages)
        return {
            "messages": [response]
        }

    # tool node
    tool_node = ToolNode(tools)

    graph = StateGraph(MessagesState)

    graph.add_node("chatbot", chat_bot_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", tools_condition)
    graph.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    # on initialise le checkpointer sql
    checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)


# cache des agents par modèle

_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """
    selected_model = normalize_model_name(model_name)
    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)
    return _AGENT_CACHE[selected_model]