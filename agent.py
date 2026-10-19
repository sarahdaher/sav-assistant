from pathlib import Path
from typing import Annotated

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition

LLM_MODEL = "qwen2.5:7b"            
EMBEDDING_MODEL = "nomic-embed-text"  

llm = ChatOllama(model=LLM_MODEL, temperature=0)


#partie rag : index et chunking

documents = TextLoader("policy.txt", encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents(documents) #20 chevauchement pour le contexte

vector_store = Chroma.from_documents(documents=chunks,embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),collection_name="refund_policy",)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


#tests simules dans la vrai vie ce sera authentifie 

orders = {
    "45872": {"status": "delayed", "days_late": 10, "customer": "Sarah", "amount": 120},
    "12345": {"status": "delivered", "days_late": 0, "customer": "Sara", "amount": 80},
    "67890": {"status": "delayed", "days_late": 3, "customer": "Sar", "amount": 45},
    "99999": {"status": "lost", "days_late": 20, "customer": "Sarraa", "amount": 650},
}

USERS ={o["customer"] for o in orders.values()}


#state

class SAVState(MessagesState):
    # Messagesstate apporte messages (avec add_messages), on ajoute le client avec api
    user: str


#les tools
    
@tool
def search_refund_policy(question: str) -> str:
    """Recherche dans la politique officielle de remboursement
    (retards, commandes perdues, seuils de validation).
    À utiliser pour toute question sur les droits à remboursement."""
    results = retriever.invoke(question)
    if not results:
        return "Aucune information pertinente trouvée."
    return "\n\n".join(doc.page_content for doc in results)


@tool
def get_order_status(order_id: str, user: Annotated[str, InjectedState("user")]) -> str:
    """Donne le statut réel d'une commande du client connecte (statut, jours de retard, montant en euros) à partir de son numéro.
    À utiliser dès qu'un numéro de commande est mentionné."""
    # InjectedState : user est lu dans le state par langraph dans state[user] pour pas qu'il change dde client (ex : je suis sarah alors que je le suis pas, il appelle que order_id comme ca)
    order_id = order_id.strip()
    order = orders.get(order_id)


    if order is None or order["customer"] != user:
        return f"Commande {order_id} introuvable pour {user}"

    return (f"Commande {order_id}: statut={order['status']}, "f"retard={order['days_late']} jours, " f"montant={order['amount']} euros")


tools = [get_order_status, search_refund_policy]
llm_with_tools = llm.bind_tools(tools)


#llm

SYSTEM_PROMPT = """
Tu es un assistant du service client d'une boutique en ligne.
Le client connecté s'appelle {user}.

Tu disposes de deux outils :
1. get_order_status : statut réel d'une commande du client.
2. search_refund_policy : politique officielle de remboursement.

Règles :
- Pour le statut d'une commande, utilise toujours get_order_status.
- Pour toute question de remboursement, utilise search_refund_policy.
- Si la question nécessite les deux, utilise les deux, puis applique la politique aux données de la commande (calcule le montant si possible).
- N'invente jamais une information absente des résultats des outils.
- Si les informations sont insuffisantes, dis-le clairement.
- Réponds en français poliment et de façon clair et précise.
"""


def assistant(state: SAVState):
    consigne = SystemMessage(content=SYSTEM_PROMPT.format(user=state["user"])) #msg role sys
    response = llm_with_tools.invoke([consigne] + state["messages"]) #rep avec l'historique
    return {"messages": [response]}



#le graphe
builder = StateGraph(SAVState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)  # outil ? -> tools sinon END
builder.add_edge("tools", "assistant")                       # la boucle

#Le checkpointer garde tte la conv de chaque thread_id
graph = builder.compile(checkpointer=InMemorySaver())


#API

#Num de conversation par client : une nvl conv l'incremente
sessions: dict[str, int] = {}


def config(user: str) -> dict:
    return {"configurable": {"thread_id": f"{user}-{sessions.get(user, 0)}"}}


def ask(user: str, question: str) -> dict:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}], "user": user},
        config(user),
    )
    messages = result["messages"]
    
    last_question = max(i for i, m in enumerate(messages) if m.type == "human")

    

    tool_calls = []
    for m in messages[last_question + 1:]:
        if m.type == "ai":      
            tool_calls += m.tool_calls  

    return {"answer": messages[-1].content, "tool_calls": tool_calls}

def reset(user: str) -> None:
    """Nouvelle conversation doncnouveau thread_id."""
    sessions[user] = sessions.get(user, 0) + 1
