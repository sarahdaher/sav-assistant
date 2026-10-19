# Assistant SAV : LangGraph avec RAG et outils

Un agent de service client : il cherche dans la politique de remboursement (RAG) et consulte les commandes (outil), puis répond


## Les outils


- `search_refund_policy` : **RAG** : cherche dans `policy.txt` 
- `get_order_status`  consulte la commande seulement si elle appartient au client connecté

## Auteur
Sarah Daher

## Source
- Hugging Face Agent Course 