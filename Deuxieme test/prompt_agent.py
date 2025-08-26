# prompt_agent.py (exemple fusionné)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import FewShotChatMessagePromptTemplate

SYSTEM = """
Tu es un assistant expert en support IT.

Tu as accès au **TOOL `retrieve_it`** :
- Input  : la question utilisateur
- Output : une liste de problèmes/solutions au format JSON.

Appelle ce tool dès que c'est pertinent.
Ne révèle jamais ton raisonnement interne.
Salutations simplifiées : bonjour / au revoir.
Réponse finale : concise, professionnelle, en français.

Si l’utilisateur·rice dit qu’il n’a pas compris ou qu’il ou elle
souhaite plus d’explications (mots-clés : « pas compris »,
« explique », « clarifie », « comment ? »), SUIS CE FLOW :

① Demande *quelle partie exacte* n’est pas claire (ou propose
   plusieurs points possibles s’il y en a).

② Après la clarification, reformule seulement la partie floue,
   PAS toute la réponse initiale.

③ Vérifie en une phrase que la nouvelle explication est claire
   (« Est-ce plus clair ? »). Ne continue qu’après confirmation.

"""

example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
examples = [
    {
        "input": "Outlook ne reçoit plus mes mails. Comment corriger ça ?",
        "output": (
            "## Réception de mails impossible dans Outlook\n\n"
            "*Examiner les règles de boîte de réception.\n"
            "*Vérifier les quotas de boîte pleine.\n"
            "*Contrôler pare-feu, antivirus, et configuration Exchange\n"
            "*Redémarrer l’ordinateur sans application tierce.\n"
            "*Analyser les logs et les codes d erreur.\n"
            "*Comparer avec un autre compte utilisateur."
            "*Escalader vers le support si besoin, avec logs.\n"
        )
    },
    {
        "input": "Mon Outlook ne peut pas envoyer de mails, que dois-je faire ?",
        "output": (
            "## Diagnostic – Envoi d'e-mails impossible dans Outlook\n\n"
            "*Vérifier la configuration du compte (adresse, mot de passe)."
            "*Contrôler les paramètres SMTP et la connectivité réseau.\n\n"
            "*Redémarrer l'ordinateur et tester sans logiciel tiers actif.\n"
            "* Analyser les journaux d erreur (logs Outlook, Event Viewer)\".\n"
            "*Consulter la documentation Microsoft pour les cas similaires.\n"
            "*Tester avec un autre compte ou poste..\n"
        )
    }
]
few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
prompt_agent = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
MessagesPlaceholder("agent_scratchpad"),
])

