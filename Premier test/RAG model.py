from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from conversation_summary_memory import ConversationSummaryBufferMessageHistory



""""
# must enter API key
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Projet"
"""
model_name = "llama3.2:1b-instruct-fp16"

# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)


# Defining the system prompt (how the AI should act)
prompt = """Tu es un assistant expert en support IT pour les entreprises.

Tu reçois :
- Une **question utilisateur** : {query}
- Un **contexte extrait automatiquement** de la base documentaire de l’entreprise (historique de tickets, guides internes, logs, etc.) voici le context {CONTEXT}

Ta mission est de résoudre la question **de façon précise et fiable**.

Voici comment procéder :
1. Raisonne étape par étape dans ta tête pour comprendre le problème et planifier une solution (ne montre pas ce raisonnement à l’utilisateur).
2. Utilise uniquement les informations du contexte si possible. Ne devine pas si tu n’es pas sûr.
3. À la fin, donne une **réponse concise et professionnelle**, directement exploitable par l’utilisateur final.

⚠️ **Important : tu ne dois JAMAIS afficher ton raisonnement. Seule la réponse finale doit être visible ET aucune phrase d'introduction**


    N'oublie pas de t'aider du context : {CONTEXT}
    """

context = """Problème : L'utilisateur de Outlook ne peut pas envoyer des e-mails.
Solution : Commencez par vérifier la configuration du compte, les paramètres smtp et la connectivité réseau. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, ne, support
---
Problème : L'utilisateur de Outlook ne peut pas recevoir des e-mails.
Solution : Commencez par vérifier les règles de boîte de réception, les quotas, les paramètres de pare-feu et le serveur exchange. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, ne, support
---
Problème : L'utilisateur de Outlook rencontre une erreur de connexion.
Solution : Commencez par vérifier les identifiants, réinitialiser le mot de passe et la synchronisation azure ad. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, rencontre, support
---
Problème : L'utilisateur de Outlook est lent ou se bloque.
Solution : Commencez par mettre à jour l'application, nettoyer le cache et vérifier l'utilisation des ressources système. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, est, support
---
Problème : L'utilisateur de Outlook affiche un message d'erreur inconnu.
Solution : Commencez par consulter les logs, rechercher le code d'erreur dans la documentation microsoft. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, affiche, support
---
Problème : L'utilisateur de Outlook ne se synchronise pas.
Solution : Commencez par vérifier la connexion réseau, ré-authentifier et vérifier les paramètres de synchronisation. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, ne, support
---
Problème : L'utilisateur de Outlook a des problèmes d’authentification à deux facteurs.
Solution : Commencez par vérifier l'application authenticator, configurer correctement mfa dans azure ad. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, a, support
---
Problème : L'utilisateur de Outlook affiche un écran blanc.
Solution : Commencez par effacer le cache, mettre à jour l'application ou le navigateur et vérifier les extensions ou plugins. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, affiche, support
---
Problème : L'utilisateur de Outlook ne démarre pas.
Solution : Commencez par réinstaller l’application, vérifier les permissions et examiner les logs de démarrage. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, ne, support
---
Problème : L'utilisateur de Outlook présente un problème de permission.
Solution : Commencez par vérifier les rôles azure ad, groupes de sécurité et droits sur sharepoint/onedrive. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, présente, support
---
Problème : L'utilisateur de Outlook ne parvient pas à accéder au fichier.
Solution : Commencez par vérifier le chemin, les permissions et l'état du fichier dans onedrive/sharepoint. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, ne, support
---
Problème : L'utilisateur de Outlook ne peut pas rejoindre la réunion.
Solution : Commencez par vérifier le lien de réunion, la connectivité réseau et les versions de teams. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, ne, support
---
Problème : L'utilisateur de Outlook renvoie une erreur 500.
Solution : Commencez par vérifier l'état des services backend, consulter les journaux et l'état du service microsoft. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, renvoie, support
---
Problème : L'utilisateur de Outlook renvoie une erreur 403.
Solution : Commencez par vérifier les droits d’accès, les politiques conditional access et les autorisations. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, renvoie, support
---
Problème : L'utilisateur de Outlook affiche un message 'espace disque insuffisant'.
Solution : Commencez par libérer de l’espace, vérifier les quotas onedrive et déplacer les fichiers volumineux. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, affiche, support
---
Problème : L'utilisateur de Outlook affiche une boîte de dialogue de mise à jour bloquée.
Solution : Commencez par redémarrer, vérifier les mises à jour windows et office et examiner les logs de mise à jour. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, affiche, support
---
Problème : L'utilisateur de Outlook ne détecte pas l’imprimante.
Solution : Commencez par vérifier le pilote, la connexion réseau et la configuration de l’imprimante. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, ne, support
---
Problème : L'utilisateur de Outlook perd des données lors de la synchronisation.
Solution : Commencez par vérifier les conflits, restaurer via la version précédente et vérifier l'intégrité du réseau. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, perd, support
---
Problème : L'utilisateur de Outlook a des problèmes de mise à jour automatique.
Solution : Commencez par vérifier les services de mise à jour, consulter les logs et redémarrer les services concernés. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, a, support
---
Problème : L'utilisateur de Outlook ne peut pas se connecter au VPN.
Solution : Commencez par vérifier la configuration vpn, le pare-feu et les identifiants. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, ne, support
---
Problème : L'utilisateur de Outlook pose des problèmes de licence.
Solution : Commencez par vérifier l’abonnement, l’affectation de licence dans microsoft 365 admin center et l’état du compte. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : outlook, pose, support
---
Problème : L'utilisateur de Teams ne peut pas envoyer des e-mails.
Solution : Commencez par vérifier la configuration du compte, les paramètres smtp et la connectivité réseau. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : teams, ne, support
---
Problème : L'utilisateur de Teams ne peut pas recevoir des e-mails.
Solution : Commencez par vérifier les règles de boîte de réception, les quotas, les paramètres de pare-feu et le serveur exchange. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : teams, ne, support
---
Problème : L'utilisateur de Teams rencontre une erreur de connexion.
Solution : Commencez par vérifier les identifiants, réinitialiser le mot de passe et la synchronisation azure ad. Ensuite, essayez de reproduire le problème en conditions minimales (ordinateur redémarré, sans applications tierces actives). Vérifiez les journaux d'erreur ou les fichiers de log associés pour identifier les messages spécifiques. Consultez la documentation officielle Microsoft ou les forums dédiés pour des cas similaires. Si le problème persiste, testez avec un autre compte utilisateur ou sur un autre poste pour isoler le problème. En dernier recours, contactez l'assistance technique interne ou Microsoft en fournissant les logs et détails collectés.
Mots clés : teams, rencontre, support
---
Problème : L'utilisateur de Teams est lent ou se bloque.
Solution : Commencez par mettre à jour l'application, nettoyer le cache et vérifier l'utilisation des ressources sy---"""

query ="Je n'arrive pas à ouvrir Word dans microsoft 365"


example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

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

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    MessagesPlaceholder(variable_name="history"),
    few_shot_prompt,
    ("user", "{query}")
])

ConversationSummaryBufferMessageHistory(llm=llm, k=4)

pipeline = (prompt_template
            | llm
            )

chat_map = {}


def get_chat_history(session_id: str, llm: ChatOllama, k: int) -> ConversationSummaryBufferMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = ConversationSummaryBufferMessageHistory(llm=llm, k=k)
    # return the chat history
    return chat_map[session_id]

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="llm",
            annotation=ChatOllama,
            name="Ollama",
            description="The LLM to use for the conversation summary",
            default=llm,
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        )
    ]
)

for i, msg in enumerate([
    "I'm researching the different types of conversational memory.",
    "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory.",
    "Buffer memory just stores the entire conversation",
    "Buffer window memory stores the last k messages, dropping the rest."
]):
    print(f"---\nMessage {i+1}\n---\n")
    pipeline_with_history.invoke(
        {"query": msg , "CONTEXT" : ''},
        config={"session_id": "id_123", "llm": llm, "k": 4}
    )