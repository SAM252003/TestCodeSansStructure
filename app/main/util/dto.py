from flask_restx import Namespace, fields

# Namespace pour la génération (RAG)
GeneratorDto = Namespace("generate", description="RAG generation endpoint")

GeneratorQuery = GeneratorDto.model(
    "GeneratorQuery",
    {
        "application": fields.String(required=True, example="Akuiteo"),
        "problem": fields.String(required=True, example="Créer un bloc de planning"),
        "summary": fields.String(required=False, example="Je ne sais pas comment recréer un bloc planning supprimé"),
        "confidence": fields.Boolean(required=False, example=True),
        "ask_user": fields.String(required=False, nullable=True),
        "file_id": fields.String(required=False, example="LivreBlancAkuiteo"),
        "session_id": fields.String(required=False, example="console-123"),
        "top_k": fields.Integer(required=False, example=4),
    },
)

# Alias pour usage dans controllers
GeneratorDto.query = GeneratorQuery


# Namespace pour l'ingestion de documents
EncoderDocumentDto = Namespace("ingest", description="Document ingestion endpoint")

# Parser pour multipart/form-data (fichier + metadata json)
from flask_restx import reqparse

parser = reqparse.RequestParser()
parser.add_argument("file_metadata", type=str, required=True, help="JSON string with file metadata")
parser.add_argument("file", type="FileStorage", location="files", required=True, help="File to ingest")

EncoderDocumentDto.parser = parser
EncoderDocumentDto.api = EncoderDocumentDto  # pour compatibilité avec controller
