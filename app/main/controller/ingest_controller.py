from flask import request
from flask_restx import Resource
from loguru import logger
import json

from app.main.util.dto import EncoderDocumentDto
from app.main.service.ingest_service import IngestService

api = EncoderDocumentDto.api
parser = EncoderDocumentDto.parser
ingest_service = IngestService()


@api.route("")
class IngestDocument(Resource):
    @api.expect(parser, validate=True)
    @api.doc("Ingest a document and index it into the vector store")
    @api.response(200, "Document ingested successfully")
    @api.response(400, "Invalid request")
    @api.response(500, "Internal error during ingestion")
    def post(self):
        try:
            # Récupère et parse le metadata JSON
            file_metadata_raw = request.form.get("file_metadata")
            if not file_metadata_raw:
                return {"error": "file_metadata is required"}, 400
            try:
                payload = json.loads(file_metadata_raw)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in file_metadata: {e}")
                return {"error": "file_metadata must be valid JSON"}, 400

            # Fichier uploadé
            if "file" not in request.files:
                return {"error": "file is required"}, 400
            file = request.files["file"]

            result = ingest_service.ingest_file(payload, file)
            return result
        except Exception as e:
            logger.error(f"Error in ingest_controller: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}, 500
