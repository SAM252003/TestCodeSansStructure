from flask import request
from flask_restx import Resource
from loguru import logger

from app.main.util.dto import GeneratorDto
from app.main.service.generator_service import GeneratorService
from app.main.util.dto import GeneratorDto
api = GeneratorDto

query_model = GeneratorDto.query
generator_service = GeneratorService()

@api.route("")
class SynthesisDocument(Resource):
    @api.expect(query_model, validate=True)
    def post(self):
        payload = request.json or {}
        session_id = payload.get("session_id")
        top_k = payload.get("top_k")
        try:
            answer = generator_service.answer_question(payload, session_id=session_id, top_k=top_k)
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Error in generator_controller: {e}")
            return {"error": str(e)}, 500
