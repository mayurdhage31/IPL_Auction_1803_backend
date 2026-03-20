"""
Gemini File Search RAG Integration.

Uploads the project PDFs / markdown docs to Gemini's File API once,
then answers auction-related questions by grounding responses in those
uploaded documents. Uses the new google-genai SDK.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Documents to upload (resolved relative to the docs_dir passed in)
_DOC_NAMES = [
    "Data Science Project_ Predicting Auction Prices.pdf",
    "IPL Player Auction Pricing Model.pdf",
    "README.md",
    "seed_document.md",
]


class GeminiRAG:
    """
    Wraps Gemini's generative model + File API to provide RAG over auction docs.

    Usage:
        rag = GeminiRAG(api_key="...", docs_dir="backend/docs")
        rag.initialise()   # uploads files on first call
        result = rag.query("Why is Virat Kohli's price range so high?")
    """

    def __init__(self, api_key: str, docs_dir: str = "docs"):
        self.api_key = api_key
        self.docs_dir = Path(docs_dir)
        self._client: Optional[genai.Client] = None
        self._uploaded_files: list = []
        self._initialised = False

    # ─────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────

    def initialise(self) -> None:
        """Upload documents to Gemini File API and prepare the client."""
        if self._initialised:
            return

        self._client = genai.Client(api_key=self.api_key)

        for doc_name in _DOC_NAMES:
            doc_path = self.docs_dir / doc_name
            if not doc_path.exists():
                logger.warning(f"RAG doc not found: {doc_path}")
                continue
            try:
                logger.info(f"Uploading {doc_name} to Gemini File API…")

                # Determine MIME type
                mime = "application/pdf" if doc_name.endswith(".pdf") else "text/plain"

                uploaded = self._client.files.upload(
                    file=str(doc_path),
                    config={"mime_type": mime, "display_name": doc_name},
                )

                # Wait for file to become ACTIVE
                for _ in range(15):
                    file_info = self._client.files.get(name=uploaded.name)
                    if file_info.state.name == "ACTIVE":
                        break
                    if file_info.state.name == "FAILED":
                        logger.warning(f"  {doc_name} failed to process.")
                        break
                    time.sleep(2)
                else:
                    file_info = self._client.files.get(name=uploaded.name)

                if file_info.state.name == "ACTIVE":
                    self._uploaded_files.append(file_info)
                    logger.info(f"  Ready: {doc_name} → {file_info.uri}")
                else:
                    logger.warning(f"  Skipped: {doc_name} state={file_info.state.name}")

            except Exception as e:
                logger.error(f"  Error uploading {doc_name}: {e}")

        self._initialised = True
        logger.info(f"GeminiRAG ready with {len(self._uploaded_files)} document(s).")

    # ─────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────

    def query(self, question: str, player_context: str = "") -> dict:
        """
        Answer a question using uploaded documents as grounding.
        """
        if not self._initialised:
            self.initialise()

        if not self._client:
            return {
                "answer": "RAG not available — Gemini client not initialised.",
                "sources_used": [],
            }

        # Build prompt parts
        contents: list = []

        # Add uploaded files as file parts
        for f in self._uploaded_files:
            contents.append(
                types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type or "application/pdf")
            )

        # Text prompt
        system_text = (
            "You are an expert IPL auction analyst. "
            "Use the provided documents as your primary knowledge source. "
            "Cite relevant findings from the documents where possible. "
            "Be concise but insightful, focusing on actionable auction intelligence.\n\n"
        )
        if player_context:
            system_text += f"## Player Context (live data):\n{player_context}\n\n"
        system_text += f"## Question:\n{question}"
        contents.append(types.Part.from_text(text=system_text))

        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
            answer = response.text
            sources = [f.display_name for f in self._uploaded_files]
            return {"answer": answer, "sources_used": sources}

        except Exception as e:
            logger.error(f"Gemini query error: {e}")
            return {
                "answer": f"Unable to generate response: {str(e)}",
                "sources_used": [],
            }

    def query_player_valuation(self, player_name: str, player_context: str) -> dict:
        """Specialised query for player valuation reasoning."""
        question = (
            f"Based on the IPL auction pricing models and historical data, "
            f"explain the key factors that drive {player_name}'s price range. "
            f"Which teams are most likely to bid? What is the typical price band?"
        )
        return self.query(question, player_context=player_context)

    def is_ready(self) -> bool:
        return self._initialised and len(self._uploaded_files) > 0

    def get_doc_count(self) -> int:
        return len(self._uploaded_files)
