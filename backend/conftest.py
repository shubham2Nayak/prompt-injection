"""
Pytest configuration for the PromptTrace backend test suite.

The application imports itself with ``backend/`` as the import root
(``from config import config``, ``from state import WorkflowState``), which is
how app.py, graph.py and every agent module resolve each other.

Some test modules instead used ``from backend.rag.parser import PDFParser``.
That spelling requires the repository root on ``sys.path`` instead, and because
``backend/`` has no ``__init__.py`` it resolves through a namespace package -
which loads the same file a second time as a distinct module object. Two copies
of the module means two copies of every class, so ``isinstance`` checks across
the boundary fail and patching one copy does not affect the other.

Puting ``backend/`` on the path and normalising the test imports to match the
application removes both failure modes.
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# The application defaults FAISS_REQUIRE_GPU to true so a host that believes it
# is doing GPU retrieval cannot quietly fall back to CPU. That is the right
# production behaviour, but a test host may have only the faiss-cpu wheel
# installed, and tests that merely construct the vector store would then abort
# on a GPU-capability question that is not what they are exercising. Opt the
# suite into the documented CPU fallback; a deployment keeps the strict default.
try:
    from config import config

    config.FAISS_REQUIRE_GPU = False
except Exception:  # pragma: no cover - config import problems surface per-test
    pass


# ---------------------------------------------------------------------------
# PDF fixture
#
# The ingestion tests referenced "sample1.pdf", which was never committed to the
# repository, so every one of them failed with FileNotFoundError before reaching
# a single assertion. The fixture is generated instead of committed: it keeps the
# repository free of binary blobs and guarantees the content the tests assert on
# is present, which a checked-in PDF could drift away from.
# ---------------------------------------------------------------------------

SAMPLE_PDF_NAME = "sample1.pdf"

_SAMPLE_PAGES = [
    (
        "Prompt Injection Research Notes",
        "Prompt injection is a technique in which malicious instructions are "
        "embedded inside otherwise legitimate document content in order to "
        "influence the behaviour of an AI assistant. This document surveys "
        "example-based analysis and experimental evaluation of injection "
        "payloads, and reviews a comprehensive overview of published defences "
        "and detection methods for the same threat class.",
    ),
    (
        "Detection Methodology",
        "Detection approaches generally fall into two families. The first looks "
        "for lexical markers such as imperative instructions embedded in prose. "
        "The second applies a content-safety classifier to each retrieved segment "
        "and records a confidence score, which is then thresholded to decide "
        "whether the segment has been successfully sanitised before reuse.",
    ),
    (
        "Evaluation Results",
        "Across the evaluated corpus the semantic classifier achieved materially "
        "higher recall than the lexical baseline, at the cost of additional "
        "latency per retrieved chunk. Batching the classifier calls recovered "
        "most of that overhead. Persistence across downstream agents remained the "
        "dominant failure mode in every configuration we measured.",
    ),
]


def _build_sample_pdf(path: str) -> str:
    """Write a small multi-page PDF with extractable text. Returns the path."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    for title, body in _SAMPLE_PAGES:
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(72, height - 72, title)
        pdf.setFont("Helvetica", 10)

        # Simple greedy wrap; the exact line breaks are irrelevant to the tests,
        # only that the text survives extraction.
        y = height - 96
        for paragraph in (title + ". " + body).split(". "):
            line = ""
            for word in paragraph.split():
                if len(line) + len(word) + 1 > 90:
                    pdf.drawString(72, y, line)
                    y -= 14
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                pdf.drawString(72, y, line)
                y -= 14
            y -= 6

        pdf.showPage()

    pdf.save()
    return path


def get_sample_pdf_path() -> str:
    """
    Return the generated sample PDF path, building it on first use.

    Lazy rather than a pytest fixture so the test module can call it from setUp
    and from module scope without depending on fixture injection. The path is
    cached in a module global, so the PDF is written once per session.
    """
    global _SAMPLE_PDF_PATH

    if _SAMPLE_PDF_PATH is None:
        import tempfile

        fixture_dir = tempfile.mkdtemp(prefix="prompttrace_fixtures_")
        _SAMPLE_PDF_PATH = _build_sample_pdf(
            os.path.join(fixture_dir, SAMPLE_PDF_NAME)
        )
    return _SAMPLE_PDF_PATH


_SAMPLE_PDF_PATH = None