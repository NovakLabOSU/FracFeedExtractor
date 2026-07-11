import logging
from logging.handlers import RotatingFileHandler

import pytest


@pytest.fixture(autouse=True)
def no_file_logging(monkeypatch):
    """Prevent log.warning/log.error calls from reaching logs/fracfeed.log during tests.

    setup_logging() attaches a RotatingFileHandler to the root logger at WARNING
    level.  When tests exercise entry-point functions (e.g., main()), that handler
    is created and persists for the rest of the pytest session, so expected
    test-error-path messages appear as real operational errors in the log file.

    Patching setup_logging in each source module blocks file-handler creation;
    raising the root level to CRITICAL blocks any residual log calls from reaching
    handlers that may already exist.
    """
    for target in (
        "src.io.pdf_text_extraction.setup_logging",
        "src.pipeline.classify_extract.setup_logging",
        "src.classifier.train_model.setup_logging",
        "src.classifier.pdf_classifier.setup_logging",
        "src.extraction.llm_client.setup_logging",
    ):
        monkeypatch.setattr(target, lambda *a, **kw: None)

    root = logging.getLogger()
    original_level = root.level
    root.setLevel(logging.CRITICAL)
    yield
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler):
            h.close()
            root.removeHandler(h)
    root.setLevel(original_level)
