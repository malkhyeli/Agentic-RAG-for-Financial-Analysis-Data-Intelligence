from src.utils.pdf_store import persist_pdf_bytes


def test_persist_pdf_bytes_returns_absolute_path(tmp_path):
    cache_dir = tmp_path / "cache"
    stored = persist_pdf_bytes(b"%PDF-1.4", "sample.pdf", cache_dir=cache_dir)
    assert stored.path.is_absolute()
    assert stored.path.exists()
    assert stored.filename == "sample.pdf"
