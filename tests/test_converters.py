import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import converters


def test_pdf_conversion_falls_back_when_pdfminer_rejects_font_structure(tmp_path, monkeypatch):
    pdf_path = tmp_path / "malformed-font.pdf"
    pdf_path.write_bytes(b"%PDF malformed font fixture")

    primary = SimpleNamespace(convert=lambda _: (_ for _ in ()).throw(KeyError("DescendantFonts")))
    monkeypatch.setattr(converters, "_get_markitdown", lambda: primary)

    class FakeTextPage:
        def get_text_range(self):
            return "Recovered agreement text"

    class FakePage:
        def get_textpage(self):
            return FakeTextPage()

    fake_pdfium = SimpleNamespace(PdfDocument=lambda _: [FakePage()])
    monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)

    text = converters.convert_file(pdf_path, raise_errors=True)

    assert text == "Recovered agreement text"


def test_pdf_conversion_raises_when_primary_and_fallback_both_fail(tmp_path, monkeypatch):
    pdf_path = tmp_path / "unreadable.pdf"
    pdf_path.write_bytes(b"not a readable pdf")

    primary = SimpleNamespace(convert=lambda _: (_ for _ in ()).throw(ValueError("primary")))
    monkeypatch.setattr(converters, "_get_markitdown", lambda: primary)
    fake_pdfium = SimpleNamespace(
        PdfDocument=lambda _: (_ for _ in ()).throw(ValueError("fallback"))
    )
    monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)

    with pytest.raises(ValueError, match="fallback"):
        converters.convert_file(pdf_path, raise_errors=True)
