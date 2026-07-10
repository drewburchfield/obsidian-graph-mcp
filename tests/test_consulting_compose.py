from pathlib import Path

import yaml


def test_consulting_stack_is_isolated_and_continuously_watches_corpus():
    compose_path = Path(__file__).parent.parent / "docker-compose.consulting.yml"
    config = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    assert config["name"] == "consulting-graph"
    assert set(config["services"]) == {"consulting-graph", "consulting-pgvector"}

    app = config["services"]["consulting-graph"]
    environment = dict(item.split("=", 1) for item in app["environment"])
    assert environment["CORPUS_PATH"] == "/corpus"
    assert environment["OBSIDIAN_WATCH_ENABLED"] == "true"
    assert environment["OBSIDIAN_WATCH_USE_POLLING"] == "true"
    assert environment["EMBEDDING_PROVIDER"] == "openrouter"
    assert environment["EMBEDDING_DIMENSIONS"] == "4096"
    assert ".pdf" in environment["CORPUS_EXTENSIONS"]
    assert ".xlsx" in environment["CORPUS_EXTENSIONS"]
    assert any(str(volume).endswith(":/corpus:ro") for volume in app["volumes"])
    assert all("/vault" not in str(volume) for volume in app["volumes"])


def test_consulting_mcp_client_disables_duplicate_watcher():
    launcher = (Path(__file__).parent.parent / "scripts" / "run_consulting_mcp.sh").read_text(
        encoding="utf-8"
    )

    assert "docker exec -i" in launcher
    assert "OBSIDIAN_WATCH_ENABLED=false" in launcher
    assert "consulting-graph" in launcher
