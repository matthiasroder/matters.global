import pytest


@pytest.fixture(autouse=True)
def isolated_llm_config(tmp_path, monkeypatch):
    """Default to no provider, regardless of the developer's configuration."""
    monkeypatch.setattr(
        "matters.llm.config.user_config_dir", lambda _name: str(tmp_path / "llm-config")
    )
    for name in (
        "MATTERS_CONFIG",
        "MATTERS_EXTRACT_MODEL",
        "MATTERS_TOTS_MODEL",
        "MATTERS_EMBED_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
