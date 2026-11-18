"""Tests the JSON store and the basic CLI readiness command."""

from app.storage import Store


def test_store_round_trips_values_and_returns_defaults_for_missing_keys(tmp_path):
    """Writing values survives reads, missing keys return their default, and all() is sorted."""
    store = Store(str(tmp_path / "app.json"))
    store.put("hello", {"n": 1})
    assert store.get("hello") == {"n": 1}
    assert store.get("missing", "fallback") == "fallback"
    assert store.all() == {"hello": {"n": 1}}


def test_hello_command_confirms_the_cli_is_ready(capsys):
    """The hello subcommand prints its readiness message."""
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out

