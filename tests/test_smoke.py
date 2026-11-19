"""Tests the basic CLI readiness command."""


def test_hello_command_confirms_the_cli_is_ready(capsys):
    """The hello subcommand prints its readiness message."""
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out

