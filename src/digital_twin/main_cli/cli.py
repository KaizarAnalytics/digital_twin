import sys
from pathlib import Path

import typer
import streamlit.web.cli as stcli

app = typer.Typer(help="CLI for digital_twin")


@app.command(help="Start the Streamlit UI for digital_twin.")
def ui(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server address"),
    port: int = typer.Option(8501, "--port", "-p", help="Server port"),
):
    """
    Start the Streamlit-app (streamlit_app.py).
    """
    # main_cli.py is in digital_twin/cli/, streamlit_app.py in digital_twin/
    project_root = Path(__file__).resolve().parent.parent
    app_path = project_root / "streamlit_app.py"

    if not app_path.exists():
        typer.echo(f"[ERROR] streamlit_app.py not found at: {app_path}")
        raise typer.Exit(code=1)

    # Streamlit expects argv as if you called 'streamlit run ...'
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        f"--server.address={host}",
        f"--server.port={port}",
    ]

    raise SystemExit(stcli.main())


def main():
    """
    Entry point if you run this script directly.
    """
    app()


if __name__ == "__main__":
    main()
