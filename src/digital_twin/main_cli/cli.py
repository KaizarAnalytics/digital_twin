"""
Command-line interface for the digital twin simulation engine.
"""
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Digital Twin - Capacity Planning Simulation Engine")
console = Console()


@app.command()
def simulate(
    config: str = typer.Argument(
        "src/digital_twin/hospital/configs/hospital_1.yml",
        help="Path to hospital configuration YAML",
    ),
    days: int = typer.Option(180, "--days", "-d", help="Simulation horizon in days"),
    runs: int = typer.Option(3000, "--runs", "-n", help="Number of Monte Carlo runs"),
    beds: Optional[int] = typer.Option(None, "--beds", "-b", help="Override bed capacity"),
    service: str = typer.Option("surgery", "--service", "-s", help="Service to simulate"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
):
    """
    Run a Monte Carlo capacity simulation.

    Loads patient data and configuration, then runs stochastic simulations
    to estimate overflow probability and capacity requirements.
    """
    import numpy as np
    from digital_twin.hospital.config_loader import load_hospital_config
    from digital_twin.hospital.data_prep import load_patients, arrivals_per_day, los_values
    from digital_twin.core.mc_simulator import simulate_occupancy, make_arrival_sampler
    from digital_twin.output.metrics import risk_summary

    console.print(f"[bold blue]Loading configuration from {config}...[/bold blue]")

    try:
        cfg = load_hospital_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        raise typer.Exit(1)

    raw_data_dir = cfg.simulation.get("data_dir", "digital_twin/hospital/data")

    # Try multiple path resolutions
    candidates = [
        Path(raw_data_dir),
        Path("src") / raw_data_dir,
        Path(__file__).parent.parent / "hospital" / "data",
    ]
    data_dir = None
    for candidate in candidates:
        if (candidate / "raw" / "patients.csv").exists():
            data_dir = str(candidate)
            break

    if data_dir is None:
        console.print(f"[red]Could not find data directory.[/red]")
        raise typer.Exit(1)

    capacity = beds if beds else cfg.simulation.get("capacity_default", 36)

    console.print(f"[dim]Data directory: {data_dir}[/dim]")
    console.print(f"[dim]Service: {service}[/dim]")
    console.print(f"[dim]Capacity: {capacity} beds[/dim]")
    console.print(f"[dim]Horizon: {days} days, {runs} runs[/dim]")
    console.print()

    # Load data
    try:
        patients = load_patients(data_dir)
        arr = arrivals_per_day(patients, service)["arrivals"]
        los = los_values(patients, service)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1)

    # Build samplers
    rng = np.random.default_rng(seed)
    arrival_sampler = make_arrival_sampler(arr)
    los_sampler = lambda n: rng.choice(los, size=n, replace=True)

    # Run simulation
    with console.status("[bold green]Running Monte Carlo simulation..."):
        max_occ, overflow_days = simulate_occupancy(
            days=days,
            n_runs=runs,
            beds=capacity,
            arrival_sampler=arrival_sampler,
            processtime_sampler=los_sampler,
        )

    # Results
    summary = risk_summary(max_occ, overflow_days, capacity)

    console.print("[bold green]Simulation complete![/bold green]")
    console.print()

    table = Table(title="Capacity Risk Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Beds", str(capacity))
    table.add_row("P(overflow)", f"{summary['P(max>100%)']:.2%}")
    table.add_row("P(>95% capacity)", f"{summary['P(max>95%)']:.2%}")
    table.add_row("Mean overflow days", f"{summary['mean_overflow_days']:.2f}")
    table.add_row("Median max occupancy", f"{summary['median_max_occ']:.0f}")
    table.add_row("95th percentile max", f"{summary['p95_max_occ']:.0f}")

    console.print(table)


@app.command()
def info():
    """Show information about the digital twin package."""
    console.print("[bold]Digital Twin - Capacity Planning Simulation Engine[/bold]")
    console.print()
    console.print("Monte Carlo simulation for capacity planning and risk assessment.")
    console.print()
    console.print("Available commands:")
    console.print("  [cyan]simulate[/cyan]  - Run Monte Carlo capacity simulation")
    console.print("  [cyan]info[/cyan]      - Show this information")
    console.print()
    console.print("Example:")
    console.print("  digital-twin simulate --beds 20 --runs 5000")
    console.print()
    console.print("For more details: https://github.com/KaizarAnalytics/digital_twin")


def main():
    app()


if __name__ == "__main__":
    main()
