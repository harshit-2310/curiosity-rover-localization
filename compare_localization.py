"""Lesson 3: compare rover-reported and ground-localized positions."""

from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
LOCALIZED_PATH = DATA_DIR / "localized_pos.csv"
TELEMETRY_PATH = DATA_DIR / "telemetry.csv"
OUTPUT_DIR = ROOT / "artifacts"


def main() -> None:
    if not TELEMETRY_PATH.exists():
        raise SystemExit(
            "Missing data/telemetry.csv. Download NASA's telemetry view with:\n"
            "curl -fL -o data/telemetry.csv "
            "https://planetarydata.jpl.nasa.gov/img/data/msl/msl_places/"
            "data_localizations/telemetry.csv"
        )

    localized = pd.read_csv(LOCALIZED_PATH)
    localized = localized.loc[
        (localized["frame"] == "ROVER") & (localized["drive"] != 0)
    ].copy()

    telemetry = pd.read_csv(TELEMETRY_PATH)
    telemetry = telemetry.loc[
        (telemetry["frame"] == "ROVER") & (telemetry["drive"] != 0)
    ].copy()

    # NASA recommends the highest pose number for each site/drive because it
    # includes the rover's later pose updates, such as visual odometry.
    telemetry = (
        telemetry.sort_values(["pose"], kind="stable")
        .drop_duplicates(["site", "drive"], keep="last")
    )

    comparison = localized.merge(
        telemetry,
        on=["site", "drive"],
        how="left",
        suffixes=("_localized", "_telemetry"),
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int((comparison["_merge"] != "both").sum())
    if unmatched:
        print(f"Warning: {unmatched} localized checkpoints had no telemetry match.")
    comparison = comparison.loc[comparison["_merge"] == "both"].drop(columns="_merge")
    comparison["position_difference_m"] = comparison.apply(
        lambda row: math.hypot(
            row["landing_x_localized"] - row["landing_x_telemetry"],
            row["landing_y_localized"] - row["landing_y_telemetry"],
        ),
        axis=1,
    )
    comparison = comparison.sort_values(
        ["sol_localized", "site", "drive"], kind="stable"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "localization_drift.csv"
    svg_path = OUTPUT_DIR / "localization_drift.svg"
    comparison.to_csv(csv_path, index=False)
    save_drift_plot(comparison, svg_path)

    drift = comparison["position_difference_m"]
    print(f"Matched localized checkpoints: {len(comparison):,} of {len(localized):,}")
    print(f"Mean rover-to-localized position difference: {drift.mean():.2f} m")
    print(f"Median position difference: {drift.median():.2f} m")
    print(f"90th percentile: {drift.quantile(0.90):.2f} m")
    print(f"Largest position difference: {drift.max():.2f} m")
    print(f"Saved comparison table: {csv_path.relative_to(ROOT)}")
    print(f"Saved drift plot: {svg_path.relative_to(ROOT)}")


def save_drift_plot(comparison: pd.DataFrame, path: Path) -> None:
    width, height = 900, 520
    margin = 75
    sols = comparison["sol_localized"].astype(float).tolist()
    differences = comparison["position_difference_m"].astype(float).tolist()
    x_min, x_max = min(sols), max(sols)
    y_min, y_max = 0.0, max(differences) * 1.08
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    points = [
        (
            margin + (x - x_min) / max(x_max - x_min, 1) * plot_width,
            height - margin - y / max(y_max, 1) * plot_height,
        )
        for x, y in zip(sols, differences)
    ]
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="#b65f36" opacity="0.58"/>'
        for x, y in points
    )
    median_y = height - margin - comparison["position_difference_m"].median() / max(y_max, 1) * plot_height
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fcfaf6"/>
<text x="{width / 2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="21" fill="#302820">Curiosity: rover position vs. localized position</text>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#635d55"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#635d55"/>
<line x1="{margin}" y1="{median_y:.2f}" x2="{width - margin}" y2="{median_y:.2f}" stroke="#347a9a" stroke-dasharray="6 5"/>
{circles}
<text x="{margin}" y="{height - 30}" font-family="sans-serif" font-size="13">Sol {x_min:.0f}</text>
<text x="{width - margin}" y="{height - 30}" text-anchor="end" font-family="sans-serif" font-size="13">Sol {x_max:.0f}</text>
<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-family="sans-serif" font-size="14">Martian sol</text>
<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="sans-serif" font-size="14">Position difference (m)</text>
<text x="{width - margin}" y="{median_y - 7:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#347a9a">median</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
