"""Lesson 2: prepare and plot Curiosity's recorded drive positions."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "localized_pos.csv"
OUTPUT_DIR = ROOT / "artifacts"


def main() -> None:
    rover = pd.read_csv(DATA_PATH)

    # Keep rover-position rows and remove the one drive=0 convenience row
    # associated with each SITE reference. Remaining rows describe drives.
    drives = rover.loc[(rover["frame"] == "ROVER") & (rover["drive"] != 0)].copy()
    drives = drives.sort_values(["sol", "site", "drive"], kind="stable").reset_index(drop=True)

    # landing_x/y are coordinates in meters relative to the landing site.
    # Straight-line distances between recorded endpoints approximate the path;
    # they do not capture every turn made during a drive.
    drives["step_m"] = (
        (drives["landing_x"].diff() ** 2 + drives["landing_y"].diff() ** 2) ** 0.5
    ).fillna(0)
    drives["cumulative_distance_m"] = drives["step_m"].cumsum()

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "curiosity_drive_positions.csv"
    plot_path = OUTPUT_DIR / "curiosity_traverse.svg"
    drives.to_csv(csv_path, index=False)
    save_traverse_plot(drives, plot_path)

    print(f"Drive records analyzed: {len(drives):,}")
    print(f"Martian sols covered: {drives['sol'].min()}–{drives['sol'].max()}")
    print(f"Estimated distance between recorded endpoints: {drives['step_m'].sum():,.1f} m")
    print(f"Largest single recorded-position change: {drives['step_m'].max():,.1f} m")
    print(f"Saved cleaned data: {csv_path.relative_to(ROOT)}")
    print(f"Saved traverse plot: {plot_path.relative_to(ROOT)}")


def save_traverse_plot(drives: pd.DataFrame, path: Path) -> None:
    # Build an SVG with the standard library so the project needs no plotting
    # package. SVG stays sharp when resized and opens in any modern browser.
    width, height = 900, 650
    margin = 75
    east = drives["landing_y"].astype(float).tolist()
    north = drives["landing_x"].astype(float).tolist()
    east_min, east_max = min(east), max(east)
    north_min, north_max = min(north), max(north)
    east_span = max(east_max - east_min, 1)
    north_span = max(north_max - north_min, 1)
    scale = min((width - 2 * margin) / east_span, (height - 2 * margin) / north_span)
    plot_width, plot_height = east_span * scale, north_span * scale
    left, top = (width - plot_width) / 2, (height - plot_height) / 2

    points = [
        (left + (x - east_min) * scale, height - top - (y - north_min) * scale)
        for x, y in zip(east, north)
    ]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.8" fill="#b65f36" opacity="0.58"/>'
        for x, y in points
    )
    start_x, start_y = points[0]
    end_x, end_y = points[-1]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fcfaf6"/>
<text x="{width / 2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="21" fill="#302820">Curiosity traverse from NASA PDS localized positions</text>
<line x1="{left:.2f}" y1="{height - top:.2f}" x2="{width - left:.2f}" y2="{height - top:.2f}" stroke="#635d55"/>
<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{height - top:.2f}" stroke="#635d55"/>
<polyline points="{polyline}" fill="none" stroke="#6d3422" stroke-width="1.4" opacity="0.62"/>
{circles}
<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="7" fill="#25834b"/>
<rect x="{end_x - 5:.2f}" y="{end_y - 5:.2f}" width="10" height="10" fill="#252525"/>
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#302820">East from landing site (m)</text>
<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="sans-serif" font-size="14" fill="#302820">North from landing site (m)</text>
<circle cx="{width - 205}" cy="55" r="6" fill="#25834b"/><text x="{width - 190}" y="60" font-family="sans-serif" font-size="13">first drive</text>
<rect x="{width - 105}" y="50" width="10" height="10" fill="#252525"/><text x="{width - 89}" y="60" font-family="sans-serif" font-size="13">last drive</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
