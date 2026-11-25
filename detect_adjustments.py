"""Lesson 4: flag unusual drive-to-drive localization adjustments."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).parent
COMPARISON_PATH = ROOT / "artifacts" / "localization_drift.csv"
OUTPUT_DIR = ROOT / "artifacts"
FEATURES = ["correction_change_m", "telemetry_step_m", "localized_step_m"]


def main() -> None:
    if not COMPARISON_PATH.exists():
        raise SystemExit("Run compare_localization.py first to create the comparison table.")

    drives = pd.read_csv(COMPARISON_PATH).sort_values(
        ["sol_localized", "site", "drive"], kind="stable"
    ).copy()

    # A correction vector is the difference between the localized and
    # rover-reported positions. Its change from one checkpoint to the next
    # tells us how much the correction shifted over that interval.
    drives["correction_x_m"] = drives["landing_x_localized"] - drives["landing_x_telemetry"]
    drives["correction_y_m"] = drives["landing_y_localized"] - drives["landing_y_telemetry"]
    drives["correction_change_m"] = (
        (drives["correction_x_m"].diff() ** 2 + drives["correction_y_m"].diff() ** 2) ** 0.5
    )
    drives["telemetry_step_m"] = (
        (drives["landing_x_telemetry"].diff() ** 2 + drives["landing_y_telemetry"].diff() ** 2) ** 0.5
    )
    drives["localized_step_m"] = (
        (drives["landing_x_localized"].diff() ** 2 + drives["landing_y_localized"].diff() ** 2) ** 0.5
    )

    model_data = drives.dropna(subset=FEATURES).copy()
    model = IsolationForest(
        n_estimators=300,
        contamination=0.03,
        random_state=42,
    )
    model_data["is_unusual"] = model.fit_predict(model_data[FEATURES]) == -1
    model_data["anomaly_score"] = -model.decision_function(model_data[FEATURES])

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "unusual_adjustments.csv"
    svg_path = OUTPUT_DIR / "unusual_adjustments.svg"
    model_data.sort_values("anomaly_score", ascending=False).to_csv(csv_path, index=False)
    save_anomaly_plot(model_data, svg_path)

    flagged = model_data[model_data["is_unusual"]]
    print(f"Localization checkpoint intervals analyzed: {len(model_data):,}")
    print(f"Intervals flagged for review: {len(flagged):,} ({len(flagged) / len(model_data):.1%})")
    print("Largest correction changes among flagged intervals:")
    print(
        flagged.nlargest(8, "correction_change_m")[
            ["site", "drive", "sol_localized", "correction_change_m", "telemetry_step_m", "localized_step_m"]
        ].to_string(index=False)
    )
    print(f"Saved scored intervals: {csv_path.relative_to(ROOT)}")
    print(f"Saved anomaly plot: {svg_path.relative_to(ROOT)}")


def save_anomaly_plot(data: pd.DataFrame, path: Path) -> None:
    width, height = 900, 520
    margin = 75
    sols = data["sol_localized"].astype(float).tolist()
    changes = data["correction_change_m"].astype(float).tolist()
    labels = data["is_unusual"].astype(bool).tolist()
    x_min, x_max = min(sols), max(sols)
    y_min, y_max = 0.0, max(changes) * 1.08
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    points = [
        (
            margin + (x - x_min) / max(x_max - x_min, 1) * plot_width,
            height - margin - y / max(y_max, 1) * plot_height,
            flagged,
        )
        for x, y, flagged in zip(sols, changes, labels)
    ]
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{3.3 if flagged else 2.1}" fill="{"#c44737" if flagged else "#66828a"}" opacity="0.8"/>'
        for x, y, flagged in points
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fcfaf6"/>
<text x="{width / 2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="21" fill="#302820">Unusual changes in rover localization correction</text>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#635d55"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#635d55"/>
{circles}
<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-family="sans-serif" font-size="14">Martian sol</text>
<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="sans-serif" font-size="14">Correction change between checkpoints (m)</text>
<circle cx="{width - 230}" cy="55" r="4" fill="#66828a"/><text x="{width - 218}" y="60" font-family="sans-serif" font-size="13">ordinary intervals</text>
<circle cx="{width - 90}" cy="55" r="5" fill="#c44737"/><text x="{width - 78}" y="60" font-family="sans-serif" font-size="13">flagged</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
