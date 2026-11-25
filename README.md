# Curiosity's Traverse: NASA Rover Position Analysis

A reproducible analysis of real Curiosity rover localization records from NASA's Planetary Data System (PDS), with an exploratory Isolation Forest review aid.

## Project question

How does Curiosity's onboard position estimate compare with the refined localization team's estimate, and how does the difference change over the rover's traverse?

## Data

We start with NASA's `localized_pos.csv` file. It contains position and orientation rows in several coordinate frames, including northing/easting, latitude/longitude, attitude, and mission sol. NASA describes the archive and its CSV views in the [PDS dataset record](https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=MSL-M-ROVER-6-RDR-PLACES-V1.0) and [PLACES file guide](https://planetarydata.jpl.nasa.gov/img/data/msl/msl_places/document/PLACES_PDS_SIS.pdf).

The public CSV is included in `data/` so the project works offline after cloning. It is about 341 KB. No API key, account, or paid service is needed.

## Lesson 1: inspect the data

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python explore_curiosity.py
```

The first lesson prints the dataset's shape, column names, a few sample rows, and missing-value counts.

## Lesson 2: map the traverse

```bash
python analyze_curiosity.py
```

This removes the `drive = 0` convenience rows, orders the remaining rover drive records by sol, estimates distances between recorded endpoints, and draws the traverse as an SVG. These straight-line distances are an approximation; they don't include every turn or movement during a drive. Results go into the ignored `artifacts/` folder.

## Lesson 3: compare rover and localization positions

Download NASA's larger `telemetry.csv` view (about 24 MB) into `data/`, then run:

```bash
curl -fL -o data/telemetry.csv https://planetarydata.jpl.nasa.gov/img/data/msl/msl_places/data_localizations/telemetry.csv
python compare_localization.py
```

The analysis matches each localized checkpoint with the highest-pose telemetry position for the same site and drive, then measures the 2D distance between them. NASA notes that telemetry preserves the rover's position before ground adjustment, while `localized_pos` contains the localization team's refined locations. It describes comparing these views as a way to estimate accumulated position disagreement (including effects such as wheel slip). This is a simple educational comparison, not a mission-grade navigation estimate. See the [NASA PLACES guide](https://planetarydata.jpl.nasa.gov/img/data/msl/msl_places/document/PLACES_PDS_SIS.pdf).

![Curiosity traverse map](docs/images/curiosity_traverse.svg)

![Position disagreement by sol](docs/images/localization_drift.svg)

## Planned project stages

1. Inspect columns, coordinate frames, and missing values. **Done.**
2. Clean the table and map the traverse. **Done.**
3. Compare telemetry positions with localized positions and quantify localization drift. **Done.**
4. Add a data-driven detector for unusual position changes, then discuss its limits. **Done.**

## Current result

With the current PDS files, the comparison matched 1,228 localized checkpoints from sols 16–4865. The median distance between the rover-reported position and the localization team's refined position was 171.46 m (mean 156.66 m). This is disagreement between two position products—not a claim that the rover was physically 171 m from its true location. NASA describes the localized positions as map-refined and explains that the rover-reported positions can accumulate error over time.

The current Isolation Forest run flags 37 of 1,227 checkpoint intervals. Flagged intervals have a median correction change of 3.38 m, compared with 0.22 m for the others. This is a descriptive result from the chosen model settings, not a validated fault-detection score.

## Lesson 4: flag unusual corrections with machine learning

After Lesson 3, run:

```bash
python detect_adjustments.py
```

This Isolation Forest uses three features: the change in the localization correction vector, the distance between rover-reported checkpoints, and the distance between localized checkpoints. It flags the most unusual 3% of checkpoint intervals for human review. Localization is sparse, so an interval can span more than one drive. There are no ground-truth fault labels in this analysis, so a flag means “unusual relative to this dataset,” not “the rover failed.”

![Unusual correction intervals](docs/images/unusual_adjustments.svg)

## Limitations and security notes

- The rover-reported and localized positions are two data products, not independent ground truth. Their difference is position disagreement, not verified absolute error.
- `localized_pos` is sparse. It records localization checkpoints, not every position during every drive.
- The Isolation Forest's 3% contamination setting chooses the flag rate. There are no labels here to validate precision, recall, or fault-detection ability.
- This is a local analysis tool, not a deployed service. It does not accept credentials, expose a network port, or execute downloaded code. The 24 MB telemetry CSV is excluded from Git.
- No dependency vulnerability scan has been run. Direct package versions are pinned in `requirements.txt`, but transitive dependencies are not locked; review advisories before using this in a deployed system.

## Source and citation

NASA Planetary Data System, Mars Science Laboratory (Curiosity) PLACES Rover Motion Counter archive, `MSL-M-ROVER-6-RDR-PLACES-V1.0`. The dataset includes position solutions from multiple sources; NASA notes that telemetry positions can accumulate error and that localized solutions are refined against the base map.
