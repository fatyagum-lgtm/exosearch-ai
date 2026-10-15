# Dataset Card — NASA TESS Objects of Interest

**Dataset:** TESS Project Candidates (TOI) table from NASA Exoplanet Archive.

**Official documentation:** https://exoplanetarchive.ipac.caltech.edu/docs/TESSMission.html

**Acquisition:** `exosearch fetch --output data/raw/nasa_toi.csv` uses the documented TAP API service at `https://exoplanetarchive.ipac.caltech.edu/TAP/sync`. Query is stored in `data/raw/nasa_toi.manifest.json` alongside UTC acquisition timestamp and SHA-256.

**Features:** `pl_orbper`, `pl_trandurh`, `pl_trandep`, `pl_rade`, `st_tmag`, `st_teff`, `st_logg`, `st_rad`, `st_dist`.

**Outcome definition:** `tfopwg_disp`; see README mapping. These are working-group dispositions, not a perfectly unbiased class label.

**Unit of analysis:** TOI (planet-like candidate), with group ID TIC (`tid`) representing its host star.

**Data inclusion policy:** Real NASA CSVs are fetched on demand and excluded from Git by default. Synthetic test fixtures must be labeled unmistakably as simulated. This ZIP includes a locally generated synthetic example only.

**Important caution:** Current NASA catalog entries can have been updated after follow-up, making historical forecasting claims invalid without time-stamped data snapshots.

**Citation:** Cite the NASA Exoplanet Archive per official guidance if publishing scientific results; the archive curates upstream mission and follow-up data.

**Local environment limitation:** The archive was unreachable from this development environment. No genuine NASA experiment measurements are claimed in this delivery.
