# Nigeria Flood Outlook

Monthly river flood risk (low / medium / high) for Nigerian states, using
GloFAS river discharge forecasts where reliable and a historical
machine-learning model otherwise.

**Live app:** https://flood-forecast-jyk4.onrender.com
(It may take about a minute to load if nobody has used it recently.)

**Not an official warning.** Check NiMet, NIHSA and NEMA for official alerts.
Coastal and tidal flooding is not covered.

## Run locally

    pip install -r requirements.txt
    python app.py

Open http://localhost:5055

## Data sources

- River discharge: GloFAS / Copernicus Emergency Management Service
- Flood labels: derived from EM-DAT (not included; request from emdat.be)
- Map tiles: © OpenStreetMap contributors
- Safety advice: NEMA public advisories

## Licence

Code: MIT.
