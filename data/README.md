# Data

| File | Source | Licence | In the repo |
|---|---|---|---|
| `raw/PrixCarburants_annuel_2026.xml` | https://donnees.roulez-eco.fr/opendata/annee/2026 (zip) | Licence Ouverte 2.0 | no, download and unzip here |
| `raw/osm_fr_fuel_stations.csv` | OpenStreetMap via Overpass: `amenity=fuel` in France with `ref:FR:prix-carburants`, `brand`, `operator`, `name` | ODbL, © OpenStreetMap contributors | yes |
| `raw/tankerkoenig/` (part 2) | Tankerkönig / MTS-K historic prices and stations (account required) | CC BY 4.0 | no |

`python -m fuellab.pipeline aggregate-fr` creates the flat price and stockout files in `raw/` and the station-day
panels in `processed/`. The results in `reports/cap/` were produced from the annual file as published on 7 October 2026.

Overpass query used for the OSM file:

```
[out:csv(::type,::id,::lat,::lon,"ref:FR:prix-carburants",brand,operator,name;true;",")][timeout:300];
area["ISO3166-1"="FR"][admin_level=2]->.fr;
nwr["amenity"="fuel"](area.fr);
out center tags;
```
