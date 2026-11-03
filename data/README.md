# Data

## Raw market records (not redistributed)

The experiments read the official downloads of the Italian power exchange (GME):

| File | Content | Used for |
|---|---|---|
| `20230101_20231231_PUN.xlsx`, `20240101_20241231_PUN.xlsx` | Hourly day-ahead prices (column `PUN`) | Day-ahead leg of every unit |
| `EsitiMSD_PrezziExAnte_<from>_<to>_Nord.xlsx` | Ex-ante outcomes of the Italian Ancillary Services Market (MSD), Nord zone, covering 2023 and 2024 | Award probabilities and energy prices of aFRR and mFRR |

They are published at https://www.mercatoelettrico.org (Download section: day-ahead market results and MSD ex-ante outcomes). Save them in `data/raw/` with the names given by the exchange: `experiments.py` takes the PUN files through `--pun-xlsx` and any number of MSD files through `--msd-xlsx`, and keeps for each year only the MSD records of that year.

## Processed inputs (in this repository)

`processed/market_inputs_<year>.csv` gives, for every hour of 2023 and 2024, the inputs exactly as they enter the scheduling model in the reference runs:

- `day_ahead_price_eur_mwh`;
- for each product (`fcr_up`, `fcr_dn`, `afrr_up`, `afrr_dn`, `mfrr_up`, `mfrr_dn`): award probability, expected activation, capacity price (FCR: half of the symmetric price in each direction), and energy price (for FCR, the day-ahead price of the hour, at which the expected activated energy is settled).

These files are produced by `src/export_inputs.py` from the stored reference schedules and allow the inputs to be inspected without downloading the raw records. The FCR award probability is an assumed time-of-day profile, and mFRR takes 0.6 of the aFRR award probabilities at the same prices (Section 2.1 and Section S1 of the Supplementary Material of the paper).

In 52 hours of 2024 (weekend mornings at 07:00 and 08:00 between September and November) the MSD records show no upward acceptance in the Nord zone, and `market_data.py` treats the empty records as missing, assigning to the upward products its default profile (award probability 0.65 at 250 EUR/MWh for aFRR and 0.55 at 180 EUR/MWh for mFRR) in place of a null probability. The processed inputs and the results reproduce the reference runs as built; solving the 2024 reference again with a null upward award probability in those hours lowers the value of aggregation by 11.0 kEUR, from 1464.2 to 1453.2 kEUR, and the value of admission by 1.4 kEUR/(MW year), from 269.4 to 268.0 kEUR/(MW year) (Section S1 of the Supplementary Material). Both runs are reproduced by `src/check_fallback_2024.py`, whose results are stored in `results/fallback_2024.json`.
