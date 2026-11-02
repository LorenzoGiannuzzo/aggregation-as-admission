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
- for each product (`fcr_up`, `fcr_dn`, `afrr_up`, `afrr_dn`, `mfrr_up`, `mfrr_dn`): award probability, expected activation, capacity price (FCR: half of the symmetric price in each direction), and energy price.

These files are produced by `src/export_inputs.py` from the stored reference schedules and allow the inputs to be inspected without downloading the raw records. The FCR award probability is an assumed time-of-day profile, and mFRR takes 0.6 of the aFRR award probabilities at the same prices (Section 2.1 and Section S1 of the Supplementary Material of the paper).
