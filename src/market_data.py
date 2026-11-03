"""
Market data of the Italian day-ahead market and of the reserve products.

  - load_pun_from_gme_xlsx_multi: hourly day-ahead prices from the official
    GME .xlsx downloads (PUN or a zonal column);
  - msd_conditioned_profiles: award probabilities and mean accepted energy
    prices of the restoration products from the ex-ante MSD outcomes, by
    season, day type, hour and direction (one year or several pooled);
  - build_overrides_from_conditioned_profiles and build_yearly_service_catalog:
    the hourly catalog of FlexibilityService objects (FCR, aFRR and mFRR,
    upward and downward) used by the scheduling model;
  - MarketWindow: the prices and services of a run, sliced day by day.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from market_products import FlexibilityService, ServiceType


# ============================================================================
# PUN synthetic generator (calibrated against 2024 statistical properties)
# ============================================================================


# ============================================================================
# Real-data loader (CSV)
# ============================================================================


def load_pun_from_gme_xlsx(
    xlsx_path: str,
    sheet_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    price_column: str = "PUN",
) -> Tuple[np.ndarray, List[datetime]]:
    """Load hourly day-ahead prices from the official GME .xlsx download.

    ZONAL vs PUN
    ------------
    The PUN is the national single price paid by CONSUMERS. Generation and
    storage settle at the ZONAL price of the bidding zone they sit in. The
    paper states that MSD data is taken for the northern zone, so the
    day-ahead leg has to be the NORD zonal series, not the PUN, or the two
    revenue streams are drawn from two different price references.

    `price_column` selects which series to read:

        "PUN"   the national single price (legacy behaviour). Correct for a
                consumer-side study, wrong for a storage operator.
        "NORD"  northern-zone price. This is the one a BESS in the north
                actually settles at, and the one to use here.
        "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD" other zones.

    The GME "MGPPrezzi" workbook carries one column per zone plus PUN, so a
    zonal run needs that file rather than the "MGP-PUNPUN" export, which
    only carries the national price. When the requested column is absent the
    function raises rather than silently falling back, because a silent
    fallback to PUN is exactly the failure this parameter exists to prevent.

    The GME download is a single-sheet Excel file (the active sheet is
    typically named 'MGP-PUNPUN' or 'MGP-Prezzi-PUN') with three columns:

        Data   : date in DD/MM/YYYY string format
        Ora    : sequential hour index within the day. By GME convention
                 Ora=1 corresponds to the FIRST clock interval of the day;
                 for normal 24-hour days this is 00:00-01:00 (local time).
        €/MWh  : price as a string with comma decimal separator (Italian
                 locale), e.g. '107,090000'.

    DST handling: in 2024 the spring transition (31 March) has 23 hourly
    entries (Ora 1-23) and the autumn transition (27 October) has 25
    entries (Ora 1-25). To produce a CONTIGUOUS 24-hour-per-day output
    suitable for the MILP/env pipeline:
      - On a short day (23 entries) we DUPLICATE the first value of the
        gap to fill position 2 of the standardised 24-hour vector.
      - On a long day (25 entries) we DROP the extra entry (the DST
        duplicate), keeping the 24 standardised hours of the day.
    These choices are approximations; for a paper-grade run we recommend
    trimming the test window to avoid DST days, or post-processing the
    array externally with the policy of your choice.

    Parameters
    ----------
    xlsx_path : str
        Path to the .xlsx file.
    sheet_name : str, optional
        Name of the sheet to read. If None, the active (first) sheet is
        used, which matches the standard GME export.
    start_date, end_date : datetime, optional
        Bound the returned series. If supplied, only data in
        [start_date, end_date) is returned.

    Returns
    -------
    prices : np.ndarray of shape (n_hours,)
        Hourly PUN prices in EUR/MWh, on a 24-hour-per-day grid.
    stamps : list of datetime, length n_hours
        Hour-start datetime for each price.
    """
    from openpyxl import load_workbook
    from collections import defaultdict

    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet_name is None:
        ws = wb.active
    else:
        ws = wb[sheet_name]

    # First pass: group entries by date in file order
    import collections as _collections
    # date -> list of hourly prices, in file order
    rows_by_date = _collections.OrderedDict()
    seen_header = False
    # Column index of the requested price series, resolved from the header
    # row. The single-price "MGP-PUNPUN" export has the price in column 2;
    # the multi-zone "MGPPrezzi" export has one column per bidding zone, so
    # the index has to be looked up by name.
    price_idx = 2
    want = str(price_column).strip().upper()
    for row in ws.iter_rows(values_only=True):
        if row is None or row[0] is None:
            continue
        if not seen_header:
            seen_header = True
            header = [str(c).strip().upper() if c is not None else ""
                      for c in row]
            matches = [i for i, h in enumerate(header) if h == want]
            if matches:
                price_idx = matches[0]
            elif want not in ("PUN", "\u20ac/MWH", "EUR/MWH"):
                raise ValueError(
                    f"price_column={price_column!r} not found in {xlsx_path}. "
                    f"Header columns: {header}. The single-price "
                    f"'MGP-PUNPUN' export does not contain zonal columns; "
                    f"use the multi-zone 'MGPPrezzi' workbook for zonal "
                    f"prices.")
            continue
        data_str, ora, price = row[0], row[1], row[price_idx]
        if isinstance(data_str, datetime):
            day = data_str.date()
        else:
            day = datetime.strptime(str(data_str), "%d/%m/%Y").date()
        if isinstance(price, str):
            price_f = float(price.replace(",", "."))
        else:
            price_f = float(price)
        rows_by_date.setdefault(day, []).append(price_f)

    # Second pass: build the 24-hour-per-day output, handling DST anomalies
    prices_list: List[float] = []
    stamps_list: List[datetime] = []
    for day, entries in rows_by_date.items():
        n = len(entries)
        if n == 24:
            normalised = list(entries)
        elif n == 23:
            # Spring DST: insert a synthetic value at position 2 (the missing
            # 02:00-03:00 local hour). Use the average of its neighbours.
            normalised = list(entries[:2]) + \
                          [(entries[1] + entries[2]) / 2.0] + \
                          list(entries[2:])
        elif n == 25:
            # Autumn DST: drop the duplicate hour (the second 02:00-03:00).
            # GME puts the duplicate at Ora=4 in this file, but for safety
            # we drop position 3 and keep position 2.
            normalised = list(entries[:3]) + list(entries[4:])
        else:
            # Unexpected day length; truncate or pad to 24
            if n > 24:
                normalised = list(entries[:24])
            else:
                normalised = list(entries) + [entries[-1]] * (24 - n)
        for h in range(24):
            ts = datetime(day.year, day.month, day.day, h)
            if start_date is not None and ts < start_date:
                continue
            if end_date is not None and ts >= end_date:
                continue
            prices_list.append(normalised[h])
            stamps_list.append(ts)

    prices = np.asarray(prices_list, dtype=np.float64)
    return prices, stamps_list


def load_pun_from_gme_xlsx_multi(
    xlsx_paths,
    sheet_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    price_column: str = "PUN",
) -> Tuple[np.ndarray, List[datetime]]:
    """Load and concatenate hourly PUN prices from one or more GME .xlsx files.

    Accepts either a single path (str) or a list of paths. When a list is
    given, each file is loaded with ``load_pun_from_gme_xlsx``, the resulting
    series are concatenated, **deduplicated by timestamp** (last wins, so
    later files in the list override earlier ones for overlapping hours), and
    sorted chronologically. The optional ``start_date``/``end_date`` window
    is applied AFTER merging, so it works regardless of how the files are
    split.

    Use this when train and test windows span calendar years and you have
    one xlsx per year (e.g. ``["2023_PUN.xlsx", "2024_PUN.xlsx"]``).
    """
    if isinstance(xlsx_paths, (str, bytes)):
        xlsx_paths = [xlsx_paths]

    all_prices: List[float] = []
    all_stamps: List[datetime] = []
    for p in xlsx_paths:
        prices_i, stamps_i = load_pun_from_gme_xlsx(
            p, sheet_name=sheet_name, price_column=price_column)
        all_prices.extend(prices_i.tolist())
        all_stamps.extend(stamps_i)

    # Dedup by timestamp (last write wins) and sort.
    by_ts: "OrderedDict[datetime, float]" = OrderedDict()
    for ts, pr in zip(all_stamps, all_prices):
        by_ts[ts] = float(pr)
    sorted_items = sorted(by_ts.items(), key=lambda kv: kv[0])
    stamps_sorted = [ts for ts, _ in sorted_items]
    prices_sorted = np.array([pr for _, pr in sorted_items], dtype=float)

    # Window filter (after merge).
    if start_date is not None or end_date is not None:
        mask = np.ones(len(stamps_sorted), dtype=bool)
        if start_date is not None:
            mask &= np.array([ts >= start_date for ts in stamps_sorted])
        if end_date is not None:
            mask &= np.array([ts < end_date for ts in stamps_sorted])
        prices_sorted = prices_sorted[mask]
        stamps_sorted = [ts for ts, m in zip(stamps_sorted, mask) if m]

    return prices_sorted, stamps_sorted


# ============================================================================
# Annual flexibility service catalog
# ============================================================================

@dataclass
class ServiceCalibration:
    """Italian ancillary service capacity/energy prices.

    Defaults are calibrated against published 2022-2024 Italian benchmarks.
    Source attributions:

      - Capacity prices: Magnus Energy 2025 "Developments in balancing and
        capacity markets"; ACER CHEST database; Terna Report on Balancing
        2022-2023. FCR uses the European FCR Cooperation 2024 benchmark
        because Italian market-based FCR is not yet operational under TIDE.
      - Energy activation prices: ACER CHEST 2022 volume-weighted aFRR
        Italy upward (411.99 EUR/MWh, rounded to 250 conservative);
        mFRR European benchmark.
      - Activation probabilities: Terna Report on Balancing 2022-2023.

    For real hourly variation in energy prices, supply the per-hour
    `energy_price_overrides` arrays (FCR / aFRR / mFRR) to
    `build_yearly_service_catalog`. Typically these come from the MSD
    ex-ante data via `load_msd_from_terna_xlsx`.
    """
    # Capacity prices (EUR per MW per hour), assumed prospective prices (Section 2.1 of the paper)
    fcr_capacity_price: float = 50.0       # assumed symmetric FCR price, half per direction
    afrr_capacity_price: float = 25.0      # assumed aFRR capacity price
    mfrr_capacity_price: float = 12.0      # assumed mFRR capacity price
    # Default energy prices for activated capacity (EUR per MWh), replaced by the MSD records where available
    fcr_energy_price: float = 0.0          # FCR energy settled at the day-ahead price in scheduling_milp.py
    afrr_energy_price: float = 250.0       # default profile
    mfrr_energy_price: float = 180.0       # default profile
    # Assumed expected activations of the awarded band
    fcr_activation_probability: float = 0.10
    afrr_activation_probability: float = 0.30
    mfrr_activation_probability: float = 0.20
    # Assumed time-of-day award probability pattern of FCR
    award_prob_night: float = 0.80      # 00-06
    award_prob_morning: float = 0.55    # 06-09, 09-12, 12-17
    award_prob_peak: float = 0.30       # 17-21
    award_prob_evening: float = 0.50    # 21-24
    # Service-specific offsets vs the time-of-day baseline
    fcr_award_offset:  float = -0.15
    afrr_award_offset: float =  0.10
    mfrr_award_offset: float =  0.00


def _time_of_day_award_probability(hour: int, base_calibration: ServiceCalibration) -> float:
    """Return the baseline award probability for the given hour of day."""
    if 0 <= hour < 6:
        return base_calibration.award_prob_night
    elif 6 <= hour < 17:
        return base_calibration.award_prob_morning
    elif 17 <= hour < 21:
        return base_calibration.award_prob_peak
    else:
        return base_calibration.award_prob_evening


def build_yearly_service_catalog(
    start_date: datetime,
    end_date: datetime,
    calibration: Optional[ServiceCalibration] = None,
    afrr_energy_overrides: Optional[np.ndarray] = None,
    mfrr_energy_overrides: Optional[np.ndarray] = None,
    fcr_energy_overrides: Optional[np.ndarray] = None,
    afrr_award_overrides: Optional[np.ndarray] = None,
    mfrr_award_overrides: Optional[np.ndarray] = None,
    fcr_award_overrides: Optional[np.ndarray] = None,
    # NEW directional inputs - if any are provided, builder emits the 5-service
    # directional catalog (FCR + aFRR_up + aFRR_dn + mFRR_up + mFRR_dn)
    afrr_up_energy_overrides: Optional[np.ndarray] = None,
    afrr_dn_energy_overrides: Optional[np.ndarray] = None,
    mfrr_up_energy_overrides: Optional[np.ndarray] = None,
    mfrr_dn_energy_overrides: Optional[np.ndarray] = None,
    afrr_up_award_overrides: Optional[np.ndarray] = None,
    afrr_dn_award_overrides: Optional[np.ndarray] = None,
    mfrr_up_award_overrides: Optional[np.ndarray] = None,
    mfrr_dn_award_overrides: Optional[np.ndarray] = None,
    # Explicit request for the 5-service directional catalogue. Until now the
    # only way to get it was to pass a directional override array, so purely
    # synthetic data could not be directional at all: a directional consumer
    # given such a catalogue finds no aFRR_up / aFRR_dn / mFRR_up / mFRR_dn on
    # offer and silently trades arbitrage only.
    directional: bool = False,
) -> List[List[FlexibilityService]]:
    """Build the per-hour FlexibilityService catalog for a contiguous date range.

    Returns a list of length n_hours, each entry a list of 3 services
    (FCR, aFRR, mFRR). The award_probability of each service combines the
    time-of-day base pattern with the service-specific offset, then clips
    to [0.05, 0.95].

    Optional per-hour overrides (priority over the synthetic profile when
    set, else fall back to calibration default):
      - `afrr_energy_overrides`, `mfrr_energy_overrides`, `fcr_energy_overrides`:
        replace the calibration's energy_price for the relevant service.
      - `afrr_award_overrides`, `mfrr_award_overrides`, `fcr_award_overrides`:
        replace the synthetic time-of-day award probability with a real
        empirical value (e.g. MSD activation rate per hour). Values of NaN
        or <=0 fall back to the synthetic profile for that hour.
    All overrides have shape (n_hours,) aligned with the catalog.
    """
    cal = calibration or ServiceCalibration()
    n_days = (end_date - start_date).days
    n_hours = n_days * 24
    catalog: List[List[FlexibilityService]] = []

    def _override(arr, default, h):
        if arr is None:
            return default
        if h >= len(arr):
            return default
        v = arr[h]
        if v is None or (isinstance(v, float) and (np.isnan(v) or v <= 0.0)):
            return default
        return float(v)

    # Directional mode: requested explicitly, or inferred from the presence of
    # any directional override (the original behaviour, kept for back-compat).
    directional_mode = bool(directional) or any(arr is not None for arr in (
        afrr_up_energy_overrides, afrr_dn_energy_overrides,
        mfrr_up_energy_overrides, mfrr_dn_energy_overrides,
        afrr_up_award_overrides,  afrr_dn_award_overrides,
        mfrr_up_award_overrides,  mfrr_dn_award_overrides,
    ))

    for h in range(n_hours):
        dt = start_date + timedelta(hours=h)
        base = _time_of_day_award_probability(dt.hour, cal)
        season_offset = 0.05 if dt.month in (1, 2, 12) else 0.0
        fcr_p_synth  = float(np.clip(base + cal.fcr_award_offset  - season_offset, 0.05, 0.95))
        afrr_p_synth = float(np.clip(base + cal.afrr_award_offset - season_offset, 0.05, 0.95))
        mfrr_p_synth = float(np.clip(base + cal.mfrr_award_offset - season_offset, 0.05, 0.95))
        fcr_p  = _override(fcr_award_overrides,  fcr_p_synth,  h)
        fcr_p  = float(np.clip(fcr_p,  0.05, 0.99))
        fcr_e  = _override(fcr_energy_overrides, cal.fcr_energy_price, h)

        if directional_mode:
            # Build the 5-service directional catalog
            afrr_up_p = float(np.clip(_override(afrr_up_award_overrides, afrr_p_synth, h), 0.05, 0.99))
            afrr_dn_p = float(np.clip(_override(afrr_dn_award_overrides, afrr_p_synth, h), 0.05, 0.99))
            mfrr_up_p = float(np.clip(_override(mfrr_up_award_overrides, mfrr_p_synth, h), 0.05, 0.99))
            mfrr_dn_p = float(np.clip(_override(mfrr_dn_award_overrides, mfrr_p_synth, h), 0.05, 0.99))
            afrr_up_e = _override(afrr_up_energy_overrides, cal.afrr_energy_price, h)
            afrr_dn_e = _override(afrr_dn_energy_overrides, cal.afrr_energy_price, h)
            mfrr_up_e = _override(mfrr_up_energy_overrides, cal.mfrr_energy_price, h)
            mfrr_dn_e = _override(mfrr_dn_energy_overrides, cal.mfrr_energy_price, h)
            hour_services = [
                FlexibilityService(
                    service_type=ServiceType.FCR,
                    capacity_price=cal.fcr_capacity_price,
                    energy_price=fcr_e,
                    activation_probability=cal.fcr_activation_probability,
                    response_time=30, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=fcr_p,
                ),
                FlexibilityService(
                    service_type=ServiceType.AFRR_UP,
                    capacity_price=cal.afrr_capacity_price,
                    energy_price=afrr_up_e,
                    activation_probability=cal.afrr_activation_probability,
                    response_time=200, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=afrr_up_p,
                ),
                FlexibilityService(
                    service_type=ServiceType.AFRR_DN,
                    capacity_price=cal.afrr_capacity_price,
                    energy_price=afrr_dn_e,
                    activation_probability=cal.afrr_activation_probability,
                    response_time=200, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=afrr_dn_p,
                ),
                FlexibilityService(
                    service_type=ServiceType.MFRR_UP,
                    capacity_price=cal.mfrr_capacity_price,
                    energy_price=mfrr_up_e,
                    activation_probability=cal.mfrr_activation_probability,
                    response_time=900, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=mfrr_up_p,
                ),
                FlexibilityService(
                    service_type=ServiceType.MFRR_DN,
                    capacity_price=cal.mfrr_capacity_price,
                    energy_price=mfrr_dn_e,
                    activation_probability=cal.mfrr_activation_probability,
                    response_time=900, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=mfrr_dn_p,
                ),
            ]
            catalog.append(hour_services)
            continue

        # Legacy 3-service path (back-compat)
        afrr_p = _override(afrr_award_overrides, afrr_p_synth, h)
        mfrr_p = _override(mfrr_award_overrides, mfrr_p_synth, h)
        afrr_p = float(np.clip(afrr_p, 0.05, 0.99))
        mfrr_p = float(np.clip(mfrr_p, 0.05, 0.99))
        afrr_e = _override(afrr_energy_overrides, cal.afrr_energy_price, h)
        mfrr_e = _override(mfrr_energy_overrides, cal.mfrr_energy_price, h)
        hour_services = [
            FlexibilityService(
                service_type=ServiceType.FCR,
                capacity_price=cal.fcr_capacity_price,
                energy_price=fcr_e,
                activation_probability=cal.fcr_activation_probability,
                response_time=30, min_capacity=1.0,
                max_duration=4, min_duration=1,
                award_probability=fcr_p,
            ),
            FlexibilityService(
                service_type=ServiceType.AFRR,
                capacity_price=cal.afrr_capacity_price,
                energy_price=afrr_e,
                activation_probability=cal.afrr_activation_probability,
                response_time=200, min_capacity=1.0,
                max_duration=4, min_duration=1,
                award_probability=afrr_p,
            ),
            FlexibilityService(
                service_type=ServiceType.MFRR,
                capacity_price=cal.mfrr_capacity_price,
                energy_price=mfrr_e,
                activation_probability=cal.mfrr_activation_probability,
                response_time=900, min_capacity=1.0,
                max_duration=4, min_duration=1,
                award_probability=mfrr_p,
            ),
        ]
        catalog.append(hour_services)
    return catalog


# ============================================================================
# Market window dataclass
# ============================================================================

@dataclass
class MarketWindow:
    """A contiguous period of market data ready to be consumed by the MILP/env."""
    start_date: datetime
    end_date: datetime
    prices_hourly: np.ndarray  # (n_hours,) EUR/MWh
    services_hourly: List[List[FlexibilityService]]  # (n_hours, 3 services)

    @property
    def n_hours(self) -> int:
        return len(self.prices_hourly)

    @property
    def n_days(self) -> int:
        return self.n_hours // 24

    def slice_day(self, day_index: int) -> Tuple[List[float], List[List[FlexibilityService]]]:
        """Return prices and services for day `day_index` (0-based)."""
        h0 = day_index * 24
        h1 = h0 + 24
        return list(self.prices_hourly[h0:h1]), list(self.services_hourly[h0:h1])


# ============================================================================
# MSD profiles conditioned on season x day type x hour: award probabilities and
# mean accepted energy prices of the restoration products, which in the 2023-2024
# records vary by about 20-30% between winter and summer and between working days
# and weekends (msd_conditioned_profiles + build_overrides_from_conditioned_profiles).
# ============================================================================

_SEASON_MAP_OPT2 = {12:'W',1:'W',2:'W',3:'Sp',4:'Sp',5:'Sp',
                    6:'Su',7:'Su',8:'Su',9:'A',10:'A',11:'A'}


def msd_conditioned_profiles(xlsx_paths, year=None, min_cell_samples: int = 20):
    """MSD ex-ante profiles conditioned on (season x daytype x hour).

    Returns dict with:
      - 'buckets': {(season, daytype): {4 length-24 arrays}}
      - 'fallback_*': hour-only marginal arrays (length 24), used where a cell
        has fewer than `min_cell_samples` observations.
    season in {'W','Sp','Su','A'}; daytype in {'wd','we'}.
    """
    import pandas as pd
    if isinstance(xlsx_paths, (str, bytes)):
        xlsx_paths = [xlsx_paths]
    df = pd.concat([pd.read_excel(p) for p in xlsx_paths], ignore_index=True)

    def _tf(s):
        return pd.to_numeric(s.astype(str).str.replace(',', '.'), errors='coerce')

    df['date'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df['hour'] = df['Ora'].astype(int) - 1
    df['p_buy'] = _tf(df['Prezzo Medio di Acquisto (€/MWh)'])
    df['p_sell'] = _tf(df['Prezzo Medio di Vendita (€/MWh)'])
    df = df.dropna(subset=['date'])
    df = df[(df['hour'] >= 0) & (df['hour'] <= 23)]
    if year is not None:
        df = df[df['date'].dt.year == year]
    df['season'] = df['date'].dt.month.map(_SEASON_MAP_OPT2)
    df['daytype'] = np.where(df['date'].dt.weekday < 5, 'wd', 'we')

    def _agg(sub):
        up_r = np.zeros(24); up_p = np.zeros(24)
        dn_r = np.zeros(24); dn_p = np.zeros(24)
        for h in range(24):
            s = sub[sub['hour'] == h]
            if len(s) == 0:
                continue
            # Direction mapping: accepted offers of sale ("offerte di vendita") are the
            # upward movement (the provider sells energy and is paid), accepted offers of
            # purchase ("offerte di acquisto") the downward one (the provider buys energy back).
            up_r[h] = float((s['p_sell'] > 0).mean())
            au = s[s['p_sell'] > 0]
            up_p[h] = float(au['p_sell'].mean()) if len(au) else 0.0
            dn_r[h] = float((s['p_buy'] > 0).mean())
            ad = s[s['p_buy'] > 0]
            dn_p[h] = float(ad['p_buy'].mean()) if len(ad) else 0.0
        return up_r, up_p, dn_r, dn_p

    fb = _agg(df)
    out = {
        'fallback_upward_award_rate': fb[0],
        'fallback_upward_price_when_active': fb[1],
        'fallback_downward_award_rate': fb[2],
        'fallback_downward_price_when_active': fb[3],
        'buckets': {},
    }
    for season in ['W', 'Sp', 'Su', 'A']:
        for dtp in ['wd', 'we']:
            sub = df[(df['season'] == season) & (df['daytype'] == dtp)]
            up_r, up_p, dn_r, dn_p = _agg(sub)
            for h in range(24):
                if len(sub[sub['hour'] == h]) < min_cell_samples:
                    up_r[h], up_p[h] = fb[0][h], fb[1][h]
                    dn_r[h], dn_p[h] = fb[2][h], fb[3][h]
            out['buckets'][(season, dtp)] = {
                'upward_award_rate': up_r, 'upward_price_when_active': up_p,
                'downward_award_rate': dn_r, 'downward_price_when_active': dn_p,
            }
    return out


def _bucket_for_date(dt: datetime):
    season = _SEASON_MAP_OPT2[dt.month]
    dtp = 'wd' if dt.weekday() < 5 else 'we'
    return (season, dtp)


def build_overrides_from_conditioned_profiles(
    conditioned: Dict,
    start_date: datetime,
    end_date: datetime,
    mfrr_award_scale: float = 0.6,
    directional: bool = False,
) -> Dict[str, np.ndarray]:
    """Like build_overrides_from_msd_profiles, but selects the (season, daytype)
    bucket of the ACTUAL calendar day for every hour in the window, so award
    rates and active prices vary day-by-day instead of using one yearly mean.

    Drop-in: same return keys as build_overrides_from_msd_profiles for both
    legacy and directional modes; mutex normalisation identical.
    """
    n_days = (end_date - start_date).days
    n_hours = n_days * 24
    buckets = conditioned['buckets']

    def prof_for(dt):
        b = buckets.get(_bucket_for_date(dt))
        if b is None:  # safety fallback to hour-only marginal
            return (conditioned['fallback_upward_award_rate'],
                    conditioned['fallback_upward_price_when_active'],
                    conditioned['fallback_downward_award_rate'],
                    conditioned['fallback_downward_price_when_active'])
        return (b['upward_award_rate'], b['upward_price_when_active'],
                b['downward_award_rate'], b['downward_price_when_active'])

    if not directional:
        afrr_award = np.zeros(n_hours); afrr_energy = np.zeros(n_hours)
        mfrr_award = np.zeros(n_hours); mfrr_energy = np.zeros(n_hours)
        for h in range(n_hours):
            dt = start_date + timedelta(hours=h)
            up_r, up_p, dn_r, dn_p = prof_for(dt)
            hh = dt.hour
            afrr_award[h] = up_r[hh]; afrr_energy[h] = up_p[hh]
            mfrr_award[h] = dn_r[hh] * mfrr_award_scale; mfrr_energy[h] = dn_p[hh]
        return {
            "afrr_award_overrides": afrr_award, "afrr_energy_overrides": afrr_energy,
            "mfrr_award_overrides": mfrr_award, "mfrr_energy_overrides": mfrr_energy,
        }

    # Directional: emit both directions, with the SAME mutex normalisation
    # (P_up_excl + P_dn_excl <= 1) as the original directional builder.
    keys = ["afrr_up", "afrr_dn", "mfrr_up", "mfrr_dn"]
    arr = {f"{k}_award_overrides": np.zeros(n_hours) for k in keys}
    arr.update({f"{k}_energy_overrides": np.zeros(n_hours) for k in keys})
    for h in range(n_hours):
        dt = start_date + timedelta(hours=h)
        up_r, up_p, dn_r, dn_p = prof_for(dt)
        hh = dt.hour
        p_up_raw, p_dn_raw = up_r[hh], dn_r[hh]
        total = p_up_raw + p_dn_raw
        if total <= 1.0:
            p_up_excl, p_dn_excl = p_up_raw, p_dn_raw
        else:
            p_up_excl, p_dn_excl = p_up_raw / total, p_dn_raw / total
        arr["afrr_up_award_overrides"][h] = p_up_excl
        arr["afrr_up_energy_overrides"][h] = up_p[hh]
        arr["afrr_dn_award_overrides"][h] = p_dn_excl
        arr["afrr_dn_energy_overrides"][h] = dn_p[hh]
        arr["mfrr_up_award_overrides"][h] = p_up_excl * mfrr_award_scale
        arr["mfrr_up_energy_overrides"][h] = up_p[hh]
        arr["mfrr_dn_award_overrides"][h] = p_dn_excl * mfrr_award_scale
        arr["mfrr_dn_energy_overrides"][h] = dn_p[hh]
    return arr