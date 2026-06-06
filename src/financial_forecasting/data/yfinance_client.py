"""yfinance client: download and normalise raw OHLC data.

This module provides :func:`download_tickers`, which fetches daily OHLC
bars from Yahoo Finance for a list of ticker symbols, reshapes the
result into a tidy long-format DataFrame, saves one CSV per ticker, and
saves a combined CSV containing all tickers.

Design decisions
----------------
* The function accepts a :class:`~financial_forecasting.parameters.data_params.DataDownloadParams`
  dataclass so that every tunable setting is explicit and testable.
* Column names are standardised to ``["Date", "Ticker", "Open",
  "High", "Low", "Close"]`` regardless of what yfinance returns.
* ``Volume`` is dropped because the project uses only OHLC.
* The ``Date`` index is reset to a regular column and stored as a
  ``datetime64[ns]`` dtype.
* Tickers are downloaded individually to avoid the yfinance SQLite
  ``database is locked`` error that occurs with concurrent downloads.
"""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from financial_forecasting.parameters.data_params import DataDownloadParams


# Columns we keep after downloading (Volume excluded by design).
_OHLC_COLS = ["Open", "High", "Low", "Close"]


def _download_single_ticker(
    ticker: str, params: DataDownloadParams, retries: int = 3
) -> pd.DataFrame:
    """Download OHLC data for one ticker with retry logic.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol.
    params:
        Download configuration dataclass.
    retries:
        Number of attempts before giving up.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns
        ``["Date", "Ticker", "Open", "High", "Low", "Close"]``.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            raw: pd.DataFrame = yf.download(
                tickers=ticker,
                start=params.start_date,
                end=params.end_date,
                interval=params.interval,
                auto_adjust=params.auto_adjust,
                # Use threads=False per-ticker to avoid SQLite lock issues.
                threads=False,
                progress=False,
            )
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"All {retries} download attempts failed for {ticker}."
                ) from exc
            wait = 2 ** attempt
            print(f"  [WARN] attempt {attempt} for {ticker} failed ({exc}); retrying in {wait}s ...")
            time.sleep(wait)
            continue

        # yfinance returns a flat index for a single ticker.
        if raw.empty:
            if attempt == retries:
                raise RuntimeError(
                    f"yfinance returned empty data for {ticker} after {retries} attempts."
                )
            wait = 2 ** attempt
            print(f"  [WARN] attempt {attempt} for {ticker} returned empty; retrying in {wait}s ...")
            time.sleep(wait)
            continue

        # Check for all-NaN OHLC (the SQLite-lock symptom).
        if raw[_OHLC_COLS].isna().all(axis=None):
            if attempt == retries:
                raise RuntimeError(
                    f"yfinance returned all-NaN OHLC for {ticker} after {retries} attempts."
                )
            wait = 2 ** attempt
            print(f"  [WARN] attempt {attempt} for {ticker} returned NaN; retrying in {wait}s ...")
            time.sleep(wait)
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            xs_df = raw.xs(ticker, axis=1, level=1, drop_level=True)
            assert isinstance(xs_df, pd.DataFrame)
            raw = xs_df

        df = raw[_OHLC_COLS].copy()
        df.index.name = "Date"
        df = df.reset_index()
        df["Ticker"] = ticker
        df = df[["Date", "Ticker"] + _OHLC_COLS]
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    raise RuntimeError(f"Download failed for {ticker}.")  # unreachable but satisfies type checker


def download_tickers(params: DataDownloadParams) -> pd.DataFrame:
    """Download OHLC data for all tickers and persist CSVs.

    Downloads each ticker individually to avoid concurrent SQLite locks.

    Steps
    -----
    1. For each ticker: fetch from Yahoo Finance with retry logic.
    2. Reshape to long format with ``Date, Ticker, Open, High, Low, Close``.
    3. Save one CSV per ticker under ``params.raw_data_dir``.
    4. Save the combined CSV at ``params.combined_raw_path``.

    Parameters
    ----------
    params:
        Download configuration dataclass.

    Returns
    -------
    pd.DataFrame
        Combined long-format DataFrame with all tickers.
    """
    params.raw_data_dir.mkdir(parents=True, exist_ok=True)
    params.combined_raw_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tickers: {params.tickers}")
    print(f"  start={params.start_date}  end={params.end_date}  interval={params.interval}")

    frames: list[pd.DataFrame] = []
    for ticker in params.tickers:
        ticker_df = _download_single_ticker(ticker, params)
        out_path = params.raw_data_dir / f"{ticker}.csv"
        ticker_df.to_csv(out_path, index=False)
        print(f"  Saved {len(ticker_df):>5} rows -> {out_path}")
        frames.append(ticker_df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Save combined CSV.
    combined.to_csv(params.combined_raw_path, index=False)
    print(f"  Saved {len(combined):>5} rows (combined) -> {params.combined_raw_path}")

    return combined
