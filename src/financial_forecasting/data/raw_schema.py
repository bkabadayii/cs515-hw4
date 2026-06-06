"""Raw data schema definition and validation routines.

:func:`validate_raw_data` checks the combined OHLC DataFrame for:

1. **Schema** - required columns are present and have the right dtypes.
2. **Missing OHLC values** - no NaN in Open, High, Low, or Close.
3. **Duplicate rows** - no duplicate ``(Date, Ticker)`` pairs.
4. **Date coverage** - each ticker covers the expected start/end window.

Validation is strict: any failing check raises a ``ValueError`` so that
downstream phases never silently operate on corrupt data.
"""
from typing import TypedDict

import pandas as pd

from financial_forecasting.parameters.data_params import DataDownloadParams


class TickerSummary(TypedDict):
    rows: int
    min_date: str | None
    max_date: str | None


class ValidationSummary(TypedDict):
    passed: bool
    errors: list[str]
    warnings: list[str]
    ticker_summaries: dict[str, TickerSummary]
    total_rows: int
    tickers_found: list[str]


def validate_raw_data(
    df: pd.DataFrame, params: DataDownloadParams
) -> ValidationSummary:
    """Validate the combined raw OHLC DataFrame against all schema rules.

    Parameters
    ----------
    df:
        Combined long-format DataFrame as returned by
        :func:`~financial_forecasting.data.yfinance_client.download_tickers`.
    params:
        Download configuration dataclass (used for required columns and
        expected date bounds).

    Returns
    -------
    dict[str, object]
        Validation summary.  Keys include ``"passed"``, ``"errors"``,
        ``"warnings"``, ``"ticker_summaries"``.

    Raises
    ------
    ValueError
        If any hard validation check fails.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Schema check: required columns present
    # ------------------------------------------------------------------
    missing_cols = [c for c in params.required_columns if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # ------------------------------------------------------------------
    # 2. Dtype checks
    # ------------------------------------------------------------------
    date_dtype_ok = False
    if "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            errors.append(
                f"Column 'Date' has dtype {df['Date'].dtype}, expected datetime64."
            )
        else:
            date_dtype_ok = True

    ohlc_cols = ["Open", "High", "Low", "Close"]
    for col in ohlc_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' has non-numeric dtype {df[col].dtype}.")

    # ------------------------------------------------------------------
    # 3. Missing OHLC values
    # ------------------------------------------------------------------
    for col in ohlc_cols:
        if col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                errors.append(f"Column '{col}' has {n_null} missing value(s).")

    # ------------------------------------------------------------------
    # 4. Duplicate (Date, Ticker) rows
    # ------------------------------------------------------------------
    if "Date" in df.columns and "Ticker" in df.columns:
        dup_mask = df.duplicated(subset=["Date", "Ticker"])
        n_dup = int(dup_mask.sum())
        if n_dup > 0:
            errors.append(
                f"Found {n_dup} duplicate (Date, Ticker) row(s)."
            )

    # ------------------------------------------------------------------
    # 5. Date coverage per ticker
    # ------------------------------------------------------------------
    expected_start = pd.Timestamp(params.start_date)
    # end_date is exclusive in yfinance; last expected date is end_date - 1 day
    expected_end = pd.Timestamp(params.end_date) - pd.Timedelta(days=1)

    ticker_summaries: dict[str, TickerSummary] = {}
    if date_dtype_ok and "Date" in df.columns and "Ticker" in df.columns:
        for ticker in params.tickers:
            sub = df[df["Ticker"] == ticker]
            if sub.empty:
                errors.append(f"Ticker '{ticker}' has zero rows in combined data.")
                ticker_summaries[ticker] = {
                    "rows": 0,
                    "min_date": None,
                    "max_date": None,
                }
                continue

            min_date = sub["Date"].min()
            max_date = sub["Date"].max()
            n_rows = len(sub)

            ticker_summaries[ticker] = {
                "rows": n_rows,
                "min_date": str(min_date.date()),
                "max_date": str(max_date.date()),
            }

            # Allow up to 7 calendar-day slack at each boundary to account
            # for weekends and market holidays.
            slack = pd.Timedelta(days=7)
            if min_date > expected_start + slack:
                warnings.append(
                    f"Ticker '{ticker}' first date {min_date.date()} "
                    f"is more than 7 days after expected start {expected_start.date()}."
                )
            if max_date < expected_end - slack:
                errors.append(
                    f"Ticker '{ticker}' last date {max_date.date()} "
                    f"is more than 7 days before expected end {expected_end.date()}."
                )

    # ------------------------------------------------------------------
    # 6. Raise if any hard error was detected
    # ------------------------------------------------------------------
    if errors:
        raise ValueError(
            "Raw data validation FAILED.\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return {
        "passed": True,
        "errors": errors,
        "warnings": warnings,
        "ticker_summaries": ticker_summaries,
        "total_rows": len(df),
        "tickers_found": (
            sorted(df["Ticker"].unique().tolist())
            if "Ticker" in df.columns
            else []
        ),
    }
