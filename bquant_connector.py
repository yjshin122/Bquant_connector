from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional, Sequence

import bql
import pandas as pd

logger = logging.getLogger(__name__)

# Ordered Bloomberg items → intermediate column names (before final display rename).
# All pulls are server-side; no client-side recomputation of these metrics.
FIELD_SPECS: list[tuple[str, str]] = [
    # Identifiers
    ("TICKER", "Ticker"),
    ("NAME", "NAME"),
    ("ID_ISIN", "ID_ISIN"),
    ("ID_BB_GLOBAL", "ID_BB_GLOBAL"),
    # Classifications
    ("GICS_INDUSTRY_NAME", "GICS_INDUSTRY_NAME"),
    ("MARKET_STATUS", "MARKET_STATUS"),
    # Primary listing exchange (FLDS alias commonly used in BQL)
    ("PRIMARY_EXCH_NAME", "PRIMARY_EXCHANGE"),
    # HK/Shanghai/Shenzhen Stock Connect eligibility (swap if your desk uses another FLDS)
    ("STK_MKT_CONNECT_ELIGIBILITY", "STOCK_CONNECT"),
    # Fundamentals (point-in-time / default fiscal context per Bloomberg item definition)
    ("RETURN_ON_INVESTED_CAPITAL", "ROIC (%)"),
    ("FREE_CASH_FLOW_YIELD", "FCF Yield (%)"),
    ("BS_TOT_ASSET", "Tot. Assets"),
    ("EBITDA", "EBITDA"),
    ("SALES_REV_TURN", "Revenue"),
    ("OPER_INC", "Op. Income"),
    ("BOOK_VAL_PER_SH", "BV/Share"),
    # Price
    ("PX_LAST", "Last Price"),
]

# Columns that must be numeric for downstream quant work
NUMERIC_COLUMNS: tuple[str, ...] = (
    "ROIC (%)",
    "FCF Yield (%)",
    "Tot. Assets",
    "EBITDA",
    "Revenue",
    "Op. Income",
    "BV/Share",
    "Last Price",
)

OUTPUT_COLUMNS: list[str] = [spec[1] for spec in FIELD_SPECS]


def get_universe(
    tickers: Optional[Sequence[str]] = None,
    eqs_screen_name: Optional[str] = None,
) -> Any:
    """
    Build a BQL universe from an explicit ticker list or a saved EQS screen name.

    Parameters
    ----------
    tickers:
        Full Bloomberg equity identifiers, e.g. ``['700 HK Equity', '9988 HK Equity']``.
    eqs_screen_name:
        Name of a saved Equity Screening (EQS) screen, if your ``bql.univ`` API
        exposes it (``EqsScreen`` / ``eqs`` / ``eqsscreen``).

    Returns
    -------
    A ``bql`` universe object suitable for ``.for_(universe)`` on a request.

    Raises
    ------
    ValueError
        If both or neither of ``tickers`` and ``eqs_screen_name`` are provided.
    RuntimeError
        If EQS is requested but no supported constructor exists in this environment.
    """
    if (tickers is not None) == (eqs_screen_name is not None):
        raise ValueError("Provide exactly one of tickers=... or eqs_screen_name=...")

    if tickers is not None:
        cleaned = [t.strip() for t in tickers if str(t).strip()]
        if not cleaned:
            raise ValueError("tickers must be a non-empty sequence of strings")
        return bql.univ.list(cleaned)

    univ = bql.univ
    assert eqs_screen_name is not None
    name = eqs_screen_name.strip()
    for attr in ("EqsScreen", "eqsscreen", "eqs", "EQS"):
        ctor = getattr(univ, attr, None)
        if callable(ctor):
            return ctor(name)

    raise RuntimeError(
        "EQS universe is not available: no EqsScreen/eqs constructor on bql.univ. "
        "Use tickers=... or check your BQuant/bql version."
    )


def build_bql_request(
    service: bql.Service,
    universe: Any,
    field_specs: Sequence[tuple[str, str]] = FIELD_SPECS,
) -> Any:
    """
    Assemble one ``get`` request over all Bloomberg items for the given universe.

    The ``get`` call lists raw Bloomberg data items (first element of each tuple)
    so expression evaluation happens on Bloomberg infrastructure before the
    response is serialized back to the notebook or job.
    """
    items = tuple(spec[0] for spec in field_specs)
    # for_(universe) scopes the query; no client-side fan-out over tickers.
    return service.get(*items).for_(universe)


def _response_to_dataframe(response: Any) -> pd.DataFrame:
    """Normalize heterogeneous bql execute outputs to a flat DataFrame."""
    if response is None:
        return pd.DataFrame()

    # Common patterns across BQuant releases
    for attr in ("dataframe", "to_dataframe", "as_dataframe"):
        fn = getattr(response, attr, None)
        if callable(fn):
            df = fn()
            if isinstance(df, pd.DataFrame):
                return df

    composite = getattr(bql, "composite", None)
    if composite is not None:
        for attr in ("to_dataframe", "dataframe"):
            fn = getattr(composite, attr, None)
            if callable(fn):
                try:
                    df = fn(response)
                    if isinstance(df, pd.DataFrame):
                        return df
                except Exception:
                    pass

    # Iterable of series-like or single container with .data
    if hasattr(response, "data") and callable(response.data):
        try:
            payload = response.data()
            if isinstance(payload, pd.DataFrame):
                return payload
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    try:
        for part in response:
            if isinstance(part, pd.DataFrame):
                frames.append(part)
            else:
                for attr in ("dataframe", "to_dataframe", "df"):
                    fn = getattr(part, attr, None)
                    if callable(fn):
                        dfp = fn()
                        if isinstance(dfp, pd.DataFrame):
                            frames.append(dfp)
                            break
    except TypeError:
        pass

    if frames:
        if len(frames) == 1:
            return frames[0]
        # Align on index/columns if multiple blocks returned
        return pd.concat(frames, axis=1)

    raise TypeError(
        "Could not convert BQL response to DataFrame; inspect ``execute`` return "
        "type in your BQuant session and extend _response_to_dataframe."
    )


def fetch_bql_data(
    service: bql.Service,
    request: Any,
    *,
    max_retries: int = 7,
    base_delay_sec: float = 1.0,
    max_delay_sec: float = 25.0,
) -> Any:
    """
    Execute a BQL request with retries for transient timeouts / throttling.

    Parameters
    ----------
    max_retries:
        Total attempts (including the first).
    base_delay_sec:
        Initial backoff; doubled each retry (exponential backoff), capped.
    """
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            return service.execute(request)
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            retryable = any(
                token in msg
                for token in (
                    "timeout",
                    "timed out",
                    "throttl",
                    "rate",
                    "503",
                    "502",
                    "429",
                    "temporar",
                    "try again",
                    "overloaded",
                )
            )
            if attempt >= max_retries - 1 or not retryable:
                logger.exception(
                    "BQL execute failed after %s attempt(s): %s",
                    attempt + 1,
                    exc,
                )
                raise

            delay = min(max_delay_sec, base_delay_sec * (2**attempt))
            logger.warning(
                "BQL execute attempt %s/%s failed (%s); retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error


def clean_data(
    df: pd.DataFrame,
    field_specs: Sequence[tuple[str, str]] = FIELD_SPECS,
    numeric_columns: Iterable[str] = NUMERIC_COLUMNS,
) -> pd.DataFrame:
    """
    Rename BQL columns to Redpoint headers, flatten MultiIndex columns, coerce dtypes.
    """
    if df.empty:
        return pd.DataFrame().reindex(columns=pd.Index(OUTPUT_COLUMNS))

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(level) for level in col if str(level) not in ("", "nan"))
            for col in out.columns.values
        ]

    direct = {spec[0]: spec[1] for spec in field_specs}
    rename_map: dict[Any, str] = {}
    for col in out.columns:
        key = str(col)
        if key in direct:
            rename_map[col] = direct[key]
            continue
        for bql_item, header in field_specs:
            if key == bql_item or key.endswith(f"_{bql_item}") or key.startswith(
                f"{bql_item}_"
            ):
                rename_map[col] = header
                break

    out = out.rename(columns=rename_map)

    for header in OUTPUT_COLUMNS:
        if header not in out.columns:
            out[header] = pd.NA

    out = out.loc[:, OUTPUT_COLUMNS]

    for col in numeric_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ("Ticker", "NAME", "ID_ISIN", "ID_BB_GLOBAL"):
        out[col] = out[col].astype("string")

    for col in (
        "GICS_INDUSTRY_NAME",
        "MARKET_STATUS",
        "PRIMARY_EXCHANGE",
        "STOCK_CONNECT",
    ):
        out[col] = out[col].astype("string")

    return out


# Extension points for beyond phase 1:
# - Universe: pass tickers= or eqs_screen_name= into run_pipeline (see get_universe).
# - Fields: edit FIELD_SPECS / build_bql_request.
# - BQL-side filters: extend build_bql_request per BQuant docs.
# - Post-fetch screening: add a helper and call it after clean_data inside run_pipeline
#   (or on the DataFrame returned by run_pipeline); keep clean_data for shaping only.


def run_pipeline(
    tickers: Optional[Sequence[str]] = None,
    eqs_screen_name: Optional[str] = None,
    service: Optional[bql.Service] = None,
    field_specs: Sequence[tuple[str, str]] = FIELD_SPECS,
) -> pd.DataFrame:
    """
    End-to-end: universe → request → execute → flattened, typed DataFrame.
    """
    svc = service or bql.Service()
    universe = get_universe(tickers=tickers, eqs_screen_name=eqs_screen_name)
    request = build_bql_request(svc, universe, field_specs=field_specs)
    response = fetch_bql_data(svc, request)
    raw = _response_to_dataframe(response)
    return clean_data(raw, field_specs=field_specs)


def main() -> pd.DataFrame:
    """Sample run with liquid Asian listings for manual validation in BQuant."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample = [
        "700 HK Equity",
        "9988 HK Equity",
        "3690 HK Equity",
        "005930 KS Equity",
        "6758 JP Equity",
        "7203 JP Equity",
        "2330 TT Equity",
        "D05 SP Equity",
        "9984 JP Equity",
    ]

    df = run_pipeline(tickers=sample)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
