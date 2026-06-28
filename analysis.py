"""
analysis.py
===========
Modulo di analisi per fondi SICAV (EIS Mercurio / Lux International Strategy).
Scarica i dati di prezzo/NAV da Yahoo Finance tramite yfinance.

Autore: Senior Python Quant/Data Engineer
Compatibile con: Google Colab, ambiente locale, Streamlit Cloud

NOTA: Se Yahoo Finance non restituisce la colonna 'Adj Close', il codice
      utilizza automaticamente 'Close' come fallback e lo segnala nel log.
"""

import logging
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
TICKER_MAP: dict[str, str] = {
    "0P0001NCDG.F": "Eurizon (Intesa)",
    "0P0001NCDW.F": "Goldman Sachs",
    "0P0001NCDY.F": "Citibank",
    "0P0001KEAL.F": "UniCredit",
    "0P0001OSNN.F": "Cesare Ponti (BPER)",
    "0P0001B56X.F": "Indosuez",
    "0P0001B56Y.F": "JPM",
}

REVERSE_TICKER_MAP: dict[str, str] = {v: k for k, v in TICKER_MAP.items()}

DEFAULT_COMPARISON_FREQ = "W-FRI"
DEFAULT_MAX_GAP_DAYS = 20


# ---------------------------------------------------------------------------
# 0. Recupero Net Assets (Size fondo) da Yahoo Finance
# ---------------------------------------------------------------------------

def fetch_fund_sizes(
    ticker_map: dict[str, str] | None = None,
) -> dict[str, float | None]:
    """
    Recupera gli Attivi Netti (Net Assets) di ciascun fondo da Yahoo Finance.

    Parameters
    ----------
    ticker_map : dict, opzionale
        Mapping {ticker: nome_leggibile}. Se None, usa TICKER_MAP globale.

    Returns
    -------
    dict
        Mapping {nome_leggibile: net_assets_float_or_None}
        Il valore è in valuta originale (generalmente EUR).
        None se il dato non è disponibile.
    """
    if ticker_map is None:
        ticker_map = TICKER_MAP

    sizes: dict[str, float | None] = {}
    for ticker, name in ticker_map.items():
        try:
            info = yf.Ticker(ticker).info
            net_assets = info.get("totalAssets") or info.get("netAssets") or None
            if net_assets is not None:
                net_assets = float(net_assets)
            sizes[name] = net_assets
            logger.info(f"[{name}] Net Assets: {net_assets}")
        except Exception as e:
            logger.warning(f"[{name}] Impossibile recuperare Net Assets: {e}")
            sizes[name] = None

    return sizes


def fmt_fund_size(value):
    """
    Formatta il valore degli attivi netti come numero arrotondato in Mln €.
    L'unità (Mln €) viene mostrata solo nell'intestazione di colonna, non qui.
    Es. 1_234_567_890 → '1.234,6'
        123_456_789  → '123,5'
        None         → 'N/D'
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/D"
    try:
        v = round(float(value) / 1_000_000)   # arrotonda all'intero più vicino
        return str(v)                           # nessun separatore, nessun decimale
    except Exception:
        return "N/D"


# ---------------------------------------------------------------------------
# 1. Download dati da Yahoo Finance
# ---------------------------------------------------------------------------

def download_data(
    ticker_map: dict[str, str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Scarica lo storico prezzi/NAV da Yahoo Finance per tutti i ticker indicati.

    Parameters
    ----------
    ticker_map : dict
        Mapping {ticker: nome_leggibile}. Se None, usa TICKER_MAP globale.
    start_date : str, opzionale
        Data di inizio nel formato 'YYYY-MM-DD'. Se None, scarica tutto lo storico.
    end_date : str, opzionale
        Data di fine nel formato 'YYYY-MM-DD'. Se None, usa oggi.

    Returns
    -------
    pd.DataFrame
        DataFrame con index=Date (DatetimeIndex) e columns=nomi leggibili dei fondi.
        Valori = Adj Close (oppure Close se Adj Close non disponibile).
    """
    if ticker_map is None:
        ticker_map = TICKER_MAP

    tickers = list(ticker_map.keys())
    logger.info(f"Scaricamento dati per {len(tickers)} ticker da Yahoo Finance...")
    logger.info(f"  Ticker: {tickers}")
    logger.info(f"  start_date={start_date}, end_date={end_date}")

    # Scarichiamo tutto in un solo batch con yfinance
    # group_by='ticker' ci permette di gestire bene il MultiIndex
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,   # Manteniamo separati Adj Close e Close
            progress=False,
            threads=True,
        )

    if raw.empty:
        logger.error("yfinance non ha restituito alcun dato. Verificare connessione/ticker.")
        return pd.DataFrame()

    logger.info(f"Download completato. Shape raw: {raw.shape}")

    # --- Gestione MultiIndex columns -----------------------------------------
    # Con più ticker, yfinance restituisce MultiIndex: (price_type, ticker)
    # Con un solo ticker restituisce colonne semplici.
    frames: dict[str, pd.Series] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        # Caso normale: più ticker -> MultiIndex (price_type, ticker)
        price_types = raw.columns.get_level_values(0).unique().tolist()
        logger.info(f"Colonne di prezzo disponibili: {price_types}")

        for ticker, name in ticker_map.items():
            try:
                if "Adj Close" in price_types:
                    series = raw["Adj Close"][ticker].copy()
                    price_col_used = "Adj Close"
                elif "Close" in price_types:
                    series = raw["Close"][ticker].copy()
                    price_col_used = "Close"
                    logger.warning(
                        f"[{name}] 'Adj Close' non disponibile: uso 'Close' come fallback."
                    )
                else:
                    logger.warning(
                        f"[{name}] Nessuna colonna di prezzo trovata. Ticker escluso."
                    )
                    continue

                # Rimuoviamo righe completamente NaN
                series = series.dropna()

                if series.empty:
                    logger.warning(f"[{name}] Serie vuota dopo drop NaN. Ticker escluso.")
                    continue

                # Statistiche di download
                logger.info(
                    f"[{name}] "
                    f"Prima data: {series.index.min().date()} | "
                    f"Ultima data: {series.index.max().date()} | "
                    f"Osservazioni: {len(series)} | "
                    f"Colonna usata: {price_col_used}"
                )

                frames[name] = series

            except KeyError:
                logger.warning(
                    f"[{name}] Ticker '{ticker}' non trovato nel DataFrame scaricato. "
                    f"Probabilmente Yahoo Finance non ha dati per questo ticker."
                )
            except Exception as e:
                logger.warning(f"[{name}] Errore durante estrazione dati: {e}. Ticker escluso.")

    else:
        # Caso con un solo ticker: colonne semplici
        ticker = tickers[0]
        name = ticker_map[ticker]
        try:
            if "Adj Close" in raw.columns:
                series = raw["Adj Close"].copy()
                price_col_used = "Adj Close"
            elif "Close" in raw.columns:
                series = raw["Close"].copy()
                price_col_used = "Close"
                logger.warning(
                    f"[{name}] 'Adj Close' non disponibile: uso 'Close' come fallback."
                )
            else:
                logger.error(f"[{name}] Nessuna colonna di prezzo trovata.")
                return pd.DataFrame()

            series = series.dropna()
            if not series.empty:
                logger.info(
                    f"[{name}] Prima data: {series.index.min().date()} | "
                    f"Ultima data: {series.index.max().date()} | "
                    f"Osservazioni: {len(series)} | "
                    f"Colonna usata: {price_col_used}"
                )
                frames[name] = series
            else:
                logger.warning(f"[{name}] Serie vuota. Ticker escluso.")
        except Exception as e:
            logger.error(f"[{name}] Errore: {e}")

    if not frames:
        logger.error("Nessun dato valido scaricato per nessun ticker.")
        return pd.DataFrame()

    # Costruisci DataFrame unificato
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    logger.info(
        f"DataFrame finale: {df.shape[0]} righe x {df.shape[1]} fondi. "
        f"Range date: {df.index.min().date()} -> {df.index.max().date()}"
    )

    # Segnala eventuali missing totali
    missing_tickers = [
        name for name in ticker_map.values() if name not in df.columns
    ]
    if missing_tickers:
        logger.warning(
            f"Fondi con ZERO dati disponibili (esclusi dall'analisi): {missing_tickers}"
        )

    return df


def load_data(
    ticker_map: dict[str, str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Wrapper pubblico di download_data. Scarica e restituisce il DataFrame prezzi/NAV.

    È possibile estendere questa funzione per supportare cache locale
    (es. Parquet) senza modificare il resto del codice.
    """
    return download_data(ticker_map=ticker_map, start_date=start_date, end_date=end_date)


# ---------------------------------------------------------------------------
# 2. Calendario di evaluation dates
# ---------------------------------------------------------------------------

def make_calendar(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    comparison_freq: str = DEFAULT_COMPARISON_FREQ,
) -> pd.DatetimeIndex:
    """
    Crea una griglia di evaluation_dates all'interno di [start_date, end_date].

    Parameters
    ----------
    start_date : str o Timestamp
    end_date   : str o Timestamp
    comparison_freq : str
        Frequenza pandas. Esempi:
        - "D"     : ogni giorno lavorativo (Business Day)
        - "W-FRI" : ogni venerdì (fine settimana)
        - "ME"    : ultimo giorno del mese

    Returns
    -------
    pd.DatetimeIndex
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if start_date >= end_date:
        raise ValueError(f"start_date ({start_date.date()}) >= end_date ({end_date.date()})")

    # "M" è deprecato in pandas >= 2.2, usiamo "ME"
    freq_map = {"M": "ME", "BM": "BME"}
    freq = freq_map.get(comparison_freq, comparison_freq)

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Filtriamo per essere sicuri di stare nel range
    dates = dates[(dates >= start_date) & (dates <= end_date)]

    if len(dates) == 0:
        logger.warning(
            f"make_calendar: nessuna evaluation_date generata con freq='{comparison_freq}' "
            f"nel range [{start_date.date()}, {end_date.date()}]. "
            f"Provo ad aggiungere end_date come unica data."
        )
        dates = pd.DatetimeIndex([end_date])

    logger.info(
        f"Calendario: {len(dates)} date con freq='{comparison_freq}' "
        f"da {dates[0].date()} a {dates[-1].date()}"
    )
    return dates


# ---------------------------------------------------------------------------
# 3. Allineamento ASOF
# ---------------------------------------------------------------------------

def align_asof(
    df_prices: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> pd.DataFrame:
    """
    Allinea i prezzi/NAV alle evaluation_dates con regola ASOF.

    Per ogni data t in evaluation_dates:
        - usa l'ultimo NAV disponibile con data <= t
        - se (t - last_nav_date) > max_gap_days, assegna NaN

    Parameters
    ----------
    df_prices : pd.DataFrame
        Prezzi grezzi con DatetimeIndex e colonne = nomi fondi.
    evaluation_dates : pd.DatetimeIndex
        Date di valutazione target.
    max_gap_days : int
        Numero massimo di giorni di gap tollerato.

    Returns
    -------
    pd.DataFrame
        DataFrame allineato con index=evaluation_dates, colonne=fondi.
    """
    if df_prices.empty:
        logger.error("align_asof: df_prices è vuoto.")
        return pd.DataFrame(index=evaluation_dates)

    results: dict[str, pd.Series] = {}

    for col in df_prices.columns:
        series = df_prices[col].dropna().sort_index()

        if series.empty:
            logger.warning(f"[{col}] Nessun dato disponibile per l'allineamento.")
            results[col] = pd.Series(np.nan, index=evaluation_dates)
            continue

        aligned_values = []
        for t in evaluation_dates:
            # Trova l'ultimo NAV disponibile con data <= t
            candidates = series[series.index <= t]
            if candidates.empty:
                aligned_values.append(np.nan)
                continue

            last_date = candidates.index[-1]
            gap = (t - last_date).days

            if gap > max_gap_days:
                # Gap troppo grande: dati non affidabili o fondo sospeso
                aligned_values.append(np.nan)
            else:
                aligned_values.append(candidates.iloc[-1])

        results[col] = pd.Series(aligned_values, index=evaluation_dates)

    df_aligned = pd.DataFrame(results)
    df_aligned.index.name = "Date"

    # Log missing per colonna
    for col in df_aligned.columns:
        n_nan = df_aligned[col].isna().sum()
        n_valid = df_aligned[col].notna().sum()
        if n_nan > 0:
            logger.info(
                f"[{col}] Allineamento: {n_valid} valori validi, "
                f"{n_nan} NaN (gap > {max_gap_days}gg o dati mancanti)"
            )

    return df_aligned


# ---------------------------------------------------------------------------
# 4. Calcolo metriche
# ---------------------------------------------------------------------------

def compute_metrics(df_aligned_nav: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcola rendimenti periodali, indice cumulativo base 100 e summary.

    Parameters
    ----------
    df_aligned_nav : pd.DataFrame
        NAV allineati su evaluation_dates (output di align_asof).

    Returns
    -------
    df_cumulative : pd.DataFrame
        Indice cumulativo base 100 (prima data valida = 100).
    df_summary : pd.DataFrame
        Riepilogo per fondo con: total_return, start_effettiva, end_effettiva,
        n_obs_valide, min_date_disponibile, max_date_disponibile.
    """
    if df_aligned_nav.empty:
        logger.error("compute_metrics: df_aligned_nav è vuoto.")
        return pd.DataFrame(), pd.DataFrame()

    # Rendimenti periodali
    df_returns = df_aligned_nav.pct_change()

    # Indice cumulativo base 100 per ciascun fondo
    # Base 100 alla prima data con NAV valido
    cumulative_dict: dict[str, pd.Series] = {}
    summary_rows = []

    for col in df_aligned_nav.columns:
        series = df_aligned_nav[col].dropna()

        if series.empty:
            logger.warning(f"[{col}] Nessun dato valido per il calcolo cumulativo.")
            cumulative_dict[col] = pd.Series(np.nan, index=df_aligned_nav.index)
            summary_rows.append({
                "Fondo": col,
                "total_return": np.nan,
                "start_effettiva": pd.NaT,
                "end_effettiva": pd.NaT,
                "n_obs_valide": 0,
                "min_date_disponibile": pd.NaT,
                "max_date_disponibile": pd.NaT,
            })
            continue

        start_eff = series.index[0]
        end_eff = series.index[-1]
        nav_start = series.iloc[0]
        nav_end = series.iloc[-1]

        if nav_start == 0 or np.isnan(nav_start):
            logger.warning(f"[{col}] NAV iniziale zero o NaN, impossibile calcolare total_return.")
            total_return = np.nan
        else:
            total_return = (nav_end / nav_start) - 1

        # Indice base 100
        aligned_col = df_aligned_nav[col]
        # Normalizza rispetto al primo valore valido
        first_valid_val = aligned_col.dropna().iloc[0]
        cumul = (aligned_col / first_valid_val) * 100.0
        cumulative_dict[col] = cumul

        logger.info(
            f"[{col}] Total Return: {total_return:.2%} | "
            f"Da {start_eff.date()} a {end_eff.date()} | "
            f"Obs valide: {len(series)}"
        )

        summary_rows.append({
            "Fondo": col,
            "total_return": total_return,
            "start_effettiva": start_eff,
            "end_effettiva": end_eff,
            "n_obs_valide": len(series),
            "min_date_disponibile": series.index.min(),
            "max_date_disponibile": series.index.max(),
        })

    df_cumulative = pd.DataFrame(cumulative_dict)
    df_cumulative.index = df_aligned_nav.index
    df_cumulative.index.name = "Date"

    df_summary = pd.DataFrame(summary_rows).set_index("Fondo")

    return df_cumulative, df_summary


# ---------------------------------------------------------------------------
# 5. Export Excel
# ---------------------------------------------------------------------------

def export_excel(
    df_aligned_nav: pd.DataFrame,
    df_cumulative: pd.DataFrame,
    df_summary: pd.DataFrame,
    output_path: str = "sicav_comparison_output.xlsx",
) -> None:
    """
    Esporta i tre DataFrame in un file Excel con fogli separati.

    Fogli creati:
        - aligned_nav   : NAV allineati alle evaluation_dates
        - cumulative    : Indice cumulativo base 100
        - summary       : Riepilogo metriche per fondo

    Parameters
    ----------
    df_aligned_nav : pd.DataFrame
    df_cumulative  : pd.DataFrame
    df_summary     : pd.DataFrame
    output_path    : str
        Percorso del file Excel di output.
    """
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_aligned_nav.to_excel(writer, sheet_name="aligned_nav")
            df_cumulative.to_excel(writer, sheet_name="cumulative")
            df_summary.to_excel(writer, sheet_name="summary")
        logger.info(f"Export Excel completato: {output_path}")
    except Exception as e:
        logger.error(f"Errore durante export Excel: {e}")
        raise


def export_excel_bytes(
    df_aligned_nav: pd.DataFrame,
    df_cumulative: pd.DataFrame,
    df_summary: pd.DataFrame,
) -> bytes:
    """
    Come export_excel, ma restituisce i bytes del file Excel (utile per Streamlit download).
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_aligned_nav.to_excel(writer, sheet_name="aligned_nav")
        df_cumulative.to_excel(writer, sheet_name="cumulative")
        df_summary.to_excel(writer, sheet_name="summary")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# 6. Pipeline completa
# ---------------------------------------------------------------------------

def run_analysis(
    ticker_map: dict[str, str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    selected_funds: list[str] | None = None,
    comparison_freq: str = DEFAULT_COMPARISON_FREQ,
    max_gap_days: int = DEFAULT_MAX_GAP_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completa: download -> allineamento -> metriche.

    Parameters
    ----------
    ticker_map     : dict, opzionale. Default = TICKER_MAP globale.
    start_date     : str YYYY-MM-DD, opzionale.
    end_date       : str YYYY-MM-DD, opzionale.
    selected_funds : list of str, opzionale. Sottoinsieme dei nomi leggibili dei fondi.
    comparison_freq: str, frequenza pandas.
    max_gap_days   : int.

    Returns
    -------
    (df_aligned_nav, df_cumulative, df_summary)
    """
    if ticker_map is None:
        ticker_map = TICKER_MAP

    # --- 1. Download --------------------------------------------------------
    # Per l'allineamento ASOF potremmo aver bisogno di dati precedenti a start_date.
    # Scarichiamo da (start_date - 60 giorni) per avere buffer sufficiente.
    download_start = None
    if start_date is not None:
        _sd = pd.Timestamp(start_date)
        download_start = (_sd - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        logger.info(
            f"Download range esteso a {download_start} (start_date - 60gg) "
            f"per garantire allineamento ASOF corretto."
        )

    df_prices = load_data(
        ticker_map=ticker_map,
        start_date=download_start,
        end_date=end_date,
    )

    if df_prices.empty:
        logger.error("Nessun dato scaricato. Analisi interrotta.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- 2. Filtra fondi selezionati ----------------------------------------
    if selected_funds is not None:
        available = [f for f in selected_funds if f in df_prices.columns]
        missing = [f for f in selected_funds if f not in df_prices.columns]
        if missing:
            logger.warning(f"Fondi richiesti ma non disponibili nei dati: {missing}")
        if not available:
            logger.error("Nessun fondo selezionato è disponibile nei dati scaricati.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df_prices = df_prices[available]

    # --- 3. Determina date default se non fornite ---------------------------
    global_min = df_prices.index.min()
    global_max = df_prices.index.max()

    if start_date is None:
        start_date = global_min.strftime("%Y-%m-%d")
        logger.info(f"start_date non fornita: uso la prima data disponibile = {start_date}")

    if end_date is None:
        end_date = global_max.strftime("%Y-%m-%d")
        logger.info(f"end_date non fornita: uso l'ultima data disponibile = {end_date}")

    # --- 4. Calendario evaluation_dates ------------------------------------
    eval_dates = make_calendar(start_date, end_date, comparison_freq)

    # --- 5. Allineamento ASOF ----------------------------------------------
    df_aligned_nav = align_asof(df_prices, eval_dates, max_gap_days)

    # --- 6. Metriche --------------------------------------------------------
    df_cumulative, df_summary = compute_metrics(df_aligned_nav)

    return df_aligned_nav, df_cumulative, df_summary


# ---------------------------------------------------------------------------
# 7. main() di esempio
# ---------------------------------------------------------------------------

def main():
    """
    Esempio di utilizzo del modulo.
    Eseguibile sia in Colab sia in ambiente locale con:
        python analysis.py
    """
    logger.info("=" * 60)
    logger.info("ANALISI FONDI SICAV - avvio pipeline di esempio")
    logger.info("=" * 60)

    # Parametri di esempio
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    comparison_freq = "W-FRI"
    max_gap_days = 20

    logger.info(f"Parametri: start={start_date}, end={end_date}, "
                f"freq={comparison_freq}, max_gap={max_gap_days}")

    df_aligned_nav, df_cumulative, df_summary = run_analysis(
        start_date=start_date,
        end_date=end_date,
        comparison_freq=comparison_freq,
        max_gap_days=max_gap_days,
    )

    if df_summary.empty:
        logger.error("Analisi fallita: nessun dato disponibile.")
        return

    logger.info("\n--- SUMMARY FINALE ---")
    print(df_summary.to_string())

    # Export Excel
    output_path = "sicav_comparison_output.xlsx"
    try:
        export_excel(df_aligned_nav, df_cumulative, df_summary, output_path)
    except Exception as e:
        logger.warning(f"Export Excel non riuscito: {e}")

    logger.info("Pipeline completata.")
    return df_aligned_nav, df_cumulative, df_summary


if __name__ == "__main__":
    main()
