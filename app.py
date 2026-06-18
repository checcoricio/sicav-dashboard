"""
app.py
======
Streamlit web app per analisi comparativa fondi SICAV.
Importa analysis.py e costruisce una UI interattiva.

Compatibile con:
  - Esecuzione locale: streamlit run app.py
  - Google Colab: tramite tunnel (vedi guida operativa)

Uso:
    streamlit run app.py
"""

import datetime
import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

# Importa il modulo di analisi (deve essere nella stessa directory)
try:
    from analysis import (
        TICKER_MAP,
        DEFAULT_COMPARISON_FREQ,
        DEFAULT_MAX_GAP_DAYS,
        run_analysis,
        export_excel_bytes,
    )
except ImportError as e:
    st.error(
        f"Impossibile importare analysis.py: {e}\n\n"
        "Assicurati che analysis.py sia nella stessa directory di app.py."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Configurazione pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SICAV Fund Comparator",
    page_icon="📈",
    layout="wide",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti UI
# ---------------------------------------------------------------------------
ALL_FUND_NAMES = list(TICKER_MAP.values())

FREQ_OPTIONS = {
    "W-FRI — Settimanale (venerdì)": "W-FRI",
    "W-MON — Settimanale (lunedì)": "W-MON",
    "D — Giornaliero (business day)": "D",
    "ME — Fine mese": "ME",
}

# Date di default ragionevoli (modificabili dall'utente)
DEFAULT_START = datetime.date(2020, 1, 1)
DEFAULT_END = datetime.date.today()


# ---------------------------------------------------------------------------
# Helper: formattazione
# ---------------------------------------------------------------------------

def fmt_pct(val):
    if pd.isna(val):
        return "N/D"
    return f"{val:.2%}"


def fmt_date(val):
    if pd.isna(val) or val is pd.NaT:
        return "N/D"
    try:
        return pd.Timestamp(val).strftime("%d/%m/%Y")
    except Exception:
        return str(val)


# ---------------------------------------------------------------------------
# Sidebar - Parametri
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("⚙️ Parametri")

    # 1. Selezione fondi
    st.sidebar.subheader("Fondi")
    selected_funds = st.sidebar.multiselect(
        label="Seleziona i fondi da analizzare",
        options=ALL_FUND_NAMES,
        default=ALL_FUND_NAMES,
        help="Seleziona uno o più fondi. Default: tutti.",
    )

    # 2. Date range
    st.sidebar.subheader("Periodo")
    date_range = st.sidebar.date_input(
        label="Intervallo di date",
        value=(DEFAULT_START, DEFAULT_END),
        min_value=datetime.date(2010, 1, 1),
        max_value=datetime.date.today(),
        help="Seleziona data di inizio e fine dell'analisi.",
    )

    # Gestisci il caso in cui l'utente abbia selezionato solo una data
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        # Dato singolo: usa come start, end = oggi
        start_date = date_range if not isinstance(date_range, (list, tuple)) else date_range[0]
        end_date = DEFAULT_END

    # 3. Frequenza comparazione
    st.sidebar.subheader("Frequenza")
    freq_label = st.sidebar.selectbox(
        label="Frequenza di comparazione",
        options=list(FREQ_OPTIONS.keys()),
        index=0,
        help="Frequenza con cui vengono calcolati i rendimenti periodali.",
    )
    comparison_freq = FREQ_OPTIONS[freq_label]

    # 4. Max gap days
    st.sidebar.subheader("Tolleranza gap")
    max_gap_days = st.sidebar.number_input(
        label="Massimo gap giorni (max_gap_days)",
        min_value=1,
        max_value=90,
        value=DEFAULT_MAX_GAP_DAYS,
        step=1,
        help=(
            "Se l'ultimo NAV disponibile è più vecchio di questo numero di giorni "
            "rispetto alla evaluation date, il valore viene impostato a NaN."
        ),
    )

    # 5. Pulsante Esegui
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Esegui analisi", use_container_width=True)

    return selected_funds, start_date, end_date, comparison_freq, int(max_gap_days), run_button


# ---------------------------------------------------------------------------
# Grafico lineare cumulativo
# ---------------------------------------------------------------------------

def plot_cumulative(df_cumulative: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    for col in df_cumulative.columns:
        series = df_cumulative[col].dropna()
        if not series.empty:
            ax.plot(series.index, series.values, label=col, linewidth=1.8)

    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Indice cumulativo (base 100 alla prima data valida)", fontsize=13, pad=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Valore (base 100)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Rendering tabella summary formattata
# ---------------------------------------------------------------------------

def render_summary_table(df_summary: pd.DataFrame):
    if df_summary.empty:
        st.warning("Nessun dato disponibile nel summary.")
        return

    display = df_summary.copy()

    # Formatta colonne
    if "total_return" in display.columns:
        display["total_return"] = display["total_return"].apply(fmt_pct)
    for col in ["start_effettiva", "end_effettiva", "min_date_disponibile", "max_date_disponibile"]:
        if col in display.columns:
            display[col] = display[col].apply(fmt_date)

    # Rinomina colonne per leggibilità
    display = display.rename(columns={
        "total_return": "Total Return",
        "start_effettiva": "Inizio effettivo",
        "end_effettiva": "Fine effettiva",
        "n_obs_valide": "Osservazioni valide",
        "min_date_disponibile": "Prima data disponibile",
        "max_date_disponibile": "Ultima data disponibile",
    })

    st.dataframe(display, use_container_width=True)


# ---------------------------------------------------------------------------
# Pagina principale
# ---------------------------------------------------------------------------

def main():
    # Titolo
    st.title("📈 SICAV Fund Comparator")
    st.markdown(
        "Analisi comparativa dei fondi **EIS Mercurio** e **Lux International Strategy Metafora**. "
        "Dati scaricati in tempo reale da Yahoo Finance."
    )

    # Sidebar
    selected_funds, start_date, end_date, comparison_freq, max_gap_days, run_button = render_sidebar()

    # ---------------------------------------------------------------------------
    # Validazioni input
    # ---------------------------------------------------------------------------
    if not selected_funds:
        st.warning("⚠️ Nessun fondo selezionato. Seleziona almeno un fondo dalla sidebar.")
        st.stop()

    if start_date >= end_date:
        st.warning(
            f"⚠️ La data di inizio ({start_date}) deve essere precedente "
            f"alla data di fine ({end_date})."
        )
        st.stop()

    # Riepilogo parametri
    with st.expander("📋 Riepilogo parametri selezionati", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fondi selezionati", len(selected_funds))
        col2.metric("Inizio", str(start_date))
        col3.metric("Fine", str(end_date))
        col4.metric("Frequenza", comparison_freq)

        st.markdown(f"**Max gap days:** {max_gap_days}")
        st.markdown("**Fondi:** " + ", ".join(selected_funds))

    # ---------------------------------------------------------------------------
    # Esecuzione analisi
    # ---------------------------------------------------------------------------
    if not run_button:
        st.info("👈 Configura i parametri nella sidebar e premi **Esegui analisi**.")
        st.stop()

    with st.spinner("⏳ Download dati da Yahoo Finance e calcolo metriche..."):
        try:
            df_aligned_nav, df_cumulative, df_summary = run_analysis(
                start_date=str(start_date),
                end_date=str(end_date),
                selected_funds=selected_funds,
                comparison_freq=comparison_freq,
                max_gap_days=max_gap_days,
            )
        except Exception as e:
            st.error(f"❌ Errore durante l'analisi: {e}")
            logger.exception("Errore in run_analysis")
            st.stop()

    # ---------------------------------------------------------------------------
    # Gestione risultati vuoti
    # ---------------------------------------------------------------------------
    if df_summary.empty or df_aligned_nav.empty:
        st.error(
            "❌ Nessun dato disponibile per i fondi e il periodo selezionati. "
            "Prova ad allargare il range di date o verificare la connessione a Yahoo Finance."
        )
        st.stop()

    st.success(f"✅ Analisi completata su {len(df_summary)} fondi.")

    # ---------------------------------------------------------------------------
    # 1. Tabella Summary
    # ---------------------------------------------------------------------------
    st.subheader("📊 Riepilogo performance")
    render_summary_table(df_summary)

    # ---------------------------------------------------------------------------
    # 2. Grafico cumulativo
    # ---------------------------------------------------------------------------
    st.subheader("📈 Indice cumulativo (base 100)")

    if df_cumulative.empty or df_cumulative.dropna(how="all").empty:
        st.warning("⚠️ Dati cumulativi non disponibili per il tracciamento del grafico.")
    else:
        fig = plot_cumulative(df_cumulative)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------------------------------------
    # 3. Expander con dati grezzi
    # ---------------------------------------------------------------------------
    with st.expander("🔍 Dati dettagliati", expanded=False):
        st.markdown("#### Indice cumulativo (valori numerici)")
        st.dataframe(
            df_cumulative.style.format("{:.2f}", na_rep="N/D"),
            use_container_width=True,
        )

        st.markdown("#### NAV allineati alle evaluation dates")
        st.dataframe(
            df_aligned_nav.style.format("{:.4f}", na_rep="N/D"),
            use_container_width=True,
        )

    # ---------------------------------------------------------------------------
    # 4. Download Excel
    # ---------------------------------------------------------------------------
    st.subheader("📥 Export dati")

    try:
        excel_bytes = export_excel_bytes(df_aligned_nav, df_cumulative, df_summary)
        st.download_button(
            label="⬇️ Scarica report Excel",
            data=excel_bytes,
            file_name="sicav_comparison_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Scarica il file Excel con i fogli: aligned_nav, cumulative, summary.",
        )
    except Exception as e:
        st.warning(f"⚠️ Impossibile generare il file Excel: {e}")

    # ---------------------------------------------------------------------------
    # Footer
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Dati forniti da Yahoo Finance tramite la libreria yfinance. "
        "Le performance passate non sono indicative di quelle future. "
        "Strumento solo a scopo informativo."
    )


if __name__ == "__main__":
    main()
