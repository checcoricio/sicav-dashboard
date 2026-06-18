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
from io import BytesIO

import matplotlib.dates as mdates
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

# Palette colori professionale, leggibile anche per daltonici, usata nei grafici
CHART_PALETTE = [
    "#2563EB",  # blu
    "#DC2626",  # rosso
    "#059669",  # verde
    "#D97706",  # ambra
    "#7C3AED",  # viola
    "#0891B2",  # ciano
    "#BE185D",  # magenta
    "#65A30D",  # lime
]

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

def plot_cumulative(
    df_cumulative: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> plt.Figure:
    """
    Crea il grafico dell'indice cumulativo con uno stile professionale:
    palette colori dedicata, etichette dei valori finali, griglia leggera
    sul solo asse Y, assi senza cornice superiore/destra e legenda
    posizionata sotto il grafico per non sovrapporsi alle linee.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    n_cols = max(len(df_cumulative.columns), 1)

    for i, col in enumerate(df_cumulative.columns):
        series = df_cumulative[col].dropna()
        if series.empty:
            continue
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        ax.plot(
            series.index, series.values,
            label=col, linewidth=2.2, color=color, solid_capstyle="round",
        )
        # Pallino + etichetta numerica sull'ultimo valore di ciascuna serie
        ax.scatter(
            series.index[-1], series.values[-1],
            color=color, s=30, zorder=5, edgecolor="white", linewidth=0.8,
        )
        ax.annotate(
            f"{series.values[-1]:.1f}",
            xy=(series.index[-1], series.values[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color=color, va="center",
        )

    ax.axhline(100, color="#9CA3AF", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

    fig.suptitle(
        "Performance comparativa — Indice cumulativo (base 100)",
        fontsize=15, fontweight="bold", color="#111827", x=0.02, y=0.975, ha="left",
    )
    if start_date and end_date:
        fig.text(
            0.02, 0.91, f"Periodo: {start_date} → {end_date}",
            fontsize=9.5, color="#6B7280", ha="left",
        )

    ax.set_ylabel("Indice (base 100)", fontsize=10, color="#374151")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#D1D5DB")

    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.35, color="#9CA3AF")
    ax.tick_params(axis="both", labelsize=9, colors="#374151")

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=min(4, n_cols), fontsize=8.5, frameon=False,
    )

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.82)
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
# Report immagine A4 orizzontale (grafico + tabella)
# ---------------------------------------------------------------------------

def build_a4_report(
    df_cumulative: pd.DataFrame,
    df_summary: pd.DataFrame,
    start_date: str,
    end_date: str,
    comparison_freq: str,
) -> bytes:
    """
    Costruisce un'unica immagine PNG (grafico in alto, tabella riepilogativa
    in basso) dimensionata per il formato A4 orizzontale (29,7 x 21,0 cm),
    pronta per la stampa o la condivisione.

    Returns
    -------
    bytes
        Contenuto PNG dell'immagine generata.
    """
    # A4 orizzontale: 29.7 x 21.0 cm = 11.69 x 8.27 pollici
    fig = plt.figure(figsize=(11.69, 8.27), dpi=200)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        2, 1, height_ratios=[2.1, 1],
        hspace=0.40, top=0.89, bottom=0.07, left=0.06, right=0.96,
    )

    # --- Intestazione -------------------------------------------------------
    fig.text(0.06, 0.965, "SICAV Fund Comparator — Report", fontsize=18, fontweight="bold", color="#111827")
    fig.text(
        0.06, 0.935,
        f"Periodo: {start_date} → {end_date}    |    Frequenza: {comparison_freq}    |    "
        f"Generato il: {datetime.date.today().strftime('%d/%m/%Y')}",
        fontsize=10, color="#6B7280",
    )

    # --- Grafico --------------------------------------------------------------
    ax_chart = fig.add_subplot(gs[0])
    n_cols = max(len(df_cumulative.columns), 1)

    for i, col in enumerate(df_cumulative.columns):
        series = df_cumulative[col].dropna()
        if series.empty:
            continue
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        ax_chart.plot(series.index, series.values, label=col, linewidth=2.0, color=color)
        ax_chart.scatter(
            series.index[-1], series.values[-1],
            color=color, s=24, zorder=5, edgecolor="white", linewidth=0.7,
        )

    ax_chart.axhline(100, color="#9CA3AF", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_chart.set_title("Indice cumulativo (base 100)", fontsize=12, fontweight="bold", color="#111827", loc="left")
    ax_chart.set_ylabel("Indice (base 100)", fontsize=9.5, color="#374151")
    ax_chart.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_chart.tick_params(labelsize=8.5, colors="#374151")
    for spine in ("top", "right"):
        ax_chart.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax_chart.spines[spine].set_color("#D1D5DB")
    ax_chart.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.3, color="#9CA3AF")
    ax_chart.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=min(4, n_cols), fontsize=8, frameon=False,
    )

    # --- Tabella riepilogativa ------------------------------------------------
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    table_df = df_summary.copy().reset_index()
    cols_order = ["Fondo", "total_return", "start_effettiva", "end_effettiva", "n_obs_valide"]
    cols_order = [c for c in cols_order if c in table_df.columns]
    table_df = table_df[cols_order]

    if "total_return" in table_df.columns:
        table_df["total_return"] = table_df["total_return"].apply(fmt_pct)
    for c in ("start_effettiva", "end_effettiva"):
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(fmt_date)

    table_df = table_df.rename(columns={
        "total_return": "Total Return",
        "start_effettiva": "Inizio",
        "end_effettiva": "Fine",
        "n_obs_valide": "Osservazioni",
    })

    tbl = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0.08, 0.05, 0.84, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    for (row, _col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if row == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F9FAFB" if row % 2 == 0 else "white")

    # --- Footer -----------------------------------------------------------
    fig.text(
        0.06, 0.015,
        "Dati: Yahoo Finance (yfinance). Le performance passate non sono indicative di quelle future. "
        "Strumento solo a scopo informativo.",
        fontsize=7.5, color="#9CA3AF",
    )

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


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
        fig = plot_cumulative(df_cumulative, start_date=str(start_date), end_date=str(end_date))
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
    # 4. Download Excel + Report immagine A4
    # ---------------------------------------------------------------------------
    st.subheader("📥 Export dati")

    col_export1, col_export2 = st.columns(2)

    with col_export1:
        try:
            excel_bytes = export_excel_bytes(df_aligned_nav, df_cumulative, df_summary)
            st.download_button(
                label="⬇️ Scarica report Excel",
                data=excel_bytes,
                file_name="sicav_comparison_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Scarica il file Excel con i fogli: aligned_nav, cumulative, summary.",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"⚠️ Impossibile generare il file Excel: {e}")

    with col_export2:
        if df_cumulative.empty or df_cumulative.dropna(how="all").empty:
            st.info("ℹ️ Report immagine non disponibile: dati cumulativi assenti.")
        else:
            try:
                report_png = build_a4_report(
                    df_cumulative, df_summary,
                    str(start_date), str(end_date), comparison_freq,
                )
                st.download_button(
                    label="🖼️ Scarica report (grafico + tabella) — A4 orizzontale",
                    data=report_png,
                    file_name="sicav_report_A4.png",
                    mime="image/png",
                    help="Immagine PNG con grafico e tabella, dimensionata per la stampa in A4 orizzontale.",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"⚠️ Impossibile generare il report immagine: {e}")

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
