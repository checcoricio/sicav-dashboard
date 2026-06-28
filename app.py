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
import numpy as np
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
        fetch_fund_sizes,
        fmt_fund_size,
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
DEFAULT_START = datetime.date(2025, 12, 29)
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


def abbreviate_label(name):
    """Abbrevia 'Lux International Strategy' in 'LIS' per etichette più compatte
    (legende, tabelle, selettori), senza alterare i nomi originali usati
    internamente per l'analisi (TICKER_MAP, selezione fondi, export Excel)."""
    if not isinstance(name, str):
        return name
    return name.replace("Lux International Strategy", "LIS")


def get_effective_period(df_cumulative: pd.DataFrame):
    """Restituisce (data_inizio, data_fine) effettive, ricavate dai dati
    realmente disponibili nel grafico (righe non interamente NaN), così che
    il periodo mostrato nel grafico sia sempre coerente con quello riportato
    in tabella, indipendentemente dal range richiesto dall'utente in sidebar."""
    if df_cumulative is None or df_cumulative.empty:
        return None, None
    valid = df_cumulative.dropna(how="all")
    if valid.empty:
        return None, None
    return valid.index.min(), valid.index.max()


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
        format_func=abbreviate_label,
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
) -> plt.Figure:
    """
    Crea il grafico della performance comparativa con uno stile professionale:
    palette colori dedicata, etichette dei valori finali in percentuale,
    griglia leggera sul solo asse Y, assi senza cornice superiore/destra e
    legenda posizionata sotto il grafico per non sovrapporsi alle linee.

    I valori vengono mostrati come rendimento percentuale rispetto alla base
    100 (es. indice a 105.3 → +5.3%), e il periodo indicato nel grafico è
    calcolato sui dati realmente disponibili (coerente con la tabella).
    """
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    n_cols = max(len(df_cumulative.columns), 1)

    for i, col in enumerate(df_cumulative.columns):
        series = df_cumulative[col].dropna()
        if series.empty:
            continue
        pct_series = series - 100.0  # rendimento % rispetto alla base 100
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        ax.plot(
            pct_series.index, pct_series.values,
            label=abbreviate_label(col), linewidth=2.2, color=color, solid_capstyle="round",
        )
        # Pallino + etichetta percentuale sull'ultimo valore di ciascuna serie
        ax.scatter(
            pct_series.index[-1], pct_series.values[-1],
            color=color, s=30, zorder=5, edgecolor="white", linewidth=0.8,
        )
        ax.annotate(
            f"{pct_series.values[-1]:+.1f}%",
            xy=(pct_series.index[-1], pct_series.values[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color=color, va="center",
        )

    ax.axhline(0, color="#9CA3AF", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

    fig.suptitle(
        "Performance comparativa — Rendimento cumulato (%)",
        fontsize=15, fontweight="bold", color="#111827", x=0.02, y=0.975, ha="left",
    )
    eff_start, eff_end = get_effective_period(df_cumulative)
    if eff_start is not None and eff_end is not None:
        fig.text(
            0.02, 0.91, f"Periodo: {fmt_date(eff_start)} → {fmt_date(eff_end)}",
            fontsize=9.5, color="#6B7280", ha="left",
        )

    ax.set_ylabel("Rendimento cumulato (%)", fontsize=10, color="#374151")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
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
# Helper: media ponderata
# ---------------------------------------------------------------------------

def _weighted_avg(df: pd.DataFrame, metric_col: str, weight_col: str) -> float | None:
    """
    Calcola la media ponderata di metric_col usando weight_col come pesi.
    Restituisce None se non ci sono pesi validi.
    """
    try:
        mask = df[weight_col].notna() & df[metric_col].notna()
        if mask.sum() == 0:
            return None
        weights = df.loc[mask, weight_col].astype(float)
        values = df.loc[mask, metric_col].astype(float)
        total_w = weights.sum()
        if total_w == 0:
            return None
        return float((values * weights).sum() / total_w)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rendering tabella summary formattata
# ---------------------------------------------------------------------------

def render_summary_table(df_summary: pd.DataFrame, fund_sizes: dict | None = None):
    if df_summary.empty:
        st.warning("Nessun dato disponibile nel summary.")
        return

    display = df_summary.copy()

    # Aggiunge colonna size numerica (raw float, per ordinamento e media ponderata)
    if fund_sizes:
        display["size_fondo"] = display.index.map(lambda f: fund_sizes.get(f))
    else:
        display["size_fondo"] = None

    # Calcolo media ponderata prima di qualsiasi formattazione
    wp_return = _weighted_avg(display, "total_return", "size_fondo")

    # Ordinamento per size decrescente
    display = display.sort_values("size_fondo", ascending=False, na_position="last")

    # Colonna formattata (solo numero, unità nella testata)
    display["Size (Mln €)"] = display["size_fondo"].apply(fmt_fund_size)

    display.index = display.index.map(abbreviate_label)

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

    # Colonne da mostrare (in ordine)
    cols_show = ["Total Return", "Inizio effettivo", "Fine effettiva",
                 "Osservazioni valide", "Size (Mln €)"]
    cols_show = [c for c in cols_show if c in display.columns]
    st.dataframe(display[cols_show], use_container_width=True)

    # --- Box riepilogativo size + media ponderata ---
    if fund_sizes:
        valid_sizes = [v for v in fund_sizes.values() if v is not None]
        total_size_mln = sum(valid_sizes) / 1_000_000 if valid_sizes else None

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric(
                "💼 Totale AUM fondi analizzati (Mln €)",
                f"{round(total_size_mln):,}".replace(",", ".") if total_size_mln else "N/D",
                help="Somma degli Attivi Netti di tutti i fondi con dati disponibili."
            )
        with col_s2:
            if wp_return is not None:
                st.metric(
                    "⚖️ Media ponderata Total Return (per AUM)",
                    fmt_pct(wp_return),
                    help="Media ponderata del Total Return usando gli Attivi Netti come peso.",
                )
            else:
                st.metric("⚖️ Media ponderata Total Return (per AUM)", "N/D",
                          help="Non calcolabile: dati size non disponibili.")


# ---------------------------------------------------------------------------
# Report immagine A4 orizzontale (grafico + tabella)
# ---------------------------------------------------------------------------

def _build_full_a4_figure(
    df_cumulative: pd.DataFrame,
    df_summary: pd.DataFrame,
    comparison_freq: str,
    fund_sizes: dict | None = None,
) -> plt.Figure:
    """
    Costruisce la figura matplotlib A4 orizzontale (grafico + tabella).
    Usata sia per il PNG sia per il PDF, così i due output sono identici.
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
    eff_start, eff_end = get_effective_period(df_cumulative)
    periodo_txt = f"{fmt_date(eff_start)} → {fmt_date(eff_end)}" if eff_start is not None else "N/D"
    fig.text(
        0.06, 0.935,
        f"Periodo: {periodo_txt}    |    Frequenza: {comparison_freq}    |    "
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
        pct_series = series - 100.0
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        ax_chart.plot(
            pct_series.index, pct_series.values,
            label=abbreviate_label(col), linewidth=2.0, color=color,
        )
        ax_chart.scatter(
            pct_series.index[-1], pct_series.values[-1],
            color=color, s=24, zorder=5, edgecolor="white", linewidth=0.7,
        )
        ax_chart.annotate(
            f"{pct_series.values[-1]:+.1f}%",
            xy=(pct_series.index[-1], pct_series.values[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, fontweight="bold", color=color, va="center",
        )

    ax_chart.axhline(0, color="#9CA3AF", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_chart.set_title("Rendimento cumulato (%)", fontsize=12, fontweight="bold", color="#111827", loc="left")
    ax_chart.set_ylabel("Rendimento cumulato (%)", fontsize=9.5, color="#374151")
    ax_chart.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
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

    # Prepara df con size e ordina per size decrescente
    df_tbl = df_summary.copy().reset_index()
    if fund_sizes:
        df_tbl["_size_raw"] = df_tbl["Fondo"].map(lambda f: fund_sizes.get(f))
    else:
        df_tbl["_size_raw"] = None
    df_tbl = df_tbl.sort_values("_size_raw", ascending=False, na_position="last")

    cols_order = ["Fondo", "total_return", "start_effettiva", "end_effettiva"]
    cols_order = [c for c in cols_order if c in df_tbl.columns]
    table_df = df_tbl[cols_order].copy()

    table_df["Fondo"] = table_df["Fondo"].apply(abbreviate_label)
    table_df["total_return"] = table_df["total_return"].apply(fmt_pct)
    for c in ("start_effettiva", "end_effettiva"):
        if c in table_df.columns:
            table_df[c] = table_df[c].apply(fmt_date)

    # Colonna size: numero arrotondato, unità nella testata
    table_df["Size (Mln €)"] = df_tbl["_size_raw"].apply(fmt_fund_size).values

    table_df = table_df.rename(columns={
        "total_return": "Total Return",
        "start_effettiva": "Inizio",
        "end_effettiva": "Fine",
    })

    # Righe aggregato: Totale AUM e Media ponderata
    extra_rows = []
    if fund_sizes:
        valid_sizes = {k: v for k, v in fund_sizes.items() if v is not None}
        total_size = sum(valid_sizes.values()) if valid_sizes else None
        total_size_fmt = f"{round(total_size / 1_000_000):,}".replace(",", ".") if total_size else "N/D"

        df_for_wp = df_summary.copy()
        df_for_wp["size_fondo"] = df_for_wp.index.map(lambda f: fund_sizes.get(f))
        wp_return = _weighted_avg(df_for_wp, "total_return", "size_fondo")

        blank_row = {c: "" for c in table_df.columns}
        extra_rows = [
            {**blank_row, "Fondo": "▶ TOTALE AUM", "Size (Mln €)": total_size_fmt},
            {**blank_row, "Fondo": "⚖ MEDIA PONDERATA (AUM)",
             "Total Return": fmt_pct(wp_return) if wp_return is not None else "N/D"},
        ]

    if extra_rows:
        table_df = pd.concat([table_df, pd.DataFrame(extra_rows)], ignore_index=True)

    tbl = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0.08, 0.05, 0.84, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)

    n_data_rows = len(table_df) - (2 if extra_rows else 0)
    for (row, _col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if row == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", fontweight="bold")
        elif extra_rows and row == n_data_rows + 1:
            cell.set_facecolor("#1D4ED8")
            cell.set_text_props(color="white", fontweight="bold")
        elif extra_rows and row == n_data_rows + 2:
            cell.set_facecolor("#065F46")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F9FAFB" if row % 2 == 0 else "white")

        fig.text(
            0.07, 0.015,
            "L'AUM corrisponde al 50% del patrimonio liquido di RED",
            fontsize=7, color="#9CA3AF",
            fontstyle="italic",
            fontfamily="DejaVu Sans",
            transform=fig.transFigure,
)

    return fig


def build_a4_report(
    df_cumulative: pd.DataFrame,
    df_summary: pd.DataFrame,
    comparison_freq: str,
    fund_sizes: dict | None = None,
) -> bytes:
    """Restituisce il report A4 orizzontale come PNG bytes."""
    fig = _build_full_a4_figure(df_cumulative, df_summary, comparison_freq, fund_sizes)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Report PDF — stessa figura A4, un foglio solo (matplotlib PdfPages)
# ---------------------------------------------------------------------------

def build_pdf_report(
    df_cumulative: pd.DataFrame,
    df_summary: pd.DataFrame,
    comparison_freq: str,
    fund_sizes: dict | None = None,
) -> bytes:
    """
    Genera il report PDF usando la stessa figura matplotlib del PNG.
    Garantisce una sola pagina, layout identico al PNG, nessuna dipendenza
    da reportlab.
    """
    import matplotlib.backends.backend_pdf as pdf_backend

    fig = _build_full_a4_figure(df_cumulative, df_summary, comparison_freq, fund_sizes)
    buf = BytesIO()
    with pdf_backend.PdfPages(buf) as pdf:
        pdf.savefig(fig, dpi=200, facecolor="white")
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
        "Analisi comparativa dei fondi **EIS Mercurio** e **LIS Metafora**. "
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
        st.markdown("**Fondi:** " + ", ".join(abbreviate_label(f) for f in selected_funds))

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
    # Recupero size fondi (Net Assets)
    # ---------------------------------------------------------------------------
    # Filtra solo i ticker dei fondi selezionati
    selected_ticker_map = {k: v for k, v in TICKER_MAP.items() if v in selected_funds}
    with st.spinner("⏳ Recupero Attivi Netti (Net Assets) da Yahoo Finance..."):
        try:
            fund_sizes = fetch_fund_sizes(selected_ticker_map)
        except Exception as e:
            logger.warning(f"Errore nel recupero Net Assets: {e}")
            fund_sizes = {f: None for f in selected_funds}

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
    render_summary_table(df_summary, fund_sizes=fund_sizes)

    # ---------------------------------------------------------------------------
    # 2. Grafico cumulativo
    # ---------------------------------------------------------------------------
    st.subheader("📈 Rendimento cumulato (%)")

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
            df_cumulative.rename(columns=abbreviate_label).style.format("{:.2f}", na_rep="N/D"),
            use_container_width=True,
        )

        st.markdown("#### NAV allineati alle evaluation dates")
        st.dataframe(
            df_aligned_nav.rename(columns=abbreviate_label).style.format("{:.4f}", na_rep="N/D"),
            use_container_width=True,
        )

    # ---------------------------------------------------------------------------
    # 4. Download Excel + Report immagine A4 + PDF
    # ---------------------------------------------------------------------------
    st.subheader("📥 Export dati")

    col_export1, col_export2, col_export3 = st.columns(3)

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
                    df_cumulative, df_summary, comparison_freq, fund_sizes=fund_sizes,
                )
                st.download_button(
                    label="🖼️ Scarica report PNG — A4 orizzontale",
                    data=report_png,
                    file_name="sicav_report_A4.png",
                    mime="image/png",
                    help="Immagine PNG con grafico e tabella, dimensionata per la stampa in A4 orizzontale.",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"⚠️ Impossibile generare il report immagine: {e}")

    with col_export3:
        if df_cumulative.empty or df_cumulative.dropna(how="all").empty:
            st.info("ℹ️ Report PDF non disponibile: dati cumulativi assenti.")
        else:
            try:
                report_pdf = build_pdf_report(
                    df_cumulative, df_summary, comparison_freq, fund_sizes=fund_sizes,
                )
                st.download_button(
                    label="📄 Scarica report PDF",
                    data=report_pdf,
                    file_name="sicav_report.pdf",
                    mime="application/pdf",
                    help="PDF con grafico, tabella, size fondi, totale AUM e media ponderata.",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"⚠️ Impossibile generare il report PDF: {e}")

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

