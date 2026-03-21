"""
╔══════════════════════════════════════════════════════════════╗
║         EDA PROFILER  —  HTML Report Generator               ║
║  Genera un reporte HTML interactivo similar a ydata-profiling ║
╚══════════════════════════════════════════════════════════════╝

USO:
    from eda_profiler import EDAProfiler
    profiler = EDAProfiler(df, title="Mi Dataset")
    profiler.to_file("reporte.html")          # guarda HTML
    profiler.to_notebook_iframe()             # muestra en Jupyter

TAMBIÉN:
    from eda_profiler import profile_report
    profile_report(df, "reporte.html", title="Mi Dataset")
"""

import base64
import io
import json
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paleta visual ──────────────────────────────────────────────────────────────
C = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "card":    "#1c2128",
    "border":  "#30363d",
    "accent":  "#7c3aed",
    "accent2": "#06b6d4",
    "accent3": "#10b981",
    "warn":    "#f59e0b",
    "danger":  "#ef4444",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "plot_bg": "#0d1117",
}
PALETTE = ["#7c3aed","#06b6d4","#10b981","#f59e0b","#ef4444","#ec4899","#84cc16","#f97316"]

plt.rcParams.update({
    "figure.facecolor":  C["plot_bg"],
    "axes.facecolor":    C["card"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["text"],
    "xtick.color":       C["muted"],
    "ytick.color":       C["muted"],
    "text.color":        C["text"],
    "grid.color":        C["border"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig, dpi=120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def _iqr_outliers(s: pd.Series) -> int:
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS POR TIPO DE COLUMNA
# ══════════════════════════════════════════════════════════════════════════════

def _analyze_numeric(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {}
    _, p_shapiro = stats.shapiro(s.sample(min(5000, len(s)), random_state=42))
    return {
        "n":          len(s),
        "mean":       s.mean(),
        "std":        s.std(),
        "min":        s.min(),
        "p5":         s.quantile(.05),
        "p25":        s.quantile(.25),
        "median":     s.median(),
        "p75":        s.quantile(.75),
        "p95":        s.quantile(.95),
        "max":        s.max(),
        "range":      s.max() - s.min(),
        "iqr":        s.quantile(.75) - s.quantile(.25),
        "skewness":   s.skew(),
        "kurtosis":   s.kurtosis(),
        "zeros":      int((s == 0).sum()),
        "zeros_pct":  (s == 0).mean() * 100,
        "negatives":  int((s < 0).sum()),
        "outliers":   _iqr_outliers(s),
        "outliers_pct": _iqr_outliers(s) / len(s) * 100,
        "is_normal":  p_shapiro > 0.05,
        "p_shapiro":  p_shapiro,
    }

def _analyze_categorical(series: pd.Series) -> dict:
    s = series.dropna()
    vc = series.value_counts()
    return {
        "n":           len(s),
        "unique":      series.nunique(),
        "unique_pct":  series.nunique() / len(series) * 100,
        "top":         vc.index[0] if len(vc) else None,
        "top_freq":    int(vc.iloc[0]) if len(vc) else 0,
        "top_pct":     vc.iloc[0] / len(series) * 100 if len(vc) else 0,
        "value_counts": vc.head(20).to_dict(),
    }

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

def _plot_numeric_col(series: pd.Series) -> str:
    s = series.dropna()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), facecolor=C["plot_bg"])
    fig.subplots_adjust(wspace=0.35)

    # Histograma + KDE
    ax = axes[0]
    ax.hist(s, bins=40, color=C["accent"], alpha=0.7, density=True, label="Histograma")
    try:
        kde = stats.gaussian_kde(s)
        x = np.linspace(s.min(), s.max(), 300)
        ax.plot(x, kde(x), color=C["accent2"], lw=2, label="KDE")
    except Exception:
        pass
    ax.set_title("Distribución", color=C["text"], fontsize=10)
    ax.grid(True); ax.legend(fontsize=8)

    # Boxplot
    ax = axes[1]
    bp = ax.boxplot(s, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=C["accent"], alpha=0.5),
                    medianprops=dict(color=C["accent3"], lw=2),
                    whiskerprops=dict(color=C["muted"]),
                    capprops=dict(color=C["muted"]),
                    flierprops=dict(marker="o", color=C["danger"], alpha=0.4, markersize=3))
    ax.set_title("Boxplot", color=C["text"], fontsize=10)
    ax.grid(True)

    # Q-Q
    ax = axes[2]
    (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
    ax.scatter(osm, osr, color=C["accent"], s=8, alpha=0.5)
    line_x = np.array([min(osm), max(osm)])
    ax.plot(line_x, slope * line_x + intercept, color=C["accent3"], lw=1.5)
    ax.set_title(f"Q-Q Plot  (r={r:.3f})", color=C["text"], fontsize=10)
    ax.grid(True)

    return _fig_to_b64(fig)

def _plot_categorical_col(series: pd.Series) -> str:
    vc = series.value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, max(3, len(vc)*0.45)), facecolor=C["plot_bg"])
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(vc))]
    bars = ax.barh(range(len(vc)), vc.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(vc)))
    ax.set_yticklabels([str(v)[:30] for v in vc.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Frecuencia (top 15)", color=C["text"], fontsize=10)
    ax.grid(True, axis="x")
    for bar, val in zip(bars, vc.values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:,}  ({val/len(series)*100:.1f}%)",
                va="center", fontsize=8, color=C["text"])
    plt.tight_layout()
    return _fig_to_b64(fig)

def _plot_missing_heatmap(df: pd.DataFrame) -> str:
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(missing)*0.5+2)),
                              facecolor=C["plot_bg"])
    # Barras
    ax = axes[0]
    bars = ax.barh(missing.index, missing.values,
                   color=[C["danger"] if v > 30 else C["warn"] if v > 5 else C["accent2"]
                          for v in missing.values], alpha=0.85)
    ax.set_xlabel("% nulos")
    ax.set_title("% Nulos por columna", color=C["text"])
    ax.axvline(5, color=C["accent3"], ls="--", alpha=0.6, label="5%")
    ax.axvline(30, color=C["danger"], ls="--", alpha=0.6, label="30%")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x")
    for bar, val in zip(bars, missing.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8, color=C["text"])

    # Mini heatmap
    ax2 = axes[1]
    subset = df[missing.index].isnull().astype(int)
    if subset.shape[0] > 500:
        subset = subset.sample(500, random_state=42)
    cmap = matplotlib.colors.ListedColormap([C["card"], C["danger"]])
    ax2.imshow(subset.T.values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax2.set_yticks(range(len(missing)))
    ax2.set_yticklabels(missing.index, fontsize=8)
    ax2.set_xlabel("Observaciones (muestra)")
    ax2.set_title("Mapa de nulos", color=C["text"])

    plt.tight_layout()
    return _fig_to_b64(fig)

def _plot_correlation(df: pd.DataFrame, num_cols: list) -> str:
    if len(num_cols) < 2:
        return None
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(max(7, len(num_cols)), max(6, len(num_cols)*0.85)),
                            facecolor=C["plot_bg"])
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=len(num_cols) <= 18, fmt=".2f", linewidths=0.5,
                linecolor=C["bg"], ax=ax, annot_kws={"size": 8},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlación de Pearson", color=C["text"], fontsize=12, pad=12)
    plt.tight_layout()
    return _fig_to_b64(fig)

def _plot_overview_bars(df: pd.DataFrame) -> str:
    """Mini chart de tipos de columnas y completitud general."""
    n = len(df)
    completeness = [(1 - df[c].isnull().mean())*100 for c in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), facecolor=C["plot_bg"])

    # Completitud por columna
    ax = axes[0]
    colors = [C["accent3"] if v >= 95 else C["warn"] if v >= 70 else C["danger"] for v in completeness]
    ax.bar(range(len(df.columns)), completeness, color=colors, alpha=0.85)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title("Completitud por columna (%)", color=C["text"])
    ax.axhline(100, color=C["border"], ls="--", lw=1)
    ax.grid(True, axis="y")

    # Tipo de columnas (donut)
    ax2 = axes[1]
    num_c  = len(df.select_dtypes("number").columns)
    cat_c  = len(df.select_dtypes(["object","category","bool"]).columns)
    date_c = len(df.select_dtypes(["datetime","datetimetz"]).columns)
    other  = len(df.columns) - num_c - cat_c - date_c
    labels, vals, clrs = [], [], []
    for lbl, v, cl in [("Numéricas", num_c, C["accent"]),
                        ("Categóricas", cat_c, C["accent2"]),
                        ("Fecha", date_c, C["accent3"]),
                        ("Otras", other, C["warn"])]:
        if v > 0:
            labels.append(f"{lbl} ({v})")
            vals.append(v)
            clrs.append(cl)
    wedges, texts, autotexts = ax2.pie(vals, labels=labels, colors=clrs,
                                        autopct="%1.0f%%",
                                        textprops={"color": C["text"], "fontsize": 9},
                                        wedgeprops={"edgecolor": C["bg"], "linewidth": 2},
                                        pctdistance=0.75)
    for at in autotexts:
        at.set_color(C["bg"])
        at.set_fontweight("bold")
    # donut hole
    circle = plt.Circle((0, 0), 0.5, color=C["plot_bg"])
    ax2.add_patch(circle)
    ax2.set_title("Tipos de columnas", color=C["text"])

    plt.tight_layout()
    return _fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ALERTAS AUTOMÁTICAS
# ══════════════════════════════════════════════════════════════════════════════

# Umbrales configurables
NULL_WARN_PCT    = 0.01   # > 0.01% nulos → warning (captura cualquier nulo)
NULL_DANGER_PCT  = 50.0   # > 50%  nulos → danger
ZERO_WARN_PCT    = 5.0    # > 5%   ceros → warning
ZERO_DANGER_PCT  = 50.0   # > 50%  ceros → danger
CORR_WARN        = 0.70   # |r| > 0.70 → correlación alta (warning)
CORR_DANGER      = 0.90   # |r| > 0.90 → correlación muy alta (danger)

def _get_high_correlations(df: pd.DataFrame) -> list:
    """
    Devuelve lista de tuplas (colA, colB, r, level) con correlaciones altas.
    level = 'danger' si |r| > CORR_DANGER, 'warning' si |r| > CORR_WARN.
    """
    num_cols = df.select_dtypes("number").columns.tolist()
    if len(num_cols) < 2:
        return []
    corr = df[num_cols].corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool))
    pairs = (corr.where(~mask).stack()
                 .reset_index()
                 .rename(columns={"level_0": "A", "level_1": "B", 0: "r"})
                 .assign(abs_r=lambda x: x["r"].abs()))
    high = pairs[pairs["abs_r"] >= CORR_WARN].sort_values("abs_r", ascending=False)
    result = []
    for row in high.itertuples():
        level = "danger" if row.abs_r >= CORR_DANGER else "warning"
        result.append((row.A, row.B, round(row.r, 4), level))
    return result


def _generate_alerts(df: pd.DataFrame) -> list:
    """
    Genera alertas agrupadas por categoría:
      NULOS     — cualquier columna con ≥1 nulo
      CEROS     — columnas numéricas con ceros significativos
      CORRELACIÓN — pares con |r| ≥ CORR_WARN
      OTROS     — duplicados, skewness, outliers, cardinalidad, etc.
    """
    alerts = []   # cada elemento: (level, col_or_title, message, category)

    # ── DUPLICADOS ────────────────────────────────────────────────────────────
    dup = df.duplicated().sum()
    if dup > 0:
        alerts.append(("warning", "Filas duplicadas",
                        f"{dup:,} filas duplicadas ({dup/len(df)*100:.1f}%)", "otros"))

    for col in df.columns:
        s = df[col]
        null_c   = s.isnull().sum()
        null_pct = s.isnull().mean() * 100

        # ── NULOS (granular: captura cualquier nulo, no solo >5%) ────────────
        if null_pct > NULL_DANGER_PCT:
            alerts.append(("danger", col,
                            f"Nulos críticos: {null_c:,} de {len(df):,} filas ({null_pct:.1f}%)",
                            "nulos"))
        elif null_pct > NULL_WARN_PCT:
            alerts.append(("warning", col,
                            f"Contiene nulos: {null_c:,} filas ({null_pct:.2f}%)",
                            "nulos"))

        if pd.api.types.is_numeric_dtype(s):
            sn = s.dropna()
            if len(sn) == 0:
                continue

            # ── CEROS ────────────────────────────────────────────────────────
            zero_c   = int((sn == 0).sum())
            zero_pct = (sn == 0).mean() * 100
            if zero_pct > ZERO_DANGER_PCT:
                alerts.append(("danger", col,
                                f"Ceros críticos: {zero_c:,} de {len(sn):,} valores ({zero_pct:.1f}%)",
                                "ceros"))
            elif zero_pct > ZERO_WARN_PCT:
                alerts.append(("warning", col,
                                f"Ceros detectados: {zero_c:,} valores ({zero_pct:.1f}%)",
                                "ceros"))

            # ── OTROS numéricos ───────────────────────────────────────────────
            if sn.nunique() == 1:
                alerts.append(("danger", col, "Columna constante (1 único valor)", "otros"))
            elif sn.nunique() == len(sn):
                alerts.append(("info",  col, "Posible columna ID (todos los valores únicos)", "otros"))
            if abs(sn.skew()) > 2:
                alerts.append(("warning", col, f"Alta asimetría: skewness = {sn.skew():.2f}", "otros"))
            out_pct = _iqr_outliers(sn) / len(sn) * 100
            if out_pct > 10:
                alerts.append(("warning", col, f"Outliers IQR: {out_pct:.1f}% de los datos", "otros"))

        elif pd.api.types.is_object_dtype(s) or hasattr(s, "cat"):
            if s.nunique() > 50:
                alerts.append(("info", col, f"Alta cardinalidad: {s.nunique()} categorías únicas", "otros"))
            if s.nunique() > 1:
                vc = s.value_counts(normalize=True)
                if vc.max() > 0.9:
                    alerts.append(("warning", col,
                                    f"Clase dominante: '{vc.index[0]}' = {vc.max()*100:.1f}%", "otros"))

    # ── CORRELACIONES ALTAS ───────────────────────────────────────────────────
    for colA, colB, r, level in _get_high_correlations(df):
        dir_lbl = "positiva" if r > 0 else "negativa"
        alerts.append((level, f"{colA} ↔ {colB}",
                        f"Correlación {dir_lbl} {'muy alta' if level == 'danger' else 'alta'}: r = {r:.4f}",
                        "correlacion"))

    return alerts

# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
:root {
  --bg: #0d1117; --surface: #161b22; --card: #1c2128; --border: #30363d;
  --accent: #7c3aed; --accent2: #06b6d4; --accent3: #10b981;
  --warn: #f59e0b; --danger: #ef4444; --info: #3b82f6;
  --text: #e6edf3; --muted: #8b949e;
}
*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { background: var(--bg); color: var(--text); font-family: 'Inter', 'Segoe UI', sans-serif;
       font-size: 14px; margin: 0; padding: 0; }

/* NAV */
nav { position: sticky; top: 0; z-index: 100; background: rgba(13,17,23,.95);
      backdrop-filter: blur(10px); border-bottom: 1px solid var(--border);
      padding: 0 2rem; display: flex; align-items: center; gap: 0; }
.nav-brand { font-size: 1.1rem; font-weight: 700; color: var(--text); padding: 1rem 1.5rem 1rem 0;
             border-right: 1px solid var(--border); margin-right: 1rem; letter-spacing: -.3px; }
.nav-brand span { color: var(--accent); }
.nav-link { color: var(--muted); text-decoration: none; padding: .8rem .9rem;
            font-size: 13px; font-weight: 500; transition: color .15s;
            border-bottom: 2px solid transparent; }
.nav-link:hover, .nav-link.active { color: var(--text); border-bottom-color: var(--accent); }

/* LAYOUT */
.container { max-width: 1280px; margin: 0 auto; padding: 2rem 2rem; }

/* SECTION */
.section { margin-bottom: 3rem; }
.section-title { font-size: 1.2rem; font-weight: 700; color: var(--text); margin: 0 0 1.25rem 0;
                 padding-bottom: .6rem; border-bottom: 1px solid var(--border);
                 display: flex; align-items: center; gap: .6rem; }
.section-title .icon { width: 28px; height: 28px; border-radius: 8px; background: var(--accent);
                        display: flex; align-items: center; justify-content: center; font-size: 14px; }

/* OVERVIEW GRID */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
             padding: 1.1rem 1.2rem; transition: border-color .2s; }
.stat-card:hover { border-color: var(--accent); }
.stat-card .label { color: var(--muted); font-size: 11px; font-weight: 600; text-transform: uppercase;
                     letter-spacing: .5px; margin-bottom: .4rem; }
.stat-card .value { font-size: 1.6rem; font-weight: 800; color: var(--text); line-height: 1; }
.stat-card .sub { color: var(--muted); font-size: 11px; margin-top: .25rem; }
.stat-card.accent .value { color: var(--accent); }
.stat-card.green  .value { color: var(--accent3); }
.stat-card.yellow .value { color: var(--warn); }
.stat-card.red    .value { color: var(--danger); }

/* ALERTS */
.alert { display: flex; align-items: flex-start; gap: .75rem; padding: .7rem 1rem;
         border-radius: 8px; margin-bottom: .5rem; font-size: 13px; border: 1px solid; }
.alert-warning  { background: rgba(245,158,11,.08); border-color: rgba(245,158,11,.3); }
.alert-danger   { background: rgba(239,68,68,.08);  border-color: rgba(239,68,68,.3); }
.alert-info     { background: rgba(59,130,246,.08); border-color: rgba(59,130,246,.3); }
.alert .badge   { font-size: 11px; font-weight: 700; padding: .2rem .5rem; border-radius: 5px;
                  text-transform: uppercase; white-space: nowrap; flex-shrink: 0; }
.alert-warning .badge  { background: var(--warn);   color: #000; }
.alert-danger  .badge  { background: var(--danger); color: #fff; }
.alert-info    .badge  { background: var(--info);   color: #fff; }
.alert .col-name { font-weight: 700; color: var(--text); margin-right: .4rem; }
.alert .msg      { color: var(--muted); }

/* ALERT CATEGORIES */
.alert-group { margin-bottom: 1.5rem; }
.alert-group-title { font-size: 11px; font-weight: 700; text-transform: uppercase;
                     letter-spacing: .8px; color: var(--muted); margin: 0 0 .6rem;
                     padding: .3rem .6rem; border-left: 3px solid var(--border);
                     display: flex; align-items: center; gap: .5rem; }
.alert-group-title.cat-nulos  { border-color: var(--danger); color: #fca5a5; }
.alert-group-title.cat-ceros  { border-color: var(--warn);   color: #fcd34d; }
.alert-group-title.cat-corr   { border-color: var(--accent2);color: #67e8f9; }
.alert-group-title.cat-otros  { border-color: var(--muted);  color: var(--muted); }
.alert-count-pill { background: var(--surface); border: 1px solid var(--border);
                    border-radius: 12px; padding: .1rem .5rem; font-size: 10px;
                    font-weight: 800; color: var(--text); }
.alert-corr-danger  { background: rgba(239,68,68,.08);  border-color: rgba(239,68,68,.3); }
.alert-corr-warning { background: rgba(6,182,212,.08);  border-color: rgba(6,182,212,.3); }
.alert-corr-danger  .badge { background: var(--danger); color: #fff; }
.alert-corr-warning .badge { background: var(--accent2); color: #000; }

/* ALERT FILTER BAR */
.alert-filter-bar { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.alert-filter-btn { background: var(--card); border: 1px solid var(--border); border-radius: 6px;
                    padding: .25rem .7rem; color: var(--muted); font-size: 12px; font-weight: 600;
                    cursor: pointer; transition: all .15s; }
.alert-filter-btn:hover  { border-color: var(--accent); color: var(--text); }
.alert-filter-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }

/* VARIABLE CARDS */
.var-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
            margin-bottom: 1rem; overflow: hidden; }
.var-header { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.2rem;
              cursor: pointer; user-select: none; transition: background .15s; }
.var-header:hover { background: rgba(124,58,237,.06); }
.var-header .var-name { font-weight: 700; font-size: 1rem; color: var(--text); flex: 1; }
.var-header .var-type { font-size: 11px; font-weight: 700; padding: .25rem .6rem;
                         border-radius: 5px; text-transform: uppercase; }
.type-num  { background: rgba(124,58,237,.2);  color: #a78bfa; }
.type-cat  { background: rgba(6,182,212,.2);   color: #67e8f9; }
.type-date { background: rgba(16,185,129,.2);  color: #6ee7b7; }
.type-bool { background: rgba(245,158,11,.2);  color: #fcd34d; }
.var-header .chevron { color: var(--muted); transition: transform .25s; font-size: 18px; }
.var-header.open .chevron { transform: rotate(90deg); }
.var-body { display: none; padding: 0 1.2rem 1.2rem; border-top: 1px solid var(--border); }
.var-body.open { display: block; }

/* STATS TABLE inside var-body */
.stats-table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 1rem; }
.stats-table th { color: var(--muted); font-weight: 600; text-align: left; padding: .35rem .6rem;
                   font-size: 11px; text-transform: uppercase; letter-spacing: .4px; }
.stats-table td { padding: .35rem .6rem; border-bottom: 1px solid var(--border); color: var(--text); }
.stats-table tr:last-child td { border-bottom: none; }
.stats-table .val { font-weight: 600; font-family: 'Fira Code', monospace; }

/* mini bar for frequencies */
.freq-bar { height: 6px; background: var(--accent); border-radius: 3px; }

/* IMAGES */
.plot-img { width: 100%; border-radius: 8px; margin-top: 1rem; display: block; }

/* TABS  */
.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 1rem; }
.tab-btn { background: none; border: none; color: var(--muted); padding: .6rem 1rem;
           font-size: 13px; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent;
           transition: color .15s, border-color .15s; }
.tab-btn.active { color: var(--text); border-bottom-color: var(--accent); }
.tab-pane { display: none; }
.tab-pane.active { display: block; }

/* SAMPLE TABLE */
.sample-table { width: 100%; border-collapse: collapse; font-size: 12.5px; overflow-x: auto; display: block; }
.sample-table th { background: var(--surface); color: var(--muted); padding: .5rem .75rem;
                    text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .4px;
                    border-bottom: 1px solid var(--border); white-space: nowrap; }
.sample-table td { padding: .4rem .75rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
.sample-table tr:hover td { background: rgba(124,58,237,.05); }

/* FOOTER */
footer { text-align: center; color: var(--muted); font-size: 12px; padding: 2rem;
         border-top: 1px solid var(--border); margin-top: 3rem; }

/* SEARCH */
.search-box { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
              padding: .5rem 1rem; color: var(--text); font-size: 13px; width: 240px;
              outline: none; transition: border-color .2s; }
.search-box:focus { border-color: var(--accent); }
.search-box::placeholder { color: var(--muted); }
.var-filter-bar { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.filter-btn { background: var(--card); border: 1px solid var(--border); border-radius: 6px;
              padding: .3rem .75rem; color: var(--muted); font-size: 12px; font-weight: 600;
              cursor: pointer; transition: all .15s; }
.filter-btn:hover, .filter-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }

/* RESPONSIVE */
@media (max-width: 768px) {
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  nav { overflow-x: auto; }
}
"""

_JS = """
// Toggle variable cards
document.querySelectorAll('.var-header').forEach(h => {
  h.addEventListener('click', () => {
    h.classList.toggle('open');
    h.nextElementSibling.classList.toggle('open');
  });
});

// Alert category filter
const alertFilterBtns = document.querySelectorAll('.alert-filter-btn[data-cat]');
alertFilterBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    alertFilterBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const cat = btn.dataset.cat;
    document.querySelectorAll('.alert-group').forEach(g => {
      g.style.display = (cat === 'all' || g.dataset.cat === cat) ? '' : 'none';
    });
  });
});

// Tabs
function switchTab(group, id) {
  document.querySelectorAll('[data-tab-group="'+group+'"] .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('[data-tab-group="'+group+'"] .tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelector('[data-tab-group="'+group+'"] [data-tab="'+id+'"]').classList.add('active');
  document.querySelector('#pane-'+id).classList.add('active');
}

// Nav highlight
const sections = document.querySelectorAll('.section[id]');
const navLinks  = document.querySelectorAll('.nav-link');
const observer  = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(l => l.classList.remove('active'));
      const l = document.querySelector('.nav-link[href="#'+e.target.id+'"]');
      if (l) l.classList.add('active');
    }
  });
}, { rootMargin: '-40% 0px -55% 0px' });
sections.forEach(s => observer.observe(s));

// Variable search & filter
const searchBox = document.querySelector('#var-search');
const filterBtns = document.querySelectorAll('.filter-btn[data-type]');
let activeType = 'all';

function filterVars() {
  const q = searchBox ? searchBox.value.toLowerCase() : '';
  document.querySelectorAll('.var-card').forEach(card => {
    const name = card.dataset.name.toLowerCase();
    const type = card.dataset.type;
    const matchQ = name.includes(q);
    const matchT = activeType === 'all' || type === activeType;
    card.style.display = (matchQ && matchT) ? '' : 'none';
  });
}
if (searchBox) searchBox.addEventListener('input', filterVars);
filterBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    filterBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeType = btn.dataset.type;
    filterVars();
  });
});
"""

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS DE TARGET
# ══════════════════════════════════════════════════════════════════════════════

def _plot_target_distribution(series: pd.Series) -> str:
    """Distribución del target: histograma+KDE+QQ para numérico, barras+donut para categórico."""
    is_num = pd.api.types.is_numeric_dtype(series)
    s = series.dropna()

    if is_num:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), facecolor=C["plot_bg"])
        fig.subplots_adjust(wspace=0.35)

        # Histograma + KDE
        ax = axes[0]
        ax.hist(s, bins=40, color=C["accent3"], alpha=0.75, density=True)
        try:
            kde = stats.gaussian_kde(s)
            x = np.linspace(s.min(), s.max(), 300)
            ax.plot(x, kde(x), color=C["accent"], lw=2.5)
        except Exception:
            pass
        ax.axvline(s.mean(),   color=C["warn"],   lw=1.5, linestyle="--", label=f"Media={s.mean():.2f}")
        ax.axvline(s.median(), color=C["accent2"],lw=1.5, linestyle=":",  label=f"Mediana={s.median():.2f}")
        ax.set_title(f"Distribución  (skew={s.skew():.2f})", color=C["text"], fontsize=10)
        ax.legend(fontsize=8); ax.grid(True)

        # Boxplot
        ax = axes[1]
        ax.boxplot(s, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=C["accent3"], alpha=0.5),
                   medianprops=dict(color=C["accent"], lw=2.5),
                   whiskerprops=dict(color=C["muted"]),
                   capprops=dict(color=C["muted"]),
                   flierprops=dict(marker="o", color=C["danger"], alpha=0.4, markersize=3))
        ax.set_title("Boxplot", color=C["text"], fontsize=10); ax.grid(True)

        # Q-Q
        ax = axes[2]
        (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
        ax.scatter(osm, osr, color=C["accent3"], s=10, alpha=0.5)
        lx = np.array([min(osm), max(osm)])
        ax.plot(lx, slope * lx + intercept, color=C["accent"], lw=2)
        ax.set_title(f"Q-Q Plot  (r={r:.3f})", color=C["text"], fontsize=10); ax.grid(True)

    else:
        vc = s.value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, len(vc)*0.5)), facecolor=C["plot_bg"])

        # Barras
        ax = axes[0]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(vc))]
        bars = ax.bar(range(len(vc)), vc.values, color=colors, alpha=0.85)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels([str(v)[:20] for v in vc.index], rotation=30, ha="right", fontsize=9)
        ax.set_title("Conteo por clase", color=C["text"], fontsize=10); ax.grid(True, axis="y")
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:,}\n({val/len(s)*100:.1f}%)", ha="center", fontsize=8, color=C["text"])

        # Donut
        ax = axes[1]
        wedges, texts, autotexts = ax.pie(
            vc.values, labels=[str(v)[:18] for v in vc.index],
            colors=colors, autopct="%1.1f%%",
            textprops={"color": C["text"], "fontsize": 9},
            wedgeprops={"edgecolor": C["bg"], "linewidth": 2},
            pctdistance=0.78)
        for at in autotexts:
            at.set_color(C["bg"]); at.set_fontweight("bold")
        ax.add_patch(plt.Circle((0, 0), 0.5, color=C["plot_bg"]))
        ax.set_title("Proporción de clases", color=C["text"], fontsize=10)

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_feature_vs_target(feature: pd.Series, target: pd.Series) -> str:
    """
    Genera el gráfico adecuado según los tipos de feature y target:
      num → num  : scatter + regresión lineal
      num → cat  : boxplot por clase
      cat → num  : barras de media por categoría
      cat → cat  : barras agrupadas / heatmap de contingencia
    """
    f_num = pd.api.types.is_numeric_dtype(feature)
    t_num = pd.api.types.is_numeric_dtype(target)
    valid = feature.notna() & target.notna()
    f_v, t_v = feature[valid], target[valid]

    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=C["plot_bg"])

    if f_num and t_num:
        # Scatter + regresión
        ax.scatter(f_v, t_v, alpha=0.25, s=10, color=C["accent"])
        try:
            m, b, r, p, _ = stats.linregress(f_v, t_v)
            x_line = np.linspace(f_v.min(), f_v.max(), 200)
            ax.plot(x_line, m*x_line + b, color=C["accent3"], lw=2)
            ax.set_title(f"r = {r:.3f}  (p={'<0.001' if p < 0.001 else f'{p:.3f}'})",
                         color=C["text"], fontsize=10)
        except Exception:
            pass
        ax.set_xlabel(feature.name); ax.set_ylabel(target.name)

    elif f_num and not t_num:
        # Boxplot del feature agrupado por clase de target
        classes = t_v.unique()
        data_by_class = [f_v[t_v == cls].dropna().values for cls in classes]
        bp = ax.boxplot(data_by_class, patch_artist=True,
                        medianprops=dict(color=C["bg"], lw=2))
        for patch, clr in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(clr); patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(classes)+1))
        ax.set_xticklabels([str(c)[:15] for c in classes], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{feature.name} por clase de {target.name}", color=C["text"], fontsize=10)
        ax.set_ylabel(feature.name)

    elif not f_num and t_num:
        # Barras de media del target por categoría de feature
        top_cats = f_v.value_counts().head(12).index
        means = [t_v[f_v == cat].mean() for cat in top_cats]
        sems  = [t_v[f_v == cat].sem()  for cat in top_cats]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(top_cats))]
        ax.bar(range(len(top_cats)), means, color=colors, alpha=0.8,
               yerr=sems, capsize=4, error_kw={"ecolor": C["muted"], "lw": 1.2})
        ax.set_xticks(range(len(top_cats)))
        ax.set_xticklabels([str(c)[:15] for c in top_cats], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Media de {target.name} por {feature.name}", color=C["text"], fontsize=10)
        ax.set_ylabel(f"Media {target.name}")

    else:
        # Heatmap de contingencia normalizado
        top_f = f_v.value_counts().head(8).index
        top_t = t_v.value_counts().head(6).index
        ct = pd.crosstab(f_v[f_v.isin(top_f)], t_v[t_v.isin(top_t)], normalize="index") * 100
        cmap = sns.light_palette(C["accent"], as_cmap=True)
        sns.heatmap(ct, annot=True, fmt=".1f", cmap=cmap, ax=ax,
                    linewidths=0.5, linecolor=C["bg"],
                    annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
        ax.set_title(f"% distribución {target.name} (por fila)", color=C["text"], fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _target_stats_html(series: pd.Series) -> str:
    """Tabla de estadísticas del target según su tipo."""
    is_num = pd.api.types.is_numeric_dtype(series)
    s = series.dropna()
    rows = ""
    if is_num:
        _, p = stats.shapiro(s.sample(min(5000, len(s)), random_state=42))
        kv = [
            ("Tipo",         str(series.dtype)),
            ("No nulos",     len(s)),
            ("Nulos",        series.isnull().sum()),
            ("Media",        s.mean()),
            ("Mediana",      s.median()),
            ("Desv. std",    s.std()),
            ("Mínimo",       s.min()),
            ("Máximo",       s.max()),
            ("Asimetría",    s.skew()),
            ("Curtosis",     s.kurtosis()),
            ("Outliers IQR", _iqr_outliers(s)),
            ("Shapiro-Wilk", f"p={p:.4f} {'✅ Normal' if p > 0.05 else '⚠️ No normal'}"),
        ]
        rows = _stat_rows(kv)
    else:
        vc = series.value_counts()
        imbalance = vc.max() / max(vc.min(), 1)
        kv = [
            ("Tipo",         str(series.dtype)),
            ("No nulos",     len(s)),
            ("Nulos",        series.isnull().sum()),
            ("Clases",       series.nunique()),
            ("Clase mayoritaria", f"{vc.index[0]} ({vc.iloc[0]:,})"),
            ("Clase minoritaria", f"{vc.index[-1]} ({vc.iloc[-1]:,})"),
            ("Ratio desbalance",  f"{imbalance:.2f}x {'⚠️' if imbalance > 3 else '✅'}"),
        ]
        rows = _stat_rows(kv)
    return f'<table class="stats-table" style="max-width:320px">{rows}</table>'


def _stat_rows(kv_pairs: list) -> str:
    rows = ""
    for k, v in kv_pairs:
        rows += f'<tr><th>{k}</th><td class="val">{_fmt(v)}</td></tr>'
    return rows

def _alert_icon(level: str) -> str:
    return {"warning": "⚠️", "danger": "🔴", "info": "ℹ️"}.get(level, "•")

def _type_badge(dtype) -> tuple:
    """Returns (label, css_class)"""
    if pd.api.types.is_numeric_dtype(dtype):
        return "Numérica", "type-num"
    if pd.api.types.is_bool_dtype(dtype):
        return "Booleana", "type-bool"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "Fecha", "type-date"
    return "Categórica", "type-cat"

def _type_key(dtype) -> str:
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "date"
    return "categorical"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class EDAProfiler:
    """
    Genera un reporte HTML interactivo completo de un DataFrame.

    Parámetros
    ----------
    df      : pd.DataFrame
    title   : str   — título del reporte
    minimal : bool  — si True, omite gráficos por columna (más rápido)
    """

    def __init__(self, df: pd.DataFrame, title: str = "EDA Report",
                 target: str = None, minimal: bool = False):
        self.df      = df.copy()
        self.title   = title
        self.target  = target if target and target in df.columns else None
        self.minimal = minimal
        self._html   = None
        if target and target not in df.columns:
            print(f"⚠️  Target '{target}' no encontrado en el DataFrame. Se ignorará.")

    def _build(self) -> str:
        df      = self.df
        target  = self.target
        n_rows, n_cols = df.shape
        total   = n_rows * n_cols
        missing = df.isnull().sum().sum()
        dup     = df.duplicated().sum()
        mem_mb  = df.memory_usage(deep=True).sum() / 1e6
        now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        num_cols  = df.select_dtypes("number").columns.tolist()
        cat_cols  = df.select_dtypes(["object","category","bool"]).columns.tolist()
        date_cols = df.select_dtypes(["datetime","datetimetz"]).columns.tolist()

        alerts = _generate_alerts(df)

        # ── OVERVIEW SECTION ──────────────────────────────────────────────────
        miss_pct = missing/total*100
        dup_pct  = dup/n_rows*100

        miss_cls = "red" if miss_pct > 20 else "yellow" if miss_pct > 5 else "green"
        dup_cls  = "red" if dup_pct > 10 else "yellow" if dup_pct > 1 else "green"

        stat_cards = f"""
        <div class="stats-grid">
          <div class="stat-card accent"><div class="label">Filas</div>
            <div class="value">{n_rows:,}</div></div>
          <div class="stat-card accent"><div class="label">Columnas</div>
            <div class="value">{n_cols}</div></div>
          <div class="stat-card"><div class="label">Numéricas</div>
            <div class="value" style="color:var(--accent)">{len(num_cols)}</div></div>
          <div class="stat-card"><div class="label">Categóricas</div>
            <div class="value" style="color:var(--accent2)">{len(cat_cols)}</div></div>
          <div class="stat-card {miss_cls}"><div class="label">Nulos</div>
            <div class="value">{miss_pct:.1f}%</div>
            <div class="sub">{missing:,} celdas</div></div>
          <div class="stat-card {dup_cls}"><div class="label">Duplicados</div>
            <div class="value">{dup_pct:.1f}%</div>
            <div class="sub">{dup:,} filas</div></div>
          <div class="stat-card"><div class="label">Memoria</div>
            <div class="value">{mem_mb:.1f}</div><div class="sub">MB</div></div>
          <div class="stat-card"><div class="label">Alertas</div>
            <div class="value" style="color:var(--warn)">{len(alerts)}</div></div>
        </div>"""

        overview_img = _plot_overview_bars(df)
        overview_plot = f'<img class="plot-img" src="data:image/png;base64,{overview_img}" alt="overview">'

        # ── ALERTS SECTION ────────────────────────────────────────────────────
        # Agrupa por categoría
        from collections import defaultdict
        alert_groups = defaultdict(list)
        for a in alerts:
            alert_groups[a[3]].append(a)

        cat_meta = {
            "nulos":      ("🕳️ Valores Nulos",     "cat-nulos",  len(alert_groups["nulos"])),
            "ceros":      ("🟡 Ceros en Variables", "cat-ceros",  len(alert_groups["ceros"])),
            "correlacion":("🔗 Correlación Alta",   "cat-corr",   len(alert_groups["correlacion"])),
            "otros":      ("⚙️ Otros",              "cat-otros",  len(alert_groups["otros"])),
        }

        def _render_alert(level, col, msg, cat):
            if cat == "correlacion":
                css = f"alert-corr-{level}"
                icon = "🔗"
            else:
                css = f"alert-{level}"
                icon = {"warning": "⚠️", "danger": "🔴", "info": "ℹ️"}.get(level, "•")
            return (f'<div class="alert {css}">'
                    f'<span class="badge">{icon} {level}</span>'
                    f'<div><span class="col-name">{col}</span>'
                    f'<span class="msg">{msg}</span></div></div>')

        groups_html = ""
        for cat_key in ["nulos", "ceros", "correlacion", "otros"]:
            items = alert_groups.get(cat_key, [])
            if not items:
                continue
            title, css_cls, cnt = cat_meta[cat_key]
            inner = "".join(_render_alert(a[0], a[1], a[2], a[3]) for a in items)
            groups_html += (f'<div class="alert-group" data-cat="{cat_key}">'
                            f'<div class="alert-group-title {css_cls}">{title}'
                            f'<span class="alert-count-pill">{cnt}</span></div>'
                            f'{inner}</div>')

        if not alerts:
            groups_html = '<p style="color:var(--accent3)">✅ No se detectaron problemas en el dataset.</p>'

        # Pills de filtro
        filter_pills = '<button class="alert-filter-btn active" data-cat="all">Todas</button>'
        for cat_key in ["nulos", "ceros", "correlacion", "otros"]:
            items = alert_groups.get(cat_key, [])
            if items:
                title, _, cnt = cat_meta[cat_key]
                # Solo el emoji + nombre corto
                short = title.split(" ", 1)[1].split(" ")[0] if " " in title else title
                filter_pills += (f'<button class="alert-filter-btn" data-cat="{cat_key}">'
                                 f'{title.split()[0]} {short} <span style="opacity:.7">({cnt})</span></button>')

        alerts_html = f'<div class="alert-filter-bar">{filter_pills}</div>{groups_html}'

        # ── MISSING SECTION ───────────────────────────────────────────────────
        missing_img = _plot_missing_heatmap(df)
        missing_plot = ""
        if missing_img:
            missing_plot = f'<img class="plot-img" src="data:image/png;base64,{missing_img}" alt="missing">'
        else:
            missing_plot = '<p style="color:var(--accent3)">✅ Dataset sin valores nulos.</p>'

        # ── CORRELATION SECTION ───────────────────────────────────────────────
        corr_img = _plot_correlation(df, num_cols)
        corr_plot = ""
        if corr_img:
            corr_plot = f'<img class="plot-img" src="data:image/png;base64,{corr_img}" alt="correlation">'
        else:
            corr_plot = '<p style="color:var(--muted)">Se necesitan ≥2 columnas numéricas.</p>'

        # Panel de correlaciones altas
        high_corr_list = _get_high_correlations(df)
        corr_alert_panel = ""
        if high_corr_list:
            rows_corr = ""
            for colA, colB, r, level in high_corr_list:
                badge_cls = "var(--danger)" if level == "danger" else "var(--accent2)"
                badge_txt = "Muy Alta" if level == "danger" else "Alta"
                dir_arrow = "↗" if r > 0 else "↘"
                rows_corr += (f"<tr>"
                              f"<td><b>{colA}</b></td><td><b>{colB}</b></td>"
                              f"<td style='color:{badge_cls};font-family:monospace;font-weight:700'>{dir_arrow} {r:.4f}</td>"
                              f"<td><span style='background:{badge_cls}20;color:{badge_cls};"
                              f"border-radius:4px;padding:.15rem .5rem;font-size:11px;font-weight:700'>"
                              f"{badge_txt}</span></td></tr>")
            corr_alert_panel = f"""
            <div style="margin-top:1.5rem">
              <h4 style="font-size:.95rem;color:var(--muted);margin:0 0 .75rem;
                         display:flex;align-items:center;gap:.5rem">
                🔗 Pares con correlación alta
                <span class="alert-count-pill">{len(high_corr_list)}</span>
              </h4>
              <div style="background:var(--surface);border:1px solid var(--border);
                          border-radius:10px;overflow:hidden">
                <table class="sample-table">
                  <tr><th>Variable A</th><th>Variable B</th><th>Pearson r</th><th>Nivel</th></tr>
                  {rows_corr}
                </table>
              </div>
              <p style="color:var(--muted);font-size:12px;margin:.6rem 0 0">
                Umbral: |r| ≥ {CORR_WARN} (alta) · |r| ≥ {CORR_DANGER} (muy alta)
              </p>
            </div>"""
        else:
            corr_alert_panel = '<p style="color:var(--accent3);margin-top:1rem">✅ No se detectaron correlaciones altas (|r| ≥ 0.70).</p>'

        # Top 15 tabla completa
        corr_table = ""
        if len(num_cols) >= 2:
            corr_df = df[num_cols].corr()
            mask    = np.triu(np.ones(corr_df.shape, dtype=bool))
            pairs   = (corr_df.where(~mask).stack().reset_index()
                              .rename(columns={"level_0":"A","level_1":"B",0:"r"})
                              .assign(abs_r=lambda x: x.r.abs())
                              .sort_values("abs_r", ascending=False).head(15))
            rows = "".join(
                f"<tr><td>{r.A}</td><td>{r.B}</td>"
                f"<td style='color:{'var(--danger)' if r.r < -0.5 else 'var(--accent3)' if r.r > 0.5 else 'var(--muted)'}'>"
                f"<b>{r.r:.4f}</b></td></tr>"
                for r in pairs.itertuples()
            )
            corr_table = f"""
            <h4 style="margin:1.5rem 0 .8rem;font-size:.95rem;color:var(--muted)">Top 15 — todas las correlaciones</h4>
            <table class="sample-table" style="max-width:420px">
              <tr><th>Col A</th><th>Col B</th><th>Pearson r</th></tr>{rows}
            </table>"""

        # ── VARIABLES SECTION ─────────────────────────────────────────────────
        vars_html = ""
        for col in df.columns:
            s     = df[col]
            tkey  = _type_key(s.dtype)
            tlbl, tcls = _type_badge(s.dtype)
            null_c   = s.isnull().sum()
            null_pct = null_c / n_rows * 100
            distinct = s.nunique()

            body_content = ""

            if pd.api.types.is_numeric_dtype(s):
                ana = _analyze_numeric(s)
                kv_left = [
                    ("Tipo dtype", str(s.dtype)),
                    ("Valores no nulos", ana.get("n", 0)),
                    ("Nulos", f"{null_c:,} ({null_pct:.1f}%)"),
                    ("Únicos", distinct),
                    ("Media", ana.get("mean", "")),
                    ("Desv. estándar", ana.get("std", "")),
                    ("Mínimo", ana.get("min", "")),
                    ("Máximo", ana.get("max", "")),
                ]
                kv_right = [
                    ("p5",  ana.get("p5", "")),
                    ("p25 (Q1)", ana.get("p25", "")),
                    ("Mediana", ana.get("median", "")),
                    ("p75 (Q3)", ana.get("p75", "")),
                    ("p95", ana.get("p95", "")),
                    ("IQR",  ana.get("iqr", "")),
                    ("Asimetría", ana.get("skewness", "")),
                    ("Curtosis",  ana.get("kurtosis", "")),
                ]
                kv_extra = [
                    ("Ceros", f"{ana.get('zeros',0):,} ({ana.get('zeros_pct',0):.1f}%)"),
                    ("Negativos", ana.get("negatives", 0)),
                    ("Outliers IQR", f"{ana.get('outliers',0):,} ({ana.get('outliers_pct',0):.1f}%)"),
                    ("Normalidad (Shapiro)", "✅ Normal" if ana.get("is_normal") else f"⚠️ No normal (p={ana.get('p_shapiro',0):.4f})"),
                ]
                body_content = f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem">
                  <div><table class="stats-table">{_stat_rows(kv_left)}</table></div>
                  <div><table class="stats-table">{_stat_rows(kv_right)}</table></div>
                </div>
                <table class="stats-table" style="margin-top:.5rem">{_stat_rows(kv_extra)}</table>"""
                if not self.minimal:
                    img = _plot_numeric_col(s)
                    body_content += f'<img class="plot-img" src="data:image/png;base64,{img}" alt="{col}">'

            else:
                ana = _analyze_categorical(s)
                kv = [
                    ("Tipo dtype", str(s.dtype)),
                    ("Valores no nulos", ana.get("n", 0)),
                    ("Nulos", f"{null_c:,} ({null_pct:.1f}%)"),
                    ("Únicos", distinct),
                    ("Único %", f"{ana.get('unique_pct',0):.1f}%"),
                    ("Moda", str(ana.get("top",""))),
                    ("Frecuencia moda", f"{ana.get('top_freq',0):,} ({ana.get('top_pct',0):.1f}%)"),
                ]
                # Freq table
                vc = ana.get("value_counts", {})
                total_s = sum(vc.values())
                freq_rows = ""
                for val, cnt in list(vc.items())[:15]:
                    pct_v = cnt / total_s * 100 if total_s else 0
                    freq_rows += f"""<tr>
                      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">{str(val)[:40]}</td>
                      <td class="val">{cnt:,}</td>
                      <td>{pct_v:.1f}%</td>
                      <td style="width:120px"><div class="freq-bar" style="width:{pct_v}%"></div></td>
                    </tr>"""

                body_content = f"""
                <div style="display:grid;grid-template-columns:200px 1fr;gap:1.5rem;margin-top:1rem;align-items:start">
                  <table class="stats-table">{_stat_rows(kv)}</table>
                  <div>
                    <table class="sample-table">
                      <tr><th>Valor</th><th>Conteo</th><th>%</th><th>Freq</th></tr>
                      {freq_rows}
                    </table>
                  </div>
                </div>"""
                if not self.minimal:
                    img = _plot_categorical_col(s)
                    body_content += f'<img class="plot-img" src="data:image/png;base64,{img}" alt="{col}">'

            # Quick stats en el header — target badge si corresponde
            target_badge = ""
            if col == target:
                target_badge = '<span style="background:rgba(16,185,129,.2);color:#6ee7b7;font-size:11px;font-weight:700;padding:.2rem .55rem;border-radius:5px;margin-right:.3rem">🎯 TARGET</span>'
            header_meta = f'<span style="color:var(--muted);font-size:12px">{_fmt(null_pct)}% nulos &nbsp;·&nbsp; {distinct:,} únicos</span>'

            # Borde especial para target
            card_style = 'style="border-color:#10b981;box-shadow:0 0 0 1px #10b98122"' if col == target else ''

            vars_html += f"""
            <div class="var-card" data-name="{col}" data-type="{tkey}" {card_style}>
              <div class="var-header">
                <span class="var-name">{col}</span>
                {header_meta}
                {target_badge}
                <span class="var-type {tcls}">{tlbl}</span>
                <span class="chevron">›</span>
              </div>
              <div class="var-body">{body_content}</div>
            </div>"""

        # ── TARGET SECTION ────────────────────────────────────────────────────
        target_section_html = ""
        if target:
            t_series = df[target]
            t_is_num = pd.api.types.is_numeric_dtype(t_series)
            task_lbl = "Regresión" if t_is_num else "Clasificación"
            task_clr = C["accent3"] if t_is_num else C["accent2"]

            # Distribución del target
            dist_img = _plot_target_distribution(t_series)

            # Relación features → target (solo las primeras 9 features, excluyendo target)
            feat_cols = [c for c in df.columns if c != target]
            feature_plots_html = ""
            for fc in feat_cols[:9]:
                fimg = _plot_feature_vs_target(df[fc], t_series)
                feature_plots_html += f"""
                <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;
                            padding:1rem;display:flex;flex-direction:column;gap:.5rem">
                  <span style="font-weight:700;font-size:.9rem;color:var(--text)">{fc}</span>
                  <img src="data:image/png;base64,{fimg}" style="border-radius:6px;width:100%">
                </div>"""

            target_section_html = f"""
            <h3 style="margin:0 0 1rem;font-size:1rem;color:var(--muted)">
              Variable objetivo:
              <span style="color:var(--accent3);font-weight:800">{target}</span>
              &nbsp;·&nbsp;
              <span style="background:{task_clr}22;color:{task_clr};border-radius:5px;
                           padding:.2rem .6rem;font-size:12px;font-weight:700">{task_lbl}</span>
            </h3>

            <div style="display:grid;grid-template-columns:auto 1fr;gap:1.5rem;align-items:start;margin-bottom:1.5rem">
              {_target_stats_html(t_series)}
              <img src="data:image/png;base64,{dist_img}" style="border-radius:8px;width:100%">
            </div>

            <h4 style="font-size:.95rem;color:var(--muted);margin:1.5rem 0 .75rem;
                       border-top:1px solid var(--border);padding-top:1rem">
              📐 Relación de cada feature con <span style="color:var(--accent3)">{target}</span>
              <span style="font-size:12px;font-weight:400"> — primeras {min(9,len(feat_cols))} variables</span>
            </h4>
            <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1rem">
              {feature_plots_html}
            </div>"""

        # ── SAMPLE TABLE ──────────────────────────────────────────────────────
        sample_head = df.head(10)
        sample_tail = df.tail(10)

        def _df_to_html_table(d):
            cols_th = "".join(f"<th>{c}</th>" for c in d.columns)
            rows = ""
            for _, row in d.iterrows():
                cells = "".join(f"<td>{str(v)[:40]}</td>" for v in row)
                rows += f"<tr>{cells}</tr>"
            return f'<div style="overflow-x:auto"><table class="sample-table"><tr>{cols_th}</tr>{rows}</table></div>'

        sample_html = f"""
        <div class="tabs" data-tab-group="sample">
          <button class="tab-btn active" data-tab="head" onclick="switchTab('sample','head')">Primeras 10 filas</button>
          <button class="tab-btn" data-tab="tail" onclick="switchTab('sample','tail')">Últimas 10 filas</button>
        </div>
        <div data-tab-group="sample">
          <div id="pane-head" class="tab-pane active">{_df_to_html_table(sample_head)}</div>
          <div id="pane-tail" class="tab-pane">{_df_to_html_table(sample_tail)}</div>
        </div>"""

        # ── ASSEMBLE ──────────────────────────────────────────────────────────
        type_counts = {"numeric": len(num_cols), "categorical": len(cat_cols), "date": len(date_cols)}
        filter_btns = '<button class="filter-btn active" data-type="all">Todas</button>'
        for tkey, lbl in [("numeric","Numéricas"),("categorical","Categóricas"),("date","Fechas")]:
            if type_counts[tkey]:
                filter_btns += f'<button class="filter-btn" data-type="{tkey}">{lbl} ({type_counts[tkey]})</button>'

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
  <style>{_CSS}</style>
</head>
<body>

<nav>
  <div class="nav-brand">EDA <span>Profiler</span></div>
  <a class="nav-link active" href="#overview">Resumen</a>
  <a class="nav-link" href="#alerts">Alertas <span style="background:rgba(245,158,11,.2);color:var(--warn);border-radius:4px;padding:.1rem .4rem;font-size:11px">{len(alerts)}</span></a>
  <a class="nav-link" href="#missing">Nulos</a>
  <a class="nav-link" href="#variables">Variables</a>
  {'<a class="nav-link" href="#target" style="color:#6ee7b7">🎯 Target</a>' if target else ''}
  <a class="nav-link" href="#correlation">Correlación</a>
  <a class="nav-link" href="#sample">Muestra</a>
</nav>

<div class="container">

  <!-- HEADER -->
  <div style="padding: 2rem 0 1.5rem">
    <h1 style="margin:0;font-size:1.8rem;font-weight:800">{self.title}</h1>
    <p style="color:var(--muted);margin:.4rem 0 0;font-size:13px">
      Generado el {now} &nbsp;·&nbsp; {n_rows:,} filas &nbsp;·&nbsp; {n_cols} columnas
    </p>
  </div>

  <!-- OVERVIEW -->
  <section class="section" id="overview">
    <h2 class="section-title"><span class="icon">📊</span> Resumen General</h2>
    {stat_cards}
    {overview_plot}
  </section>

  <!-- ALERTS -->
  <section class="section" id="alerts">
    <h2 class="section-title"><span class="icon">⚠️</span> Alertas
      <span style="font-size:13px;font-weight:400;color:var(--muted)">— {len(alerts)} problema(s) detectado(s)</span>
    </h2>
    {alerts_html}
  </section>

  <!-- MISSING -->
  <section class="section" id="missing">
    <h2 class="section-title"><span class="icon">🕳️</span> Valores Nulos</h2>
    {missing_plot}
  </section>

  <!-- VARIABLES -->
  <section class="section" id="variables">
    <h2 class="section-title"><span class="icon">📋</span> Variables ({n_cols})</h2>
    <div class="var-filter-bar">
      <input class="search-box" id="var-search" type="text" placeholder="Buscar variable...">
      {filter_btns}
    </div>
    {vars_html}
  </section>

  <!-- TARGET -->
  {'<section class="section" id="target"><h2 class="section-title"><span class="icon" style="background:var(--accent3)">🎯</span> Análisis del Target</h2>' + target_section_html + '</section>' if target else ''}

  <!-- CORRELATION -->
  <section class="section" id="correlation">
    <h2 class="section-title"><span class="icon">🔗</span> Correlaciones</h2>
    {corr_plot}
    {corr_alert_panel}
    {corr_table}
  </section>

  <!-- SAMPLE -->
  <section class="section" id="sample">
    <h2 class="section-title"><span class="icon">🗂️</span> Muestra de datos</h2>
    {sample_html}
  </section>

</div>

<footer>
  Generado con <strong>EDA Profiler</strong> &nbsp;·&nbsp; {now}
</footer>

<script>{_JS}</script>
</body>
</html>"""
        return html

    def to_file(self, path: str = "eda_report.html"):
        """Guarda el reporte HTML en disco."""
        html = self._build()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ Reporte guardado en: {path}")
        return self

    def to_notebook_iframe(self, height: int = 900):
        """Muestra el reporte embebido en un Jupyter Notebook."""
        from IPython.display import IFrame, display
        path = "_eda_tmp_report.html"
        self.to_file(path)
        display(IFrame(path, width="100%", height=height))
        return self

    def to_html(self) -> str:
        """Devuelve el HTML como string."""
        return self._build()


# ── Función de conveniencia ────────────────────────────────────────────────────
def profile_report(df: pd.DataFrame,
                   output_path: str = "eda_report.html",
                   title: str = "EDA Report",
                   target: str = None,
                   minimal: bool = False) -> EDAProfiler:
    """
    Genera y guarda un reporte HTML con una sola llamada.

    Ejemplo:
        profile_report(df, "reporte.html", title="Diabetes", target="outcome")
    """
    p = EDAProfiler(df, title=title, target=target, minimal=minimal)
    p.to_file(output_path)
    return p
