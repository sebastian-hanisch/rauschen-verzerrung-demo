"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import rau_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _de(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def build_compare(table: list[dict]):
    """Je Modell zwei Balken in Prozentpunkten: der Lagenunterschied Δ und das Rauschen (100 minus Fairwert, der Mitte der beiden Lagen: Näherung für den Anteil, in dem die Stärkste nicht gewinnt)."""
    models = [r["model"] for r in table]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Lagenunterschied Δ", y=models, x=[100 * r["delta"] for r in table], orientation="h", marker_color=[C.COLORS[m] for m in models],
                         text=[_de(100 * r["delta"], 2) for r in table], textposition="outside", hovertemplate="%{y}: Δ = %{x:.2f} Pp<extra></extra>"))
    fig.add_trace(go.Bar(name="Rauschen (100 minus Fairwert)", y=models, x=[100 * r["noise"] for r in table], orientation="h", marker_color=C.COLOR_GREY,
                         text=[_de(100 * r["noise"], 1) for r in table], textposition="outside", hovertemplate="%{y}: Rauschen = %{x:.1f} Pp<extra></extra>"))
    fig.update_xaxes(title_text="Prozentpunkte der Gewinnchance der stärksten Seite", range=[0, 105])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="group")
    return _base(fig, 380)


def build_needed(table: list[dict]):
    """Benötigte Zahl beobachteter Wettbewerbe je Lage (logarithmische Achse); die Spanne zeigt Δ ± 2 Standardfehler bei den simulierten Modellen."""
    models = [r["model"] for r in table]
    finite = [r for r in table if not math.isinf(r["n"])]
    xs = [r["n"] if not math.isinf(r["n"]) else None for r in table]
    err_hi = [(r["n_high"] - r["n"]) if (not math.isinf(r["n"]) and not math.isinf(r["n_high"])) else 0 for r in table]
    err_lo = [(r["n"] - r["n_low"]) if not math.isinf(r["n"]) else 0 for r in table]
    fig = go.Figure(go.Bar(y=models, x=xs, orientation="h", marker_color=[C.COLORS[m] for m in models],
                           error_x=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo, visible=True),
                           text=[(f"{int(r['n']):,}".replace(",", ".") if not math.isinf(r["n"]) else "unbestimmt") for r in table], textposition="outside",
                           hovertemplate="%{y}: N = %{x:,.0f}<extra></extra>"))
    top = max([r["n_high"] if not math.isinf(r["n_high"]) else r["n"] for r in finite] + [10]) if finite else 10
    fig.update_xaxes(type="log", title_text="beobachtete Wettbewerbe N (logarithmische Achse)", range=[0, math.log10(top * 8)])
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 330)


def build_example(path, n_two: float, delta: float, model: str):
    """Saison für Saison: gemessene Differenz der Anteile (gute minus schlechte Lage) mit Zufallsband; gestrichelt der wahre Wert Δ, senkrecht die Zahl der Formel."""
    s = path.seasons
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([s, s[::-1]]), y=np.concatenate([100 * path.band, -100 * path.band[::-1]]), fill="toself", fillcolor="rgba(150,150,150,0.25)",
                             line=dict(width=0), hoverinfo="skip", name="Zufallsband (nur Zufall)"))
    fig.add_trace(go.Scatter(x=s, y=100 * path.diff, mode="lines", line=dict(color=C.COLORS[model], width=2), name="gemessener Unterschied",
                             hovertemplate="Saison %{x}: %{y:.2f} Pp<extra></extra>"))
    fig.add_trace(go.Scatter(x=[s[0], s[-1]], y=[100 * delta, 100 * delta], mode="lines", line=dict(color=C.COLORS[model], dash="dash", width=1.5), name="wahrer Lagenunterschied Δ",
                             hovertemplate="Δ = %{y:.2f} Pp<extra></extra>"))
    if not math.isinf(n_two) and n_two <= s[-1]:
        fig.add_vline(x=n_two, line=dict(color="#555", dash="dot"), annotation_text="N nach der Formel", annotation_position="top")
    top = max(float(np.nanmax(np.abs(100 * path.diff[min(len(s) - 1, 4):]))) if len(s) > 5 else 10.0, 100 * abs(delta), 5.0)
    fig.update_yaxes(title_text="Unterschied der Siegquoten (Pp)", range=[-1.3 * top, 1.3 * top])
    fig.update_xaxes(title_text="beobachtete Saisons (je Saison ein Wettbewerb in jeder Lage)")
    return _base(fig, 400)


def build_noise(rows: list[dict], selected: int):
    """Chance der Stärksten an der Spitze ohne Verzerrung gegen die Zahl der Partien je Paar (Raster)."""
    xs = [str(r["games"]) for r in rows]
    colors = [C.COLORS["Spielplan und Form"] if r["games"] == selected else C.COLOR_GREY for r in rows]
    fig = go.Figure(go.Bar(x=xs, y=[100 * r["p_top"] for r in rows], marker_color=colors, error_y=dict(type="data", array=[200 * r["se"] for r in rows], visible=True),
                           text=[_de(100 * r["p_top"], 1) for r in rows], textposition="outside", hovertemplate="%{x} Partien je Paar: %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Partien je Paar", type="category")
    fig.update_yaxes(title_text="Stärkste an der Spitze (%)", range=[0, 112])
    return _base(fig, 330)


def build_sweep(series: dict, rows_n: dict):
    """Benötigte Zahl gegen den Lagenunterschied Δ, eine Linie je Reihe (beide Achsen logarithmisch); Punkte ohne Unterschied (Δ = 0) fehlen, ihr N ist unendlich."""
    fig = go.Figure()
    palette = {"plan": C.COLORS["Spielplan und Form"], "duel": C.COLORS["Hin- und Rückspiel"], "judge": C.COLORS["Auftrittsreihenfolge"], "shoot": C.COLORS["Elfmeterschießen"]}
    dashes = ["solid", "dash", "dot", "dashdot"]
    counters: dict = {}
    for name, (family, rows) in series.items():
        pts = [(100 * r["delta"], n, r["x"]) for r, n in zip(rows, rows_n[name]) if r["delta"] > 0 and not math.isinf(n)]
        if not pts:
            continue
        k = counters.get(family, 0)
        counters[family] = k + 1
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=name, line=dict(color=palette[family], dash=dashes[k % 4]),
                                 customdata=[p[2] for p in pts], hovertemplate=f"{name}<br>Δ = %{{x:.2f}} Pp<br>N = %{{y:,.0f}}<br>Stärke = %{{customdata}}<extra></extra>"))
    for level in (10, 100, 1000, 10000):
        fig.add_hline(y=level, line=dict(color="rgba(120,120,120,0.35)", width=1, dash="dot"), annotation_text=f"{level:,}".replace(",", "."), annotation_position="right")
    fig.update_xaxes(type="log", title_text="Lagenunterschied Δ (Prozentpunkte, logarithmische Achse)")
    fig.update_yaxes(type="log", title_text="beobachtete Wettbewerbe N je Lage")
    fig.update_layout(legend=dict(orientation="h", y=-0.35))
    return _base(fig, 470)
