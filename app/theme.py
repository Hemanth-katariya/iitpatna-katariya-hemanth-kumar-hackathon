"""Chart styling shared by every dashboard page: one palette, one set of mark rules."""
from __future__ import annotations

import plotly.graph_objects as go

# Categorical slots, used in this fixed order and never cycled (colour-blind-safe as ordered).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#c3c2b7"
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]  # negative red, neutral grey, positive blue

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Entities keep their colour everywhere they appear.
SECTOR_COLOR = dict(
    zip(
        [
            "Information Technology", "Communication Services", "Consumer Discretionary", "Financials",
            "Health Care", "Energy", "Industrials", "Consumer Staples",
        ],  # fmt: skip
        SERIES,
    )
)


def style(fig: go.Figure, height: int = 340, legend: bool = True, y_title: str = "", x_title: str = "") -> go.Figure:
    axis = dict(
        gridcolor=GRID, gridwidth=1, linecolor=AXIS, zeroline=False, ticks="", automargin=True,
        tickfont=dict(color=MUTED, size=12), title_font=dict(color=INK_2, size=12),
    )  # fmt: skip
    fig.update_layout(
        template="none",
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        margin=dict(l=56, r=24, t=36 if legend else 16, b=44),
        colorway=SERIES,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2, size=12)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, color=INK, size=12)),
        hovermode="x unified",
        bargap=0.25,
    )
    fig.update_xaxes(**axis, showgrid=False, title_text=x_title)
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9:
        return f"{sign}${x / 1e9:,.2f}bn"
    return f"{sign}${x / 1e6:,.1f}m"
