"""Draw docs/architecture.png."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from riskengine.config import ROOT

INK, INK_2, LINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
FILL = {"source": "#eef4fd", "engine": "#ffffff", "store": "#fdf1ea", "app": "#eaf7f1", "offline": "#f4f3ef"}
EDGE = {"source": "#2a78d6", "engine": "#0b0b0b", "store": "#eb6834", "app": "#1baf7a", "offline": "#898781"}


def box(ax, x, y, w, h, title, lines=(), kind="engine", title_size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12", fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    top = y + h - 0.2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 0.36 - 0.27 * i, line, ha="center", va="top", fontsize=8.6, color=INK_2)


def arrow(ax, a, b, label="", rad=0.0, label_dy=0.14):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=LINE, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + label_dy, label, ha="center", va="bottom", fontsize=8.2, color=INK_2, style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.3, 8.62, "AI/NLP Financial Risk Engine: architecture and data flow", fontsize=15, fontweight="bold", color=INK)

    # 1. sources
    ax.text(0.3, 7.95, "1  SOURCES", fontsize=9, color=LINE, fontweight="bold")
    box(ax, 0.3, 6.0, 2.6, 1.75, "News", ["Google News RSS", "Yahoo Finance RSS", "GDELT DOC API", "Replay: 2020 headlines"], "source")
    box(ax, 0.3, 3.95, 2.6, 1.75, "Social media", ["StockTwits streams", "Replay: 2020 stock tweets", "(sampling weights keep", "true post volume)"], "source")

    # 2. ingestion
    ax.text(3.5, 7.95, "2  INGESTION", fontsize=9, color=LINE, fontweight="bold")
    box(ax, 3.5, 4.6, 2.0, 2.5, "Source adapters", ["one adapter per feed", "common Document", "schema: time, source,", "publisher, text, weight", "", "failures isolated", "per source"], "source")
    arrow(ax, (2.9, 6.85), (3.5, 6.2))
    arrow(ax, (2.9, 4.85), (3.5, 5.4))

    # 3. engine
    ax.text(6.1, 7.95, "3  RISK ENGINE (one pass per day of documents)", fontsize=9, color=LINE, fontweight="bold")
    ax.add_patch(FancyBboxPatch((6.0, 3.35), 5.25, 4.45, boxstyle="round,pad=0.0,rounding_size=0.15", fc="#f7f7f5", ec=LINE, lw=1.0, ls="-"))
    box(ax, 6.2, 6.3, 2.35, 1.3, "Clean and filter", ["de-duplicate, drop", "boilerplate and spam"])
    box(ax, 8.7, 6.3, 2.35, 1.3, "Entity linking", ["company and sector, or", "MARKET for macro text"])
    box(ax, 6.2, 4.85, 2.35, 1.3, "Sentiment", ["FinBERT", "score in [-1, 1]"])
    box(ax, 8.7, 4.85, 2.35, 1.3, "Event classification", ["embeddings + TF-IDF,", "11 event types"])
    box(ax, 6.2, 3.5, 2.35, 1.2, "Attention", ["volume vs the stream's", "own 20-day norm"])
    box(ax, 8.7, 3.5, 2.35, 1.2, "Impact score 1-10", ["additive, calibrated on", "realised price moves"])
    arrow(ax, (5.5, 5.85), (6.2, 6.8))
    arrow(ax, (8.55, 6.95), (8.7, 6.95))
    arrow(ax, (9.875, 6.3), (9.875, 6.15))
    arrow(ax, (8.7, 5.5), (8.55, 5.5))
    arrow(ax, (7.375, 4.85), (7.375, 4.7))
    arrow(ax, (8.55, 4.1), (8.7, 4.1))

    # 4. store
    ax.text(11.85, 7.95, "4  SIGNALS", fontsize=9, color=LINE, fontweight="bold")
    box(ax, 11.85, 5.2, 1.75, 2.4, "Signal store", ["SQLite", "+ CSV export", "", "ticker, sentiment,", "event type, impact,", "impact drivers"], "store")
    arrow(ax, (11.05, 4.3), (11.85, 5.6), rad=-0.15)
    box(ax, 11.85, 3.5, 1.75, 1.3, "REST API", ["FastAPI: /signals,", "/analyze, /stress"], "store")
    arrow(ax, (12.725, 5.2), (12.725, 4.8))

    # 5. applications
    ax.text(14.1, 7.95, "5  APPLICATIONS", fontsize=9, color=LINE, fontweight="bold")
    box(ax, 14.1, 6.05, 1.65, 1.6, "Module A", ["index", "rebalancer", "(sentiment)"], "app")
    box(ax, 14.1, 4.25, 1.65, 1.6, "Module B", ["stress testing", "(event type,", "impact)"], "app")
    arrow(ax, (13.6, 6.7), (14.1, 6.85))
    arrow(ax, (13.6, 6.0), (14.1, 5.2))
    box(ax, 11.85, 1.55, 3.9, 1.5, "Streamlit dashboard", ["signal feed with score explanations,", "headline analyser, index weights and backtest,", "stress results, model validation"], "app")
    arrow(ax, (14.925, 4.25), (14.6, 3.05))
    arrow(ax, (12.725, 3.5), (12.725, 3.05))

    # offline
    ax.text(0.3, 3.0, "OFFLINE: TRAINING, CALIBRATION AND EVALUATION", fontsize=9, color=LINE, fontweight="bold")
    box(ax, 0.3, 1.05, 2.9, 1.7, "Labelled public text", ["financial tweet topics (events)", "financial tweet sentiment", "Financial PhraseBank"], "offline")
    box(ax, 3.5, 1.05, 2.9, 1.7, "Model training and tests", ["event classifier vs keyword rules", "FinBERT vs lexicon baseline", "held-out splits only"], "offline")
    box(ax, 6.7, 1.05, 2.3, 1.7, "Market data", ["yfinance daily prices", "abnormal returns vs SPY", "in volatility units"], "offline")
    box(ax, 9.3, 1.05, 2.2, 1.7, "Impact calibration", ["fit on 2015-2018,", "test on 2019", "35k headlines"], "offline")
    arrow(ax, (3.2, 1.9), (3.5, 1.9))
    arrow(ax, (9.0, 1.9), (9.3, 1.9))
    arrow(ax, (4.95, 2.75), (7.2, 3.35))
    ax.text(6.55, 2.86, "trained models", fontsize=8.2, color=INK_2, style="italic")
    arrow(ax, (10.4, 2.75), (10.0, 3.5))
    ax.text(10.45, 3.02, "coefficients", fontsize=8.2, color=INK_2, style="italic")
    box(ax, 0.3, 0.15, 11.2, 0.6, "Module A also reads prices for its backtest; Module B reads a synthetic wholesale portfolio of loans, bonds and derivatives.", (), "offline", title_size=8.8)

    out = ROOT / "docs" / "architecture.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
