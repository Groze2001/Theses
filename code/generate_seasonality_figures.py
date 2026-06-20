"""
Generates the sector-stratified seasonality figures needed for the thesis.
Reads pre-computed CSVs from Theses/results/ and saves PNGs to Theses/results/figures/.
Run with: python generate_seasonality_figures.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

BASE_PATH     = Path(r"C:\Users\GONCA\Desktop\Iscte\MCD\Theses")
METADATA_PATH = BASE_PATH / "Dataset" / "filtered_metadata.csv"
OUTPUT_DIR    = BASE_PATH / "results"
FIG_DIR       = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def kde_hist(ax, data, color, label=None, bins=30):
    data = data.dropna()
    ax.hist(data, bins=bins, density=True, alpha=0.45, color=color, label=label)
    if len(data) > 3:
        kde = gaussian_kde(data)
        xs  = np.linspace(data.min(), data.max(), 300)
        ax.plot(xs, kde(xs), color=color, lw=2)


def load_all(pattern):
    files = sorted(OUTPUT_DIR.glob(pattern))
    if not files:
        print(f"  WARN No files for: {pattern}")
    return {f.stem.split("_")[-1]: pd.read_csv(f) for f in files}


def filter_by_ids(d, valid_ids, id_col="user"):
    out = {}
    for suffix, df in d.items():
        kept = df[df[id_col].isin(valid_ids)] if id_col in df.columns else df
        if not kept.empty:
            out[suffix] = kept
    return out


def get_top2_sectors():
    meta = pd.read_csv(METADATA_PATH)
    sectors = {
        s: set(g["user"])
        for s, g in meta.groupby("sector")
    }
    sectors = dict(sorted(sectors.items(), key=lambda x: len(x[1]), reverse=True))
    top2 = dict(list(sectors.items())[:2])
    for name, ids in top2.items():
        print(f"  Sector '{name}': {len(ids)} users")
    return top2


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  OK Saved: {path}")
    plt.close(fig)


def generate():
    daily_all  = load_all("seasonality_results_daily_*.csv")
    weekly_all = load_all("seasonality_results_weekly_*.csv")

    if not daily_all or not weekly_all:
        print("No seasonality CSVs found — aborting.")
        return

    sectors = get_top2_sectors()
    colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for sector_name, valid_ids in sectors.items():
        daily  = filter_by_ids(daily_all,  valid_ids)
        weekly = filter_by_ids(weekly_all, valid_ids)
        slug   = sector_name.lower().replace(" ", "_")
        suffixes = sorted(set(daily) | set(weekly))

        print(f"\n-- {sector_name} ({len(valid_ids)} users) --")

        # ── Daily KDE ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, s in enumerate(suffixes):
            c = colors[i % len(colors)]
            if s in daily:
                vals = daily[s]["seasonal_strength_daily"]
                kde_hist(ax, vals, c, label=s)
                ax.axvline(vals.median(), color=c, ls="--", lw=1.2)
        ax.set_title(f"Daily Seasonal Strength — {sector_name}", fontsize=12)
        ax.set_xlabel("Seasonal Strength ($F_s$)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        plt.tight_layout()
        save_fig(fig, f"1a_{slug}_daily")

        # ── Weekly KDE ───────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, s in enumerate(suffixes):
            c = colors[i % len(colors)]
            if s in weekly:
                vals = weekly[s]["seasonal_strength_weekly"]
                kde_hist(ax, vals, c, label=s)
                ax.axvline(vals.median(), color=c, ls="--", lw=1.2)
        ax.set_title(f"Weekly Seasonal Strength — {sector_name}", fontsize=12)
        ax.set_xlabel("Seasonal Strength ($F_s$)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        plt.tight_layout()
        save_fig(fig, f"1a_{slug}_weekly")

        # ── Boxplot daily vs weekly across periods ───────────────────────
        records = []
        for s in suffixes:
            for freq, dct, col in [("Daily", daily, "seasonal_strength_daily"),
                                   ("Weekly", weekly, "seasonal_strength_weekly")]:
                if s in dct:
                    for v in dct[s][col].dropna():
                        records.append({"period": s, "freq": freq, "Fs": v})
        if records:
            df_box = pd.DataFrame(records)
            groups = df_box.groupby(["period", "freq"])["Fs"].apply(list)
            labels = [f"{p}\n{f}" for p, f in groups.index]
            fig, ax = plt.subplots(figsize=(max(6, 3 * len(labels)), 5))
            bp = ax.boxplot(list(groups), patch_artist=True)
            for patch, c in zip(bp["boxes"], colors * 10):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Seasonal Strength ($F_s$)")
            ax.set_title(f"Seasonal Strength Boxplot — {sector_name}", fontsize=12)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            save_fig(fig, f"1b_{slug}_boxplot")

    print("\nDone. All figures generated.")


if __name__ == "__main__":
    generate()
