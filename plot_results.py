from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


project_dir = Path(__file__).resolve().parent
results_dir = project_dir / "results"
data = pd.read_csv(results_dir / "test_predictions.csv")

actual = data["is_suspicious"]
predicted = data["prediction"]

tp = int(((actual == 1) & (predicted == 1)).sum())
fn = int(((actual == 1) & (predicted == 0)).sum())
fp = int(((actual == 0) & (predicted == 1)).sum())

total_suspicious = tp + fn
total_alerts = tp + fp

if total_suspicious == 0 or total_alerts == 0:
    raise ValueError(
        "Für diese Darstellung werden Verdachtsfälle "
        "und mindestens eine Warnung benötigt."
    )

recall = 100 * tp / total_suspicious
precision = 100 * tp / total_alerts

green = "#267865"
orange = "#F0B078"
dark = "#20332E"


def percent(value):
    return f"{value:.1f}".replace(".", ",") + " %"


fig, axes = plt.subplots(2, 1, figsize=(12, 6.5))
fig.subplots_adjust(
    left=0.06, right=0.97, top=0.76, bottom=0.18, hspace=1.2
)

panels = [
    (
        axes[0],
        "Wie viele Verdachtsfälle finden wir?",
        f"Basis: {total_suspicious} eingebaute Verdachtsfälle",
        recall,
        f"{percent(recall)} erkannt\n{tp} Fälle",
        f"{percent(100 - recall)} übersehen\n{fn} Fälle",
    ),
    (
        axes[1],
        "Wie verlässlich sind unsere Warnungen?",
        f"Basis: {total_alerts} ausgegebene Warnungen",
        precision,
        f"{percent(precision)} berechtigt\n{tp} Warnungen",
        f"{percent(100 - precision)} Fehlalarme\n{fp} Warnungen",
    ),
]

for ax, title, subtitle, value, left_label, right_label in panels:
    ax.barh(0, value, color=green, height=0.65)
    ax.barh(
        0, 100 - value, left=value,
        color=orange, height=0.65
    )

    ax.text(
        value / 2, 0, left_label,
        ha="center", va="center",
        color="white", fontsize=14, fontweight="bold",
    )
    ax.text(
        value + (100 - value) / 2, 0, right_label,
        ha="center", va="center",
        color=dark, fontsize=14, fontweight="bold",
    )

    ax.text(
        0, 1.42, title, transform=ax.transAxes,
        fontsize=15, fontweight="bold", color=dark,
    )
    ax.text(
        0, 1.12, subtitle, transform=ax.transAxes,
        fontsize=11, color="#555555",
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")

fig.suptitle(
    "Das Modell erkennt 6 von 10 Verdachtsfällen",
    x=0.06, y=0.96, ha="left",
    fontsize=21, fontweight="bold", color=dark,
)

fig.text(
    0.06, 0.08,
    "Random Forest | 1.000 synthetische Testtransaktionen"
    " | Warnschwelle: 0,5",
    fontsize=11, color="#555555",
)
fig.text(
    0.06, 0.035,
    "Bewertung anhand künstlicher Labels; "
    "kein Nachweis der Leistung bei echter Geldwäsche.",
    fontsize=10, color="#555555",
)

output_path = results_dir / "model_results.png"
fig.savefig(output_path, dpi=200, facecolor="white")
plt.close(fig)

print(f"Grafik gespeichert: {output_path}")