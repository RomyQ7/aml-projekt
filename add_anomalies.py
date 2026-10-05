from pathlib import Path

import numpy as np
import pandas as pd


rng = np.random.default_rng(43)

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"

# Zzuvor erzeugte Ausgangsdaten laden
transactions = pd.read_csv(data_dir / "transactions_base.csv")

# Ausgangspunkt Simulation
transactions["is_suspicious"] = 0

# 500 verschiedene Zeilen zufällig auswählen,
# dadurch liegen Verdachtsfälle verteilt im Datensatz.
selected = rng.choice(
    transactions.index.to_numpy(),
    size=500,
    replace=False,
)

suspicious_high = selected[:125]
suspicious_risk = selected[125:250]
legitimate_high = selected[250:375]
legitimate_risk = selected[375:500]


# Muster 1: Höhere Beträge, viele neue Empfänger.
transactions.loc[suspicious_high, "amount"] = np.round(
    rng.lognormal(mean=8.5, sigma=0.7, size=125), 2
)
transactions.loc[suspicious_high, "new_recipient"] = (
    rng.binomial(1, 0.85, size=125)
)
transactions.loc[suspicious_high, "is_suspicious"] = 1


# Muster 2: Kleinere bis mittlere Beträge, erhöhtes
# fiktives Länderrisiko und viele neue Empfänger.
transactions.loc[suspicious_risk, "amount"] = np.round(
    rng.lognormal(mean=6.5, sigma=0.8, size=125), 2
)
transactions.loc[suspicious_risk, "country_risk"] = np.round(
    rng.beta(5, 2, size=125), 3
)
transactions.loc[suspicious_risk, "new_recipient"] = (
    rng.binomial(1, 0.85, size=125)
)
transactions.loc[suspicious_risk, "is_suspicious"] = 1


# Gegenbeispiel 1: Legitime Transaktionen mit hohen Beträgen.
transactions.loc[legitimate_high, "amount"] = np.round(
    rng.lognormal(mean=8.5, sigma=0.7, size=125), 2
)

# Gegenbeispiel 2: Legitime Transaktionen mit
# erhöhtem fiktiven Länderrisiko.
transactions.loc[legitimate_risk, "country_risk"] = np.round(
    rng.beta(5, 2, size=125), 3
)


# Als neue Datei speichern; Ausgangsdatei bleibt erhalten.
output_path = data_dir / "transactions_labeled.csv"
transactions.to_csv(output_path, index=False)

print(f"Gespeichert: {output_path}")
print(f"\nTransaktionen insgesamt: {len(transactions)}")
print("\nVerteilung der Labels:")
print(transactions["is_suspicious"].value_counts().sort_index())

print("\nDurchschnittliche Merkmale je Label:")
print(
    transactions.groupby("is_suspicious")[
        ["amount", "new_recipient", "country_risk"]
    ].mean().round(3)
)