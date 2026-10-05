from pathlib import Path

import numpy as np
import pandas as pd


# Fester Startwert: Bei jedem Durchlauf entstehen dieselben Daten.
rng = np.random.default_rng(42)

# Erster überschaubarer Datensatz
n_transactions = 5000

# Verschiedene Beträge
amounts = np.round(
    np.clip(
        rng.lognormal(mean=5.5, sigma=1.3, size=n_transactions),
        1,
        100000,
    ),
    2,
)

# Eine Zeile = eine künstlichen Überweisung
transactions = pd.DataFrame({
    "transaction_id": [
        f"TX{i:06d}" for i in range(1, n_transactions + 1)
    ],
    "timestamp": pd.date_range(
        start="2026-01-01",
        periods=n_transactions,
        freq="min",
        tz="UTC",
    ),
    "sender_id": [
        f"S{i:04d}"
        for i in rng.integers(1, 501, size=n_transactions)
    ],
    "receiver_id": [
        f"R{i:04d}"
        for i in rng.integers(1, 1001, size=n_transactions)
    ],
    "amount": amounts,
    "new_recipient": rng.binomial(
        n=1, p=0.2, size=n_transactions
    ),
    "country_risk": np.round(
        rng.beta(a=2, b=8, size=n_transactions), 3
    ),
})

# Datenordner neben Python-Datei erstellen.
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"
data_dir.mkdir(exist_ok=True)

# Ausgangsdaten speichern
output_path = data_dir / "transactions_base.csv"
transactions.to_csv(output_path, index=False)

# Kurze Kontrolle im Terminal
print(f"Gespeichert: {output_path}")
print(f"Anzahl Transaktionen: {len(transactions)}")
print("\nDie ersten fünf Transaktionen:")
print(transactions.head().to_string(index=False))
print("\nZusammenfassung der Beträge:")
print(transactions["amount"].describe().round(2))