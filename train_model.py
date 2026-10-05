from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"
model_dir = project_dir / "models"
results_dir = project_dir / "results"

model_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

data = pd.read_csv(data_dir / "transactions_labeled.csv")

# Drei Merkmale für Modell als Eingabe.
features = ["amount", "new_recipient", "country_risk"]

# Ganze Zeilen aufteilen, damit IDs und Labels erhalten bleiben.
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["is_suspicious"],
)

X_train = train_data[features]
y_train = train_data["is_suspicious"]
X_test = test_data[features]
y_test = test_data["is_suspicious"]

print(f"Training: {len(train_data)} Transaktionen")
print(f"Test: {len(test_data)} Transaktionen")
print(f"Verdachtsfälle im Test: {int(y_test.sum())}")

# Referenz: Immer die häufigste Klasse vorhersagen.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

# Vorab festgelegte Modelleinstellungen.
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Modellwert für Klasse 1 bestimmen.
positive_column = list(model.classes_).index(1)
scores = model.predict_proba(X_test)[:, positive_column]

# Festgelegter Grenzwert für eine Warnmeldung.
threshold = 0.5
predictions = (scores >= threshold).astype(int)


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


comparison = pd.DataFrame({
    "Immer legitim": evaluate(y_test, baseline_predictions),
    "Random Forest": evaluate(y_test, predictions),
}).T

print("\nVergleich auf den Testdaten:")
print(comparison.round(3).to_string())

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

print("\nRandom Forest im Detail:")
print(f"Richtig als legitim erkannt: {tn}")
print(f"Fehlalarme:                  {fp}")
print(f"Übersehene Verdachtsfälle:   {fn}")
print(f"Erkannte Verdachtsfälle:     {tp}")

# Modell und Angaben für die spätere API zusammen speichern.
joblib.dump(
    {
        "model": model,
        "features": features,
        "threshold": threshold,
    },
    model_dir / "aml_model.joblib",
)

# Ergebnisse für Grafiken und Präsentation speichern.
comparison.to_csv(results_dir / "metrics.csv", index_label="model")

test_results = test_data.copy()
test_results["model_score"] = scores
test_results["prediction"] = predictions
test_results.to_csv(
    results_dir / "test_predictions.csv", index=False
)

# Aufteilung für spätere Nachvollziehbarkeit dokumentieren.
split_ids = {
    "train_ids": train_data["transaction_id"].tolist(),
    "test_ids": test_data["transaction_id"].tolist(),
}
with open(results_dir / "split_ids.json", "w", encoding="utf-8") as file:
    json.dump(split_ids, file, indent=2)

print("\nModell und Ergebnisse gespeichert.")