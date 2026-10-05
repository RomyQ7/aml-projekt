from pathlib import Path
from time import perf_counter, sleep

import pandas as pd
import requests


project_dir = Path(__file__).resolve().parent
results_dir = project_dir / "results"

# Testtransaktionen in zeitlicher Reihenfolge 
data = pd.read_csv(results_dir / "test_predictions.csv")
data = data.sort_values("timestamp").reset_index(drop=True)

api_url = "http://127.0.0.1:8000"
pause_seconds = 0.1
records = []

with requests.Session() as session:
    # Vor Beginn prüfen, ob API erreichbar ist.
    try:
        health = session.get(f"{api_url}/health", timeout=5)
        health.raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(
            f"API nicht erreichbar. Läuft der Server?\n{error}"
        )

    print(f"Starte Datenstrom mit {len(data)} Transaktionen.")

    for number, row in enumerate(data.itertuples(index=False), start=1):
        payload = {
            "transaction_id": row.transaction_id,
            "amount": float(row.amount),
            "new_recipient": int(row.new_recipient),
            "country_risk": float(row.country_risk),
        }

        record = {
            "transaction_id": row.transaction_id,
            "status_code": None,
            "success": False,
            "model_score": None,
            "alert": None,
            "matches_offline": None,
            "error": "",
        }

        start = perf_counter()

        try:
            response = session.post(
                f"{api_url}/predict",
                json=payload,
                timeout=10,
            )
            record["status_code"] = response.status_code
            response.raise_for_status()
            result = response.json()

            # Grundlegende Prüfung der API-Antwort.
            score = result["model_score"]
            alert = result["alert"]

            if (
                result["transaction_id"] != row.transaction_id
                or not isinstance(score, (int, float))
                or not 0 <= score <= 1
                or not isinstance(alert, bool)
            ):
                raise ValueError("Unerwartetes Antwortformat")

            record["model_score"] = score
            record["alert"] = alert
            record["matches_offline"] = (
                int(alert) == int(row.prediction)
                and abs(score - float(row.model_score)) < 1e-10
            )
            record["success"] = True

        except (requests.RequestException, ValueError, KeyError) as error:
            record["error"] = str(error)

        record["response_ms"] = (perf_counter() - start) * 1000
        records.append(record)

        if not record["success"]:
            print(f"{number}: Technischer Fehler: {record['error']}")
        elif record["alert"]:
            print(
                f"{number}: WARNUNG für {row.transaction_id}"
                f" | Modellwert: {record['model_score']:.3f}"
            )
        elif number % 100 == 0:
            print(f"{number} Transaktionen bearbeitet.")

        if number < len(data):
            sleep(pause_seconds)


# Alle Versuche einschließlich technischer Fehler speichern.
results = pd.DataFrame(records)
results.to_csv(results_dir / "stream_results.csv", index=False)

successful = results[results["success"]]
errors = len(results) - len(successful)

summary = {
    "requests": len(results),
    "successful": len(successful),
    "technical_errors": errors,
}

print("\n--- Zusammenfassung ---")
print(f"Gesendete Anfragen: {len(results)}")
print(f"Erfolgreich bewertet: {len(successful)}")
print(f"Technische Fehler: {errors}")

if len(successful) > 0:
    warnings = int(successful["alert"].sum())
    mismatches = int((successful["matches_offline"] == False).sum())
    mean_ms = float(successful["response_ms"].mean())
    p95_ms = float(successful["response_ms"].quantile(0.95))

    summary.update({
        "alerts": warnings,
        "offline_mismatches": mismatches,
        "mean_response_ms": mean_ms,
        "p95_response_ms": p95_ms,
    })

    print(f"Warnungen: {warnings}")
    print(f"Abweichungen vom Offline-Test: {mismatches}")
    print(f"Mittlere Antwortzeit: {mean_ms:.2f} ms")
    print(f"95. Perzentil der Antwortzeit: {p95_ms:.2f} ms")

pd.DataFrame([summary]).to_csv(
    results_dir / "stream_summary.csv", index=False
)

print("\nErgebnisse im Ordner results gespeichert.")