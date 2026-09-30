import json

import pandas as pd

import wandb

RUN_ID="lttpt72r"

api = wandb.Api()
run = api.run("c4cw5vfmtd-/deep_rl_decentralized_crl/" + RUN_ID)   # z.B. fdol7wi9

df = pd.DataFrame(run.scan_history())
df = df.drop(columns=[c for c in df.columns if c.startswith("vis")], errors="ignore")  # HTML-Visualisierung raus
df.to_csv(f"{run.id}_history.csv", index=False)

with open(f"{run.id}_config.json", "w") as f:        # Hyperparameter mit sichern
    json.dump(dict(run.config), f, indent=2)

print(df.shape)              # nur Größe anzeigen statt alles
print(df.columns.tolist())   # welche Metriken gibt es