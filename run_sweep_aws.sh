#!/usr/bin/env bash
# Startet einen wandb-Sweep-Agent auf AWS und faehrt die Instanz danach automatisch
# herunter -- egal ob alle Laeufe durch sind oder der Agent mit einem Fehler abbricht.
# Der Agent holt sich nacheinander offene Konfigurationen des Sweeps. Mehrere Instanzen
# koennen parallel denselben Sweep abarbeiten.
#
# Nutzung (am besten in tmux):
#   wandb sweep sweeps/critic.yaml                      # einmalig, gibt die Sweep-ID aus
#   bash run_sweep_aws.sh <entity>/deep_rl_sweep/<sweep_id>
#   COUNT=4 bash run_sweep_aws.sh <sweep>               # nur 4 Laeufe auf dieser Instanz
#   MAX_HOURS=24 bash run_sweep_aws.sh <sweep>          # Notbremse nach 24h statt 72h
#
# Herunterfahren abbrechen:  sudo shutdown -c
set -u
cd "$(dirname "$0")"
source venv/bin/activate

if [ $# -lt 1 ]; then
  echo "Nutzung: bash run_sweep_aws.sh <entity>/<project>/<sweep_id>"
  exit 1
fi
SWEEP_ID="$1"

# GPU-Check: ohne GPU gar nicht erst starten (sonst laeuft alles langsam auf der CPU)
if ! python -c "import jax, sys; sys.exit(0 if jax.default_backend() == 'gpu' else 1)" 2>/dev/null; then
  echo "[run_sweep_aws] FEHLER: JAX findet keine GPU – Sweep wird NICHT gestartet."
  echo "[run_sweep_aws] Pruefen mit: python -c 'import jax; print(jax.devices())'"
  exit 1
fi

MAX_HOURS=${MAX_HOURS:-72}
COUNT_ARGS=()
if [ -n "${COUNT:-}" ]; then
  COUNT_ARGS=(--count "$COUNT")
fi
mkdir -p logs
LOG="logs/sweep_$(date +%Y%m%d_%H%M%S).log"

# Notbremse: spaetestens nach MAX_HOURS herunterfahren, falls etwas haengt
sudo shutdown -h +$((MAX_HOURS * 60)) "run_sweep_aws.sh: maximale Laufzeit erreicht" 2>/dev/null
echo "[run_sweep_aws] Sweep $SWEEP_ID. Notbremse: Herunterfahren spaetestens in ${MAX_HOURS}h. Log: $LOG"

wandb agent "${COUNT_ARGS[@]}" "$SWEEP_ID" 2>&1 | tee "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[run_sweep_aws] Sweep-Agent ERFOLGREICH beendet." | tee -a "$LOG"
else
  echo "[run_sweep_aws] Sweep-Agent mit FEHLER beendet (Exit-Code $EXIT_CODE)." | tee -a "$LOG"
fi

# 2 Minuten Puffer, damit wandb fertig hochladen kann (und du ggf. abbrechen kannst)
echo "[run_sweep_aws] Instanz faehrt in 2 Minuten herunter (abbrechen: sudo shutdown -c)." | tee -a "$LOG"
sudo shutdown -c 2>/dev/null
sudo shutdown -h +2 "run_sweep_aws.sh: Sweep beendet (Exit-Code $EXIT_CODE)"
