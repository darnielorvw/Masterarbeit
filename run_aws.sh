#!/usr/bin/env bash
# Startet das Training auf AWS und faehrt die Instanz danach automatisch herunter –
# egal ob das Training erfolgreich durchlaeuft oder mit einem Fehler abbricht.
#
# Nutzung (am besten in tmux):
#   bash run_aws.sh --num-envs 1024 --seed 42
#   MAX_HOURS=12 bash run_aws.sh ...        # Notbremse nach 12h statt 24h
#
# Herunterfahren abbrechen:  sudo shutdown -c
# Ctrl+C im Training = manueller Abbruch -> KEIN automatisches Herunterfahren
# (die Notbremse nach MAX_HOURS bleibt aber aktiv).
set -u
cd "$(dirname "$0")"
source venv/bin/activate

# GPU-Check: ohne GPU gar nicht erst starten (sonst laeuft alles langsam auf der CPU)
if ! python -c "import jax, sys; sys.exit(0 if jax.default_backend() == 'gpu' else 1)" 2>/dev/null; then
  echo "[run_aws] FEHLER: JAX findet keine GPU – Training wird NICHT gestartet."
  echo "[run_aws] Pruefen mit: python -c 'import jax; print(jax.devices())'"
  exit 1
fi

MAX_HOURS=${MAX_HOURS:-24}
mkdir -p logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"

# Notbremse: spaetestens nach MAX_HOURS herunterfahren, falls etwas haengt
sudo shutdown -h +$((MAX_HOURS * 60)) "run_aws.sh: maximale Laufzeit erreicht" 2>/dev/null
echo "[run_aws] Notbremse: Herunterfahren spaetestens in ${MAX_HOURS}h. Log: $LOG"

python train.py --wandb-mode online "$@" 2>&1 | tee "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[run_aws] Training ERFOLGREICH beendet." | tee -a "$LOG"
else
  echo "[run_aws] Training mit FEHLER beendet (Exit-Code $EXIT_CODE)." | tee -a "$LOG"
fi

# 2 Minuten Puffer, damit wandb fertig hochladen kann (und du ggf. abbrechen kannst)
echo "[run_aws] Instanz faehrt in 2 Minuten herunter (abbrechen: sudo shutdown -c)." | tee -a "$LOG"
sudo shutdown -c 2>/dev/null
sudo shutdown -h +2 "run_aws.sh: Training beendet (Exit-Code $EXIT_CODE)"
