#!/usr/bin/env bash
# Wendet lokale Fixes auf installierte Pakete an, die mit der gepinnten
# brax/mujoco-Kombination nicht kompatibel sind (mjx.ncon existiert in
# mujoco 3.2.6 nicht mehr).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAX_DIR="$(python -c "import brax, os; print(os.path.dirname(brax.__file__))")"

patch --forward -p1 -d "$BRAX_DIR/.." < "$SCRIPT_DIR/brax_contact_mjx_ncon.patch" \
  || echo "Patch bereits angewendet oder brax-Version stimmt nicht ueberein - bitte pruefen."
