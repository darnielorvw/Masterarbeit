# Deep RL — Decentralized Contrastive RL für Quantruped

Dezentrales Contrastive-RL-Training (unabhängige Actor/Alpha pro Bein, ein globaler Critic)
für einen vierbeinigen Roboter (`Quantruped`), implementiert in JAX/Flax mit
Brax/MuJoCo als Physik-Simulation.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
bash patches/apply_patches.sh   # notwendiger Fix für brax 0.10.1 + mujoco 3.2.6
```

Für GPU-Training (empfohlen, z.B. auf AWS) `jax`/`jaxlib` stattdessen mit
CUDA-Support installieren:

```bash
pip install "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Training starten

```bash
python train.py
```

Alle Argumente werden über [tyro](https://github.com/brentyi/tyro) als CLI-Flags
bereitgestellt, z.B.:

```bash
python train.py --total-env-steps 5000000 --num-envs 256 --seed 42 --wandb-mode online
```

Hilfe zu allen verfügbaren Flags: `python train.py --help`

## Argumente

### Logging & Tracking

| Argument | Default | Beschreibung |
|---|---|---|
| `--exp-name` | `train` | Freier Name für den Experiment-Lauf (aktuell nur informativ, fließt nicht in `run_name` ein). |
| `--seed` | `1000` | Zufalls-Seed für Python/NumPy/JAX — steuert Env-Initialisierung, Netzwerk-Init und Sampling. |
| `--track` / `--no-track` | `True` | Aktiviert Weights & Biases Logging. Bei `False` läuft das Training ohne wandb. |
| `--wandb-project-name` | `deep_rl_decentralized_crl` | Name des wandb-Projekts, in dem der Run erscheint. |
| `--wandb-entity` | `""` (leer) | wandb-Entity (Team/User). Leer = dein Standard-Account. |
| `--wandb-mode` | `offline` | `online`, `offline` oder `disabled`. `offline` speichert Logs nur lokal (für Cluster ohne Internetzugang gedacht, muss später per `wandb sync` hochgeladen werden); auf AWS mit Internetzugang meist `online` sinnvoll. |
| `--wandb-dir` | `.` | Verzeichnis, in dem wandb seine lokalen Dateien ablegt. |
| `--wandb-group` | `.` | Gruppiert mehrere Runs in wandb (z.B. für Seed-Sweeps). `.` = keine Gruppe. |
| `--capture-vis` / `--no-capture-vis` | `True` | Rendert am Ende des Trainings Rollouts der gelernten Policy als HTML-Visualisierung (`vis.html`) und loggt sie zu wandb. |
| `--vis-length` | `1000` | Anzahl Simulationsschritte pro aufgezeichnetem Rollout in der Visualisierung. |
| `--checkpoint` / `--no-checkpoint` | `True` | Speichert Modell-Parameter (und optional den Replay-Buffer) während und am Ende des Trainings als Pickle-Dateien. |

### Environment

| Argument | Default | Beschreibung |
|---|---|---|
| `--episode-length` | `1000` | Maximale Anzahl Simulationsschritte pro Episode, bevor sie zurückgesetzt wird. |

### Trainingsumfang

| Argument | Default | Beschreibung |
|---|---|---|
| `--total-env-steps` | `100000000` | Gesamtzahl an Environment-Steps über das gesamte Training (über alle parallelen Envs summiert). Bestimmt zusammen mit `num-epochs` die Trainingsdauer. |
| `--num-epochs` | `100` | Anzahl Trainings-Epochen. Nach jeder Epoche wird evaluiert/geloggt. |
| `--num-envs` | `512` | Anzahl parallel simulierter Environments (Batch-Parallelität in Brax). Höher = schnellerer Datendurchsatz, aber mehr GPU-Speicherbedarf. |
| `--num-eval-envs` | `128` | Anzahl parallel simulierter Environments für die Evaluation (getrennt von den Trainings-Envs). |

### Optimierung

| Argument | Default | Beschreibung |
|---|---|---|
| `--actor-lr` | `3e-4` | Lernrate für den Actor-Optimizer (Adam). |
| `--critic-lr` | `3e-4` | Lernrate für den Critic-Optimizer (State-Action- und Goal-Encoder). |
| `--alpha-lr` | `3e-4` | Lernrate für den Temperatur-Parameter `alpha` (Entropie-Gewichtung, SAC-artig). |
| `--batch-size` | `256` | Batch-Größe pro Gradientenschritt beim Sampling aus dem Replay-Buffer. |
| `--gamma` | `0.99` | Discount-Faktor für zukünftige Rewards/Goals. |
| `--logsumexp-penalty-coeff` | `0.1` | Gewichtung einer Regularisierungs-Penalty (`logsumexp²`) auf die Critic-Logits — stabilisiert das kontrastive Lernen. |

### Replay Buffer

| Argument | Default | Beschreibung |
|---|---|---|
| `--max-replay-size` | `10000` | Maximale Anzahl gespeicherter Trajektorien im Replay-Buffer. |
| `--min-replay-size` | `1000` | Mindestanzahl an Trajektorien, die vor dem ersten Trainingsschritt gesammelt werden müssen (Prefill-Phase). |

### Rollout / Netzwerk

| Argument | Default | Beschreibung |
|---|---|---|
| `--unroll-length` | `62` | Anzahl Environment-Steps, die pro Rollout am Stück gesammelt werden, bevor sie in den Replay-Buffer geschrieben werden. |
| `--critic-network-width` | `256` | Breite (Neuronen pro Layer) der Critic-Netzwerke (State-Action- und Goal-Encoder). |
| `--actor-network-width` | `256` | Breite der Actor-Netzwerke. |
| `--actor-depth` | `4` | Anzahl Hidden Layers im Actor-Netzwerk. |
| `--critic-depth` | `4` | Anzahl Hidden Layers in den Critic-Netzwerken. |
| `--num-sgd-batches-per-training-step` | `800` | Anzahl Gradientenschritte (Batches), die pro Trainings-Step aus dem Replay-Buffer gezogen und trainiert werden. |

### Sonstiges

| Argument | Default | Beschreibung |
|---|---|---|
| `--entropy-param` | `0.5` | Skaliert die Ziel-Entropie (`target_entropy = -entropy_param * 2`, da 2 Aktuatoren pro Bein) — steuert, wie explorativ die Policy sein soll. |
| `--disable-entropy` | `0` | Wenn `1`, wird der Entropie-Term im Actor-Loss deaktiviert (deterministischeres Training ohne Exploration-Bonus). |
| `--use-relu` | `0` | Wenn `1`, verwenden alle Netzwerke ReLU statt der Standard-Aktivierung. |
| `--num-render` | `10` | Anzahl Rollouts, die für die Visualisierung aufgezeichnet werden (nur relevant wenn `--capture-vis`). |
| `--save-buffer` | `0` | Wenn `1`, wird der komplette Replay-Buffer beim Checkpointing mit abgespeichert (kann sehr groß werden). |

> Die restlichen Felder (`env_steps_per_actor_step`, `num_prefill_env_steps`,
> `num_prefill_actor_steps`, `num_training_steps_per_epoch`) werden zur
> Laufzeit aus den obigen Argumenten berechnet und sollten nicht manuell
> gesetzt werden.

## Actor: Input und Output

Jedes der vier Beine (`FL`, `HL`, `HR`, `FR`) hat einen **eigenen, unabhängigen
Actor** (kein Parameter-Sharing) — es gibt also vier separate
Netzwerkinstanzen von `Actor` ([networks.py](networks.py)), die jeweils nur
lokale Informationen sehen (dezentrale Steuerung).

### Input

Die volle Environment-Observation ist `STATE_DIM (37) + GOAL_DIM (2) = 39`-dimensional
und wird zunächst in `state` (37-dim, Roboterzustand) und `goal` (2-dim,
Zielposition x/y) aufgeteilt ([train.py:241-242](train.py#L241-L242)):

```python
def split_obs(obs):
    return obs[..., :STATE_DIM], obs[..., STATE_DIM:STATE_DIM + GOAL_DIM]
```

Pro Bein wird daraus der **lokale Input** zusammengestellt
([train.py:248](train.py#L248)):

```python
actor_input = jnp.concatenate([local_state(state, leg), goal], axis=-1)
```

`local_state(state, leg)` ([leg_topology.py:80-82](leg_topology.py#L80-L82))
schneidet aus dem 37-dim `state` nur die für dieses Bein relevanten
31 Dimensionen heraus:

- **Globale Torso-Infos** (13 Dims): xyz-Position, Quaternion, lineare + angulare Geschwindigkeit
- **Eigenes Bein** (6 Dims): Gelenkwinkel (hip, knee), Gelenkgeschwindigkeiten, letzte Aktion
- **Beide Nachbarbeine im Ring** `FL↔HL↔HR↔FR↔FL` (2 × 6 = 12 Dims): dieselben drei Größen wie beim eigenen Bein

→ `LOCAL_STATE_DIM = 31` + `GOAL_DIM = 2` = **33-dimensionaler Input** pro
Actor-Aufruf ([leg_topology.py:59](leg_topology.py#L59)). Ein Bein bekommt
also nie den globalen Zustand aller vier Beine, sondern nur sich selbst, seine
zwei Nachbarn und die Torso-/Zielinfos.

### Output

`Actor.__call__` ([networks.py:99-125](networks.py#L99-L125)) gibt für
`action_size=2` (hip + knee) zwei 2-dimensionale Vektoren zurück:

```python
mean = nn.Dense(self.action_size, ...)(x)
log_std = nn.Dense(self.action_size, ...)(x)
log_std = nn.tanh(log_std)
log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
return mean, log_std
```

Das Netzwerk sagt also **Mittelwert und Log-Standardabweichung** einer
Gauß-Verteilung über die zwei Gelenk-Aktionen voraus (SAC-artiger stochastischer
Actor), `log_std` dabei auf `[-5, 2]` geclampt.

Wie daraus die finale Aktion wird, unterscheidet sich zwischen Training und
Evaluation:

- **Training** ([train.py:271-273](train.py#L271-L273)): es wird aus der
  Gauß-Verteilung gesampelt und danach durch `tanh` auf `[-1, 1]` gequetscht
  (reparametrisierter Sample-Trick, Standard bei SAC):
  ```python
  actions_per_leg[leg] = nn.tanh(means + stds * jax.random.normal(...))
  ```
- **Evaluation/Rendering** ([train.py:249-250](train.py#L249-L250)): es wird
  deterministisch nur der `tanh`-gequetschte Mittelwert genommen, kein Sampling:
  ```python
  actions_per_leg[leg] = nn.tanh(means)
  ```

Die vier 2-dim Bein-Aktionen werden anschließend über `assemble_action(...)`
([leg_topology.py:72-76](leg_topology.py#L72-L76)) zu einem **8-dimensionalen
Aktionsvektor** in der Reihenfolge der `<actuator>`-Einträge im MJCF
(`FR, FL, HL, HR`) zusammengesetzt und so an `env.step(...)` übergeben.

## Beispiel: kurzer Testlauf

```bash
python train.py \
  --total-env-steps 1000000 \
  --num-epochs 10 \
  --num-envs 64 \
  --no-track \
  --no-capture-vis
```

## Training auf AWS (EC2 GPU)

```bash
git clone https://github.com/darnielorvw/Masterarbeit.git
cd Masterarbeit/code/deep_rl
python3 -m venv venv && source venv/bin/activate
pip install -U pip
pip install "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt --no-deps
bash patches/apply_patches.sh
wandb login
tmux new -s train
source venv/bin/activate
python train.py --track --wandb-mode online && sudo shutdown -h now
# Ctrl+B, D zum Detachen
```
