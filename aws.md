SSH

```bash
ssh -i ~/.ssh/aws-ma.pem ubuntu@<PUBLIC-IP>
```

Setup
```bash
nvidia-smi                                   # GPU (Tesla T4) sichtbar?

sudo apt update && sudo apt install -y python3.12-venv tmux
git clone https://github.com/darnielorvw/Masterarbeit.git deep_rl
cd deep_rl
python3 -m venv venv && source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
bash patches/apply_patches.sh

python -c "import jax; print(jax.devices())"   # muss cuda(id=0) zeigen
wandb login    
```

Training run
```bash
tmux new -s train
cd ~/deep_rl && git pull
bash run_aws.sh --total-env-steps 10000000 --seed 42
```
Hyperparameter-Sweep (Width x Depth)
```bash
tmux new -s sweep
cd ~/deep_rl && git pull
wandb sweep sweeps/actor.yaml               # gibt <entity>/deep_rl_sweep/<sweep_id> aus
bash run_sweep_aws.sh <entity>/deep_rl_sweep/<sweep_id>
# parallel auf weiteren Instanzen denselben Befehl starten -> Laeufe werden verteilt
# danach: beste Critic-Konfig in sweeps/actor.yaml eintragen und dasselbe mit actor.yaml
```

````bash
tmux ls
tmux kill-session -t sweep
tmux kill-server
 #shutdown abbrechen
sudo shutdown -c


```



