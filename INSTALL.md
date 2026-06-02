# Install

```bash
# CPU-only (recommended — GPU version is several GB)
pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu --no-deps
pip install numpy"<2" pygame opencv-python wandb filelock sympy networkx typing-extensions
```

# Train

```bash
python train.py                       # grayscale (default), WandB on
python train.py --no-wandb            # no WandB
python train.py --mode threshold      # ablation: binary threshold
python train.py --mode rgb            # ablation: full colour
```

# Evaluate

```bash
python test.py                        # load latest checkpoint, headless
python test.py --episodes 20          # 20 evaluation episodes
python test.py --render               # show game window (needs display)
```

Training always resumes from the latest checkpoint in `checkpoints/`.
