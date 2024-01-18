import wandb
import sys

run = wandb.init()
artifact = run.use_artifact(f'cubig_ai/AZOO/log:v{sys.argv[1]}', type='dataset')
artifact_dir = artifact.download()