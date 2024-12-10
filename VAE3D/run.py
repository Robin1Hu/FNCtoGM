import os
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from models import *
from experiment import VAEXperiment
from dataset import VAEDataset
import torch
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vq_vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Set up logging and checkpoint directories
log_dir = os.path.join(config['logging_params']['save_dir'], config['model_params']['name'])
Path(f"{log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

model = vae_models[config['model_params']['name']](**config['model_params'])
data = VAEDataset(**config["data_params"])
train_loader = DataLoader(data.train_dataset, batch_size=config['trainer_params']['batch_size'], shuffle=True,
                          pin_memory=True)
val_loader = DataLoader(data.val_dataset, batch_size=config['trainer_params']['batch_size'], shuffle=False,
                        pin_memory=True)

experiment = VAEXperiment(model, config['exp_params'], data)

# Set up optimizers and schedulers
optimizer = torch.optim.Adam(model.parameters(), lr=config['trainer_params']['lr'])
scheduler = StepLR(optimizer, step_size=config['trainer_params']['lr_step_size'],
                   gamma=config['trainer_params']['lr_gamma'])


def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


# Training loop
for epoch in range(config['trainer_params']['max_epochs']):
    print(f"======= Training Epoch {epoch} =======")
    experiment.run_training_epoch(train_loader)
    experiment.run_validation_epoch(val_loader)
    scheduler.step()
    experiment.sample_images()

    # Save checkpoints
    checkpoint_path = os.path.join(log_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
    save_checkpoint(epoch, model, optimizer, checkpoint_path)

print("Training completed.")
