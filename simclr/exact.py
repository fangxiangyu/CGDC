import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored

parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():
  p = create_config(args.config_env, args.config_exp)
  print(colored(p, 'red'))
  
  # Model
  print(colored('Retrieve model', 'blue'))
  model = get_model(p)
  print('Model is {}'.format(model.__class__.__name__))
  print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
  print(model)
  model = model.cuda()
  
  # CUDNN
  print(colored('Set CuDNN benchmark', 'blue')) 
  torch.backends.cudnn.benchmark = True
  
  # Dataset
  print(colored('Retrieve dataset', 'blue'))
  #train_transforms = get_train_transformations(p)
  #print('Train transforms:', train_transforms)
  model_dict = torch.load(p['pretext_model'])
  model.load_state_dict(model_dict)
  model.cuda()

  model.eval()
  
  val_transforms = get_val_transformations(p)
  print('Validation transforms:', val_transforms)
  #train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
  #                                    split='train+unlabeled') # Split is for stl-10
  val_dataset = get_val_dataset(p, val_transforms) 
  #train_dataloader = get_train_dataloader(p, train_dataset)
  val_dataloader = get_val_dataloader(p, val_dataset)
  #print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
  
  # Memory Bank
  #print(colored('Build MemoryBank', 'blue'))
  base_dataset = get_train_dataset(p, val_transforms, train='train+test') # Dataset w/o augs for knn eval
  base_dataloader = get_val_dataloader(p, base_dataset) 
  #base_dataset = get_val_dataset()
  labels = base_dataset.targets
  labels = np.array(labels)
  np.save('cifar10-labels.npy',labels)
  # val_labels = val_dataset.targets
  # val_labels = np.array(val_labels)
  # np.save('val-labels.npy',val_labels)
  
  print(len(base_dataset))
  batch_size = 512

  for i, batch in enumerate(base_dataloader):
    images = batch['image'].cuda(non_blocking=True)
    #targets = batch['target'].cuda(non_blocking=True)
    output = model(images).data.cpu().numpy()
    if i == 0:
      features = np.zeros((len(base_dataset),output.shape[1]),dtype='float32')
    if i < len(base_dataloader)-1:
      features[i * batch_size:(i+1)*batch_size] = output
    else:
      features[i * batch_size:] = output


  np.save('cifar10-features.npy',features)


if __name__ == '__main__':
  main()