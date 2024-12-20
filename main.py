import os
import argparse

from torch.utils import data
from dataloader import MtatDataset
from solver import Solver

def main(config):
    # path for models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    # audio length
    if config.model_type == 'fcn' or config.model_type == 'crnn':
        config.input_length = 29 * 16000
    elif config.model_type == 'musicnn':
        config.input_length = 3 * 16000
    elif config.model_type in ['sample', 'se', 'short', 'short_res']:
        config.input_length = 59049
    elif config.model_type == 'hcnn':
        config.input_length = 80000
    elif config.model_type == 'vit':
        config.input_length = 15 * 16000
    # get data loder
    train_loader = data.DataLoader(
        dataset = MtatDataset(split='TRAIN', input_length=config.input_length, aug=config.aug,aug_prob=config.aug_prob),
	    batch_size = config.batch_size,
	    shuffle = True,
	    drop_last = False,
		num_workers = config.num_workers
    )

    solver = Solver(train_loader, config)
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='fcn',
						choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'hcnn', 'vit'])
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--aug', type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--aug_prob', type=float, default=0.5)
    config = parser.parse_args()

    print(config)
    main(config)
