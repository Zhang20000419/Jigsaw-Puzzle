from template.losses.Loss import LitCrossEntropyLoss
from template.models.Model import LitNet
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from template.datasets.Dataset import JigsawDataset2
from datetime import datetime


# config
config_version = 'v1'
config_save_path = './config/saved_configs'
config_save_name = 'config_normal_{}'.format(config_version)

# data_paths
data_train_path = './data/puzzle_2x2/train'
csv_train_path = './data/puzzle_2x2/train.csv'
data_test_path = './data/puzzle_2x2/test'
csv_test_path = './data/puzzle_2x2/test.csv'
data_valid_path = './data/puzzle_2x2/valid'
csv_valid_path = './data/puzzle_2x2/valid.csv'

# checkpoint_callback settings
monitor = 'val_loss'
checkpoint_filename = 'model_{epoch:02d}-{val_loss:.2f}'
checkpoint_save_path = './checkpoints/{}'.format(config_save_name)
save_top_k = 1
mode = 'min'
save_last = True

# trainer settings
gpus = [0]
max_epochs = 16
accelerator = 'gpu'
check_val_every_n_epoch = 2
strategy = 'auto'
logger = False

# hyperparameters
lr = 0.001
batch_size = 512
num_labels = 4

# model
net = LitNet(num_labels)

# loss
loss = LitCrossEntropyLoss(num_labels)

# optimizer
optimizer = SGD(net.parameters(), lr=lr)

# dataloader
dataset_train = JigsawDataset2(data_train_path, csv_train_path)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=32)

dataset_test = JigsawDataset2(data_test_path, csv_test_path)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=32)

dataset_valid = JigsawDataset2(data_valid_path, csv_test_path)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=32)






