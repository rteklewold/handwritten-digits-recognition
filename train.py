import argparse
from tensorboardX import SummaryWriter
import torch
import matplotlib.pyplot as plt

from dataset import build_dataset_and_loader
from model import MNISTClassifier
from utils import model_func_decorator, create_logger, plot_image, plot_value_array
from train_utils import train_model, load_checkpoint
from eval_classifier import compute_PR_curve


parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('model_cfg', type=int, help="list of number of neurons in each hidden layer. "
                                                "format: n_hidden_1,n_hidden_2 (e.g.: 128,256)")
parser.add_argument('--batch_size', type=int, default=512, help='size of a batch of data for training and testing')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--total_epoch', type=int, default=10, help='length of training')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--checkpoint', type=str, default=None, help='full path to checkpoint which you want to load')
parser.add_argument('--tensorboard_file', type=str, default='tb_log', help='name of file into which tensorboard data is saved')
parser.add_argument('--log_file', type=str, default='training_log', help='name of log file')
args = parser.parse_args()

batch_size = args.batch_size
model_cfg = args.model_cfg
start_epoch = args.start_epoch
total_epoch = args.total_epoch
lr = args.lr
weight_decay = args.weight_decay
checkpoint_file = args.checkpoint
tb_log_name = args.tensorboard_file
log_name = args.log_file

logger = create_logger(log_file=f'{log_name}.txt')

train_set, train_loader = build_dataset_and_loader(
    batch_size=batch_size,
    partition='training',
    logger=logger,
    data_dir="./data"
)

model = MNISTClassifier(cfg=model_cfg)
logger.info(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

tb_log = SummaryWriter(log_dir=f"./output/tensorboard/{tb_log_name}")

# load checkpoint if it is provided to continue training
accumulated_iter = 0
if checkpoint_file is not None:
    logger.info(f"Load checkpoint: {checkpoint_file}")
    start_epoch, accumulated_iter = load_checkpoint(checkpoint_file, model, optimizer)

if start_epoch < total_epoch - 1:
    logger.info("-------------------- Start Training --------------------")
    train_model(
        model,
        model_func_decorator,
        optimizer,
        train_loader,
        start_epoch=start_epoch,
        total_epochs=total_epoch,
        start_iter=accumulated_iter,
        ckpt_save_dir='./output',
        tb_log=tb_log
    )
    logger.info("-------------------- Training Finishes --------------------")

# eval
logger.info("-------------------- Start Eval --------------------")
test_set, test_loader = build_dataset_and_loader(
    batch_size=batch_size,
    partition='testing',
    logger=logger,
    data_dir="./data"
)

if checkpoint_file is None:
    checkpoint_file = f"./output/checkpoint_epoch_{total_epoch}.pth"
logger.info(f"Eval checkpoit: {checkpoint_file}")

model.eval()
load_checkpoint(checkpoint_file, model)

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
test_loader_iter = iter(test_loader)
batch = next(test_loader_iter)
with torch.no_grad():
    ret_dict = model_func_decorator(model, batch)
predictions = torch.softmax(ret_dict['logits'], dim=1)
predictions = predictions[:num_images].numpy()

test_images = batch['image'][:num_images].reshape(num_images, 28, 28)
test_labels = batch['label'][:num_images]

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

p_list, r_list, acc_list = compute_PR_curve(model, test_loader, logger, nthreshold=10)
fig, axes = plt.subplots(1, 2)
axes[0].plot(r_list, p_list)
axes[0].set_xlabel("recall")
axes[0].set_ylabel("precision")
axes[0].grid()

axes[1].plot(r_list, acc_list)
axes[1].set_xlabel("recall")
axes[1].set_ylabel("accuracy")
axes[1].grid()

plt.show()
