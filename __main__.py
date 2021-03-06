import os
import sys
import random
import numpy as np
import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score
from nn_models import *
from dataloader_wav import WavDataset
from config import OUTPUT_DIR

# model_t = densenet201
model_t = resnet34
# model_t = AlexNet

def resume(save_path: str) -> dict or None:
    import glob
    checkpoints = glob.glob(os.path.join(save_path, 'checkpoint*'))
    if len(checkpoints) > 0:
        checkpoint_epochs = list(map(lambda x: int(x.split('/')[-1].split('_')[1]), checkpoints))
        ch = 'checkpoint_{}'.format(max(checkpoint_epochs))
        print("Using checkpoint {}".format(ch))
        return torch.load(os.path.join(save_path, ch))
    else:
        return None


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init():
    print("initializing")
    seed = 40
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # type
    # feature = "logfbank"
    feature = sys.argv[1]
    model_name = model_t.__name__

    workspace_path = os.path.join(OUTPUT_DIR, 'orca-biclf')
    namespace = feature + "_" + model_name  # namespace is the dir name under output/orca-biclf
    save_path = os.path.join(workspace_path, namespace, "models")
    log_path = os.path.join(workspace_path, namespace, "log")

    configs = resume(save_path)
    if configs is not None:
        return configs

    # create save_path/log_path if they don't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

    # data
    dataset_dir = './index/'
    train_utt2data = [[line.split()[0], line.split()[1]] for line in open(dataset_dir + 'train_utt2wav')]
    label2int = dict([line.split() for line in open(dataset_dir + 'label2int')])
    dev_utt2data = [[line.split()[0], line.split()[1]] for line in open(dataset_dir + 'devel_utt2wav')]
    utt2label = dict([line.split() for line in open(dataset_dir + 'utt2label')])

    data_info = dict(dataset_dir=dataset_dir, train_utt2data=train_utt2data,
                     label2int=label2int, dev_utt2data=dev_utt2data, utt2label=utt2label)

    # net
    learning_rate = 0.1 # resnet
    # learning_rate = 0.2 # alexnet
    net = model_t(n_classes=2)
    net = nn.DataParallel(net)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
#     optimizer = optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, min_lr=1e-4)

    print("logs are stored at {}".format(log_path))
    print("models are stored at {}".format(save_path))

    configs = {
        'n_epochs': 30,
        'epoch': 0,  # current epochs
        'batch_size': 64,
        'learning_rate': learning_rate,
        'resume_model_name': '',
        'resume_lr': None,
        'data_info': data_info,
        'feature': feature,
        'model_name': model_name,
        'namespace': namespace,
        'save_path': save_path,
        'log_path': log_path,
        'seed': seed,
        'model': net,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }
    return configs


def train(configs):
    data_info = configs['data_info']
    net = configs['model']
    optimizer = configs['optimizer']
    criterion = configs['criterion']
    # data loader preparation
    train_dataset = WavDataset(data_info['train_utt2data'], data_info['utt2label'], data_info['label2int'],
                               need_aug=True, with_label=True, shuffle=False, feat=configs['feature'])
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], num_workers=4)
    net.train()

    iteration = 0
    losses = AverageMeter()

    if not os.path.exists(configs['log_path']):
        os.makedirs(configs['log_path'], exist_ok=True)
    f_log = open(os.path.join(configs['log_path'], 'train.log'), 'a')

    for batch_utt, batch_sx, batch_sy in tqdm(train_dataloader, total=len(train_dataloader)):
        iteration += 1
        batch_sx = torch.unsqueeze(batch_sx, dim=1).float().cuda()
        # batch_sx.shape: [128, 1, 99, 64]
        batch_sy = batch_sy.cuda()

        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, _ = net(batch_sx)

        loss = criterion(outputs, batch_sy)
        loss.backward()
        optimizer.step()

        # update loss
        losses.update(loss.data, batch_sx.size()[0])

        if iteration % 30 == 29:
            _, pred = torch.max(outputs, 1)  # .data.max(1)[1] # get the index of the max log-probability
            correct = pred.eq(batch_sy.data.view_as(pred)).long().cpu().sum()
            curr_log = 'epoch {0:d}, iter {1:d} loss {2:.3f}, acc {3:d}/{4:d}'.format(
                configs['epoch'], iteration + 1, losses.avg, correct, configs['batch_size'])
            tqdm.write(curr_log)
            f_log.write(curr_log + '\n')

    return losses.avg


def save_model(model_dict):
#     torch.save(configs, os.path.join(configs['save_path'], "checkpoint_" + str(configs['epoch'])))
    torch.save(model_dict, os.path.join(configs['save_path'], "checkpoint_" + str(configs['epoch'])))

def main(configs):
    while configs['epoch'] < configs['n_epochs'] + 1:
        losses_avg = train(configs)
        configs['epoch'] += 1
        e = configs['epoch']
        configs['scheduler'].step(losses_avg)
        if e in [1, 3, 5] or e % 10 == 0:
            validate(configs)
            save_model(configs)


def validate(configs):
    net = configs['model']
    net.eval()
    data_info = configs['data_info']
    dev_dataset = WavDataset(data_info['dev_utt2data'], data_info['utt2label'], data_info['label2int'], need_aug=True,
                             with_label=True, shuffle=False, feat=configs['feature'])
    dev_loader = DataLoader(dev_dataset, batch_size=configs['batch_size'], num_workers=4)
    losses = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_utt, batch_x, batch_y in tqdm(dev_loader, total=len(dev_loader)):
            batch_x = batch_x.unsqueeze(1).float().cuda()
            batch_y = batch_y.cuda()
            outputs, _ = net(batch_x)
            _, pred = torch.max(outputs, 1)
            test_loss = configs['criterion'](outputs, batch_y)
            losses.update(test_loss, batch_x.size()[0])
            y_true.append(batch_y.float().cpu())
            y_pred.append(pred.float().cpu())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    classes = [0, 1]
    roc_auc = roc_auc_score(y_true, y_pred, labels=classes)
    print('AUC on Devel {0:.1f}'.format(roc_auc * 100))
    print('Confusion matrix (Devel):')
    print(classes)
    print(confusion_matrix(y_true, y_pred, labels=classes))
    print('Test set: Loss: {:.3f}.\n'.format(losses.avg))
    return losses.avg


if __name__ == '__main__':
    configs = init()
    main(configs)

