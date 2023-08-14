import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torch
import torch.nn as nn
import torch.distributed as dist
from models import *
from datasets import *
import platform
import copy
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader
#from milesial_unet_model import UNet
from leejunhyun_unet_models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AttU_Net_S
from sklearn.metrics import confusion_matrix


TRAIN = 'train'
VAL = 'validation'

MASTER_RANK = 0
SAVE_INTERVAL = 1

# Make sure to change these paths!
DATASET_PATH = '/home/jack/paying-attention-to-wildfire/data/next-day-wildfire-spread'
SAVE_MODEL_PATH = '/home/jack/paying-attention-to-wildfire/savedModels'


def main():
    parser = argparse.ArgumentParser()
    # Master node may need to be an IP address, sardine is a lab machine at CSU.
    parser.add_argument('-m','--master', default='sardine',
                        help='master node')
    parser.add_argument('-p', '--port', default='30437',
                         help = 'master node')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=1, type=int,
                        metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    print(f'initializing ddp: GLOBAL_RANK: {args.nr}, MEMBER: {int(args.nr)+1} / {args.nodes}')
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
    os.environ['MASTER_ADDR'] = args.master              #
    os.environ['MASTER_PORT'] = args.port                  #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################


def create_data_loaders(rank, gpu, world_size):
    batch_size = 50

    datasets = {
        TRAIN: RotatedWildfireDataset(
            f"{DATASET_PATH}/{TRAIN}.data",
            f"{DATASET_PATH}/{TRAIN}.labels",
            features=[0, 2, 5, 7, 8, 9, 11],
            crop_size=32
        ),
        VAL: WildfireDataset(
            f"{DATASET_PATH}/{VAL}.data",
            f"{DATASET_PATH}/{VAL}.labels",
            features=[0, 2, 5, 7, 8, 9, 11],
            crop_size=32
        )
    }

    samplers = {
        TRAIN: torch.utils.data.distributed.DistributedSampler(
            datasets[TRAIN],
            num_replicas=world_size,
            rank=rank
        )
    }

    dataLoaders = {
        TRAIN: torch.utils.data.DataLoader(
            dataset=datasets[TRAIN],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=samplers[TRAIN]
        ),
        VAL: torch.utils.data.DataLoader(
            dataset=datasets[VAL],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    }

    return dataLoaders


def get_model_state_dict(filename):
    state_dict = torch.load(f"{SAVE_MODEL_PATH}/{filename}", map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def pickle_loss_history(loss_history, filename):
    with open(f"{SAVE_MODEL_PATH}/{filename}", "wb") as handle:
        pickle.dump(loss_history, handle)
    print(f"Successfully pickled the loss history in {SAVE_MODEL_PATH}/{filename}")


def perform_validation(model, loader):
    model.eval()

    loss_val = 0
    acc_val = 0
    total_pixels = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)

            labels = torch.flatten(labels)
            outputs = torch.flatten(outputs)

            threshold = 0.5
            preds = torch.where(torch.sigmoid(outputs) > threshold, 1, 0)

            loss = torchvision.ops.sigmoid_focal_loss(outputs, labels, alpha=0.85, gamma=2, reduction="mean")
            #loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels, pos_weight=torch.Tensor([5]).cuda())
            #loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)

            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data)
            total_pixels += len(labels)
            tn, fp, fn, tp = confusion_matrix(labels.cpu(), preds.cpu(), labels=[0,1]).ravel()
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
    
        print(f"Validation - tp={total_tp} fp={total_fp} fn={total_fn} tn={total_tn}")
        curr_avg_loss_val = loss_val / len(loader)
        curr_avg_acc_val = 100 * acc_val / total_pixels
        curr_precision = total_tp / (total_tp + total_fp)
        curr_recall = total_tp / (total_tp + total_fn)
        curr_f1_score = 2 * curr_precision * curr_recall / (curr_precision + curr_recall)

    return curr_avg_loss_val, curr_avg_acc_val, curr_precision, curr_recall, curr_f1_score


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    validate = True
    print("Current GPU", gpu,"\n RANK: ",rank)

    dataLoaders = create_data_loaders(rank, gpu, args.world_size)

    torch.manual_seed(0)

    model = AttU_Net_S(7, 1)
    #model = U_Net(12, 1)
    #model = R2U_Net(12, 1)
    #model = AttU_Net(12, 1)
    #model = R2AttU_Net(12, 1)
    #model = BinaryClassifierCNN(11, 32)

    # Uncomment the lines below to load in an old model if you would like to
    #new_state_dict = get_model_state_dict(filename="model-UNet-bestLoss-Rank-0.weights")
    #model.load_state_dict(new_state_dict)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5])).cuda(gpu)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    start = datetime.now()
    print(f'TRAINING ON: {platform.node()}, Starting at: {datetime.now()}')

    total_step = len(dataLoaders[TRAIN])
    best_epoch = 0
    best_avg_loss_val = float("inf")
    best_avg_acc_val = -float("inf")
    best_f1_score = -float("inf")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(args.epochs):
        model.train()

        loss_train = 0

        for i, (images, labels) in enumerate(dataLoaders[TRAIN]):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Not entirely sure if this flattening is required or not
            labels = torch.flatten(labels)
            outputs = torch.flatten(outputs)

            #loss = criterion(outputs, labels)
            loss = torchvision.ops.sigmoid_focal_loss(outputs, labels, alpha=0.85, gamma=2, reduction="mean")

            loss_train += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print('Epoch [{}/{}], Steps [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i,
                    total_step,
                    loss.item())
                )

        train_loss_history.append(loss_train / len(dataLoaders[TRAIN]))
    
        if validate:
            curr_avg_loss_val, curr_avg_acc_val, curr_precision, curr_recall, curr_f1_score = perform_validation(model, dataLoaders[VAL])

            print(f"Average validation batch loss = {curr_avg_loss_val}")
            print(f"Validation acc = {curr_avg_acc_val}%")
            print(f"Precision = {curr_precision}")
            print(f"Recall = {curr_recall}")
            print(f"F1 Score = {curr_f1_score}")

            val_loss_history.append(curr_avg_loss_val)

            if best_f1_score < curr_f1_score:
                print("Saving model...")
                best_epoch = epoch
                best_f1_score = curr_f1_score
                filename = f'model-{model.module.__class__.__name__}-bestF1Score-Rank-{rank}.weights'
                torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}/{filename}')
                print("Model has been saved!")
            else:
                print("Model is not being saved")

    pickle_loss_history(train_loss_history, filename=f"model-{model.module.__class__.__name__}-train-loss-Rank-{rank}.history")
    pickle_loss_history(val_loss_history, filename=f"model-{model.module.__class__.__name__}-validation-loss-Rank-{rank}.history")
        
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        print(f"Endtime: {datetime.now()}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best F1 score: {best_f1_score}")
    

if __name__ == '__main__':
    main()

