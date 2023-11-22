import torch
from torchvision.models import vgg16, resnet50
from torch.utils.data import DataLoader
import argparse
from typing import Tuple, Callable
import logging
import os
from ClassifyDataset import ClassifyDataset
from tqdm import tqdm
from torchmetrics.functional import accuracy

def argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        required=False,
        type=str
    )
    parser.add_argument(
        '--train_data_dir',
        required=False,
        type=str
    )
    parser.add_argument(
        '--valid_data_dir',
        required=False,
        type=str
    )
    parser.add_argument(
        '--test_data_dir',
        required=False,
        type=str
    )
    parser.add_argument(
        '--suffix',
        required=True,
        type=str
    )
    parser.add_argument(
        '--log_dir',
        required=False,
        default='logs',
        type=str
    )
    parser.add_argument(
        '--device',
        required=True,
        type=int
    )
    parser.add_argument(
        '--epochs',
        required=False,
        default=30,
        type=int
    )
    parser.add_argument(
        '--batch',
        required=False,
        default=64,
        type=int
    )
    parser.add_argument(
        '--lr',
        required=False,
        default=1.e-5,
        type=float
    )
    parser.add_argument(
        '--pretrained_model',
        required=False,
        default='resnet50',
        choices=['vgg16', 'resnet50'],
        type=str
    )
    parser.add_argument(
        '--train',
        required=False,
        action='store_true'
    )
    parser.add_argument(
        '--test',
        required=False,
        action='store_true'
    )
    parser.add_argument(
        '--valid',
        required=False,
        action='store_true'
    )
    parser.add_argument(
        '--checkpoint',
        required=False,
        type=str
    )
    parser.add_argument(
        '--num_classes',
        required=True,
        type=int
    )
    parser.add_argument(
        '--task',
        required=False,
        type=str
    )
    parser.add_argument(
        '--limit',
        required=False,
        type=int
    )
    parser.add_argument(
        '--dataset',
        required=False,
        type=str
    )
    args = parser.parse_args()

    if args.data_dir is None and args.dataset is None:
        raise Exception("Either data directory or dataset must be provided")

    log_dir = os.path.join(args.log_dir, args.suffix)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    if args.data_dir is not None:
        args.train_data_dir = os.path.join(args.data_dir, 'train') if args.train_data_dir is None else args.train_data_dir
        args.test_data_dir = os.path.join(args.data_dir, 'test') if args.test_data_dir is None else args.test_data_dir
        args.valid_data_dir = os.path.join(args.data_dir, 'valid') if args.valid_data_dir is None else args.valid_data_dir

    args.task = 'binary' if args.num_classes == 2 else 'multiclass'
    return args

def set_log_dir(log_dir: str) -> None:
    logging.basicConfig(filename=os.path.join(log_dir, 'debug.log'), level=logging.DEBUG, filemode='w')
    logging.basicConfig(filename=os.path.join(log_dir, 'info.log'), level=logging.INFO, filemode='w')
    logging.basicConfig(filename=os.path.join(log_dir, 'warning.log'), level=logging.WARNING, filemode='w')

def get_dataset(name: str, size: Tuple, train: bool) -> torch.Tensor:
    from torchvision import transforms as T
    import torchvision
    transform = T.Compose(
        [
            T.Resize(size=[256, 256], interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(size=size),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32)
        ])
    if name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('./data', train=train, download=True, transform=transform)
    else:
        dataset = None
    return dataset

def validate(dataloader: DataLoader, model: torch.nn.Module, criterion: Callable, device: str, task: str, num_classes: int, dataset: str=None):
    total_loss = 0
    total_acc = 0
    progress_bar = tqdm(len(dataloader), desc="Validating")
    for idx, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        acc = accuracy(pred, y, task=task, num_classes=num_classes)

        total_acc += acc.item()
        total_loss += loss.item()

        avg_loss = total_loss / (idx + 1)
        avg_acc = total_acc / (idx + 1)
        progress_bar.update(1)
        progress_bar.set_postfix(loss=avg_loss, acc=avg_acc)
    logging.info(f"Test loss: {avg_loss}, Accuracy: {avg_acc}")    

def train(train_data_dir: str, size: Tuple, epochs: int, batch_size: int, lr: float, model: torch.nn.Module, suffix: str, device: str, task: str, num_classes: int, limit: int, valid: bool, valid_data_dir: str=None, dataset: str=None):
    model.to(device)
    if dataset is not None:
        train_dataset = get_dataset(dataset, size, True)
    else:
        train_dataset = ClassifyDataset(train_data_dir, size, limit=limit, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=112, pin_memory=True, drop_last=True)
    if valid:
        if dataset is not None:
            valid_dataset = get_dataset(dataset, size, False)
        else:
            valid_dataset = ClassifyDataset(valid_data_dir, size, limit=limit, num_classes=num_classes)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=112, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_step = epochs * len(train_loader)

    logging.info("Train Started")
    logging.info(f"Data directory: {train_data_dir}")
    logging.info(f"Total dataset: {len(train_dataset)}")
    logging.info(f"Total epochs: {epochs}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Total Steps: {total_step}")

    progress_bar = tqdm(range(total_step))
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        progress_bar.set_description(f"Epoch: {epoch}")
        for idx, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            model.train()
            pred = model(x)
            loss = criterion(pred, y)
            acc = accuracy(pred, y, task=task, num_classes=num_classes)

            total_acc += acc.item()
            total_loss += loss.item()
            avg_loss = total_loss / (idx + 1)
            avg_acc = total_acc / (idx + 1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=avg_loss, acc=avg_acc)
        if (epoch + 1) % 5 == 0 and valid:
            validate(valid_loader, model, criterion, device, task, num_classes)
            

    torch.save(model.state_dict(), f'checkpoints/{suffix}_model.pt')

def test(data_dir: str, size: Tuple, batch_size: int, model: torch.nn.Module, device: str, task: str, num_classes: int, limit: int, dataset: str=None):
    model.to(device)
    if dataset is not None:
        test_dataset = get_dataset(dataset, size, False)
    else:
        test_dataset = ClassifyDataset(data_dir, size, limit=limit, num_classes=num_classes)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0
    total_acc = 0
    avg_acc = 0
    avg_loss = 0

    progress_bar = tqdm(len(test_loader))
    for idx, batch in enumerate(test_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        acc = accuracy(pred, y, task=task, num_classes=num_classes)
        total_acc += acc.item()
        total_loss += loss.item()
        avg_loss = total_loss / (idx + 1)
        avg_acc = total_acc / (idx + 1)
        progress_bar.set_postfix(loss = avg_loss, acc=avg_acc)
        progress_bar.update(1)
    
    logging.info(f"Test loss: {avg_loss}, Accuracy: {avg_acc}")

def get_model(model_name: str, num_classes: int, train: bool, checkpoint=None) -> torch.nn.Module:
    if model_name == 'vgg16':
        if train:
            model = vgg16(pretrained=True)
            model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes)
            for name, param in model.named_parameters():
                param.requires_grad = True
                if 'classifier.6' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            model = vgg16()
            model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes)
            model.load_state_dict = checkpoint
    elif model_name == 'resnet50':
        if train:
            model = resnet50(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            fc = model.fc
            model.fc = torch.nn.Linear(in_features=fc.in_features, out_features=num_classes)
        else:
            model = resnet50(pretrained=True)
            fc = model.fc
            model.fc = torch.nn.Linear(in_features=fc.in_features, out_features=num_classes)
            model.load_state_dict(checkpoint)
            for param in model.parameters():
                param.requires_grad = False
    else:
        logging.error("Wrong pretrained model")
    return model
            

if __name__ == '__main__':
    args = argument()
    set_log_dir(args.log_dir)
    logging.debug(args)
    SIZE = {
        'vgg16': (224, 224),
        'resnet50': (224, 224)
    }
    device = f"cuda:{args.device}"
    if args.train:
        model = get_model(args.pretrained_model, args.num_classes, True)
        train(args.train_data_dir, SIZE[args.pretrained_model], args.epochs, args.batch, args.lr, model, args.suffix, device, args.task, args.num_classes, args.limit, args.valid, args.valid_data_dir, args.dataset)
        args.checkpoint = f'checkpoints/{args.suffix}_model.pt'
    if args.test:
        checkpoint = torch.load(args.checkpoint)
        model = get_model(args.pretrained_model, args.num_classes, False, checkpoint)
        model.eval()
        test(args.test_data_dir, SIZE[args.pretrained_model], args.batch, model, device, args.task, args.num_classes, args.limit, args.dataset)

    

    

    

    
            
            