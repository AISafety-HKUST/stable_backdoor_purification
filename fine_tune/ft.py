import sys, os
import math
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import  pformat
import numpy as np
import torch
import time
import copy
import logging
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate_ft import dataset_and_transform_generate
from utils.bd_dataset import prepro_cls_DatasetBD

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from load_data import CustomDataset, CustomDataset_v2

from test import test


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--device', type = str)
    parser.add_argument('--ft_mode', type = str, default='all')
    
    parser.add_argument('--attack', type = str, )
    parser.add_argument('--attack_label_trans', type=str, default='all2one',
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str,
                        help='which dataset to use'
                        )
    parser.add_argument('--dataset_path', type=str,default='../data')
    parser.add_argument('--attack_target', type=int,default=0,
                        help='target class in all2one attack')
    parser.add_argument('--batch_size', type=int,default=128)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--random_seed', default=0,type=int,
                        help='random_seed')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    
    parser.add_argument('--split_ratio', type=float,
                        help='part of the training set for defense')
    
    parser.add_argument('--log', action='store_true',
                        help='record the log')
    parser.add_argument('--initlr', type=float, help='initial learning rate for training backdoor models')
    parser.add_argument('--pre', action='store_true', help='load pre-trained weights')
    parser.add_argument('--save', action='store_true', help='save the model checkpoint')
    parser.add_argument('--linear_name', type=str, default='linear', help='name for the linear classifier')
    parser.add_argument('--lb_smooth', type=float, default=None, help='label smoothing')
    parser.add_argument('--alpha', type=float, default=0.2, help='fst')
    return parser

def main():

    ### 1. config args, save_path, fix random seed
    
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    
    fix_random(args.random_seed)
    
    
    if args.lb_smooth is not None:
        lbs_criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=args.lb_smooth)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.ft_mode == 'fe-tuning':
        init = True
        log_name = 'FE-tuning'
    elif args.ft_mode == 'ft-init':
        init = True
        log_name = 'FT-init'
    elif args.ft_mode == 'ft':
        init = False
        log_name = 'FT'
    elif args.ft_mode == 'lp':
        init = False
        log_name = 'LP'
    elif args.ft_mode == 'fst':
        assert args.alpha is not None
        init = True
        log_name = 'FST'
    else:
        raise NotImplementedError('Not implemented method.')

    if not args.pre:
        
        args.folder_path = f'../record_{args.dataset}/{args.attack}/' + f'pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-initlr_{args.initlr}'
        os.makedirs(f'../logs_{args.model}_{args.dataset}/{log_name}/{args.attack}', exist_ok=True)
        args.save_path = f'../logs_{args.model}_{args.dataset}/{log_name}/{args.attack}/' + f'pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-lr_{args.lr}-initlr_{args.initlr}-mode_{args.ft_mode}-epochs_{args.epochs}'
    else:
        args.folder_path = f'../record_{args.dataset}_pre/{args.attack}/' + f'pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-initlr_{args.initlr}'
        os.makedirs(f'../logs_{args.model}_{args.dataset}_pre/{log_name}/{args.attack}', exist_ok=True)
        args.save_path = f'../logs_{args.model}_{args.dataset}_pre/{log_name}/{args.attack}/' + f'pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-lr_{args.lr}-initlr_{args.initlr}-mode_{args.ft_mode}-epochs_{args.epochs}'
        
        
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()

    if args.log:
        fileHandler = logging.FileHandler(args.save_path + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)


    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))


    ### 2. set the clean train data and clean test data
    if not args.pre:
        _, train_img_transform, \
                    train_label_transfrom, \
        test_dataset_without_transform, \
                    test_img_transform, \
                    test_label_transform, \
                    ft_dataset_without_transform = dataset_and_transform_generate(args)
    else:
        from utils.aggregate_block.dataset_and_transform_generate_ft import dataset_and_transform_generate_pre
        _, train_img_transform, \
                    train_label_transfrom, \
        test_dataset_without_transform, \
                    test_img_transform, \
                    test_label_transform, \
                    ft_dataset_without_transform = dataset_and_transform_generate_pre(args)
    
    benign_train_ds = prepro_cls_DatasetBD(
            full_dataset_without_transform=ft_dataset_without_transform,
            poison_idx=np.zeros(len(ft_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_img_transform,
            ori_label_transform_in_loading=train_label_transfrom,
            add_details_in_preprocess=True,
        )
    

    benign_test_ds = prepro_cls_DatasetBD(
            test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_img_transform,
            ori_label_transform_in_loading=test_label_transform,
            add_details_in_preprocess=True,
        )


    model_dict = torch.load(args.folder_path + '/attack_result.pt')
    adv_test_dataset = model_dict['bd_test']
    
    if 'x' in adv_test_dataset.keys():
        adv_test_dataset = CustomDataset(adv_test_dataset['x'], adv_test_dataset['y'], test_img_transform) # For BackdoorBench v1
    else:
        import glob
        image_list = glob.glob(args.folder_path + '/bd_test_dataset/*/*.png')
        adv_test_dataset = CustomDataset_v2(image_list, args.attack_target, test_img_transform)

    ### 3. generate dataset for backdoor defense and evaluation

    train_data = DataLoader(
            dataset = benign_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
    
    test_dataset_dict={
                "test_data" :benign_test_ds,
                "adv_test_data" :adv_test_dataset,
        }

    test_dataloader_dict = {
            name : DataLoader(
                    dataset = test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                )
            for name, test_dataset in test_dataset_dict.items()
        }   
    
    if not args.pre:
        net  = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )
    else:
        
        if args.model == "resnet18":        
            from torchvision.models import resnet18, ResNet18_Weights        
            net = resnet18().to(device)
            net.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True).to(device) 
            
        elif args.model == "resnet50":        
            from torchvision.models import resnet50, ResNet50_Weights        
            net = resnet50().to(device)    
            net.fc = nn.Linear(in_features=2048, out_features=args.num_classes, bias=True).to(device) 
            
        elif args.model == 'swin_b':
            from torchvision.models import swin_b        
            net = swin_b().to(device)
            net.head = nn.Linear(in_features=1024, out_features=args.num_classes, bias=True).to(device) 
            
        elif args.model == 'swin_t':        
            from torchvision.models import swin_t        
            net = swin_t().to(device)
            net.head = nn.Linear(in_features=768, out_features=args.num_classes, bias=True).to(device) 
            
        else:        
            raise NotImplementedError(f"{args.model} is not supported")

    net.load_state_dict(model_dict['model'])   
    net.to(device)

    for dl_name, test_dataloader in test_dataloader_dict.items():
        metrics = test(net, test_dataloader, device)
        metric_info = {
            f'{dl_name} acc': metrics['test_correct'] / metrics['test_total'],
            f'{dl_name} loss': metrics['test_loss'],
        }
        if 'test_data' == dl_name:
            cur_clean_acc = metric_info['test_data acc']
        if 'adv_test_data' == dl_name:
            cur_adv_acc = metric_info['adv_test_data acc']
    logging.info('*****************************')
    logging.info(f"Load from {args.folder_path + '/attack_result.pt'}")
    logging.info(f'Fine-tunning mode: {args.ft_mode}')
    logging.info('Original performance')
    logging.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
    logging.info('*****************************')


    original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')

    param_list = []
    for name, param in net.named_parameters():
        if args.linear_name in name:
            if init:
                if 'weight' in name:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    std = 1 / math.sqrt(param.size(-1)) 
                    param.data.uniform_(-std, std)
                    
                else:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    param.data.uniform_(-std, std)
        if args.ft_mode == 'lp':
            if args.linear_name in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        elif args.ft_mode == 'ft' or args.ft_mode == 'fst' or args.ft_mode == 'ft-init':
            param.requires_grad = True
            param_list.append(param)
        elif args.ft_mode == 'fe-tuning':
            if args.linear_name not in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        
        

    optimizer = optim.SGD(param_list, lr=args.lr,momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        
        batch_loss_list = []
        train_correct = 0
        train_tot = 0
        
    
        logging.info(f'Epoch: {epoch}')
        net.train()

        for batch_idx, (x, labels, *additional_info) in enumerate(train_data):

            
            x, labels = x.to(device), labels.to(device)
            log_probs= net(x)
            if args.lb_smooth is not None:
                loss = lbs_criterion(log_probs, labels)
            else:
                if args.ft_mode == 'fst':
                    loss = torch.sum(eval(f'net.{args.linear_name}.weight') * weight_mat_ori)*args.alpha + criterion(log_probs, labels.long())
                else:
                    loss = criterion(log_probs, labels.long())
            loss.backward()
            
            
            optimizer.step()
            optimizer.zero_grad()

            exec_str = f'net.{args.linear_name}.weight.data = net.{args.linear_name}.weight.data * original_linear_norm  / torch.norm(net.{args.linear_name}.weight.data)'
            exec(exec_str)

            _, predicted = torch.max(log_probs, -1)
            train_correct += predicted.eq(labels).sum()
            train_tot += labels.size(0)
            batch_loss = loss.item() * labels.size(0)
            batch_loss_list.append(batch_loss)

    
        scheduler.step()
        one_epoch_loss = sum(batch_loss_list)


        logging.info(f'Training ACC: {train_correct/train_tot} | Training loss: {one_epoch_loss}')
        logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        logging.info('-------------------------------------')
        
        cur_clean_acc, cur_adv_acc = 0,0

        if epoch == args.epochs-1:
            for dl_name, test_dataloader in test_dataloader_dict.items():
                metrics = test(net, test_dataloader, device)
                metric_info = {
                    f'{dl_name} acc': metrics['test_correct'] / metrics['test_total'],
                    f'{dl_name} loss': metrics['test_loss'],
                }
                if 'test_data' == dl_name:
                    cur_clean_acc = metric_info['test_data acc']
                if 'adv_test_data' == dl_name:
                    cur_adv_acc = metric_info['adv_test_data acc']
            logging.info('Defense performance')
            logging.info(f"Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}") 
            logging.info('-------------------------------------')
    
    if args.save:
        model_save_path = f'defense_results/{args.attack}/pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-lr_{args.lr}-initlr_{args.initlr}-mode_{args.ft_mode}-epochs_{args.epochs}'
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(net.state_dict(), f'{model_save_path}/checkpoint.pt')
        
    
if __name__ == '__main__':
    main()
    
