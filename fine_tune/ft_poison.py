import sys, os
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import pformat
import numpy as np
import torch
import logging
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import glob   
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, dataset_and_transform_generate
from utils.aggregate_block.fix_random import fix_random
from utils.bd_dataset import prepro_cls_DatasetBD

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from load_data import CustomDataset, CustomDataset_v2
from test import test
from utils.save_load_attack import load_attack_result
from utils.choose_index import choose_index_v2
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
import yaml
import random


def add_bd_yaml_to_args(args):
    with open(args.bd_yaml_path, 'r') as f:
        mix_defaults = yaml.safe_load(f)
    mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = mix_defaults


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--attack', type = str, )
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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--random_seed', default=42,type=int,
                        help='random_seed')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    
    
    parser.add_argument('--log', action='store_true',
                        help='record the log')
    parser.add_argument('--split_ratio', type=float, default=0.02, help='the data ratio for fine-tuning attack')
    parser.add_argument('--poison_num', type=int, default=5, help='the number of poisoned samples for fine-tuning attack')
    parser.add_argument('--pre', action='store_true', help='load pre-trained weights')
    parser.add_argument('--save', action='store_true', help='save the model checkpoint')
    parser.add_argument('--defense_type', type=str, default='fst', help='defense type for fine-tuning attack')
   
    return parser

def main():

    ### 1. Config args, save_path, fix random seed
    
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    fix_random(args.random_seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()


    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    folder_path = f'../record/{args.dataset}/{args.attack}/pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}'
    ### 2. Set up the training and test dataset

    train_dataset_without_transform, train_img_transform, \
                train_label_transfrom, \
    test_dataset_without_transform, \
                test_img_transform, \
                test_label_transform = dataset_and_transform_generate(args)
  
    
    result = load_attack_result(folder_path + '/attack_result.pt')
    clean_dataset = prepro_cls_DatasetBD_v2(result['clean_train'].wrapped_dataset)
    data_all_length = len(clean_dataset)
    ran_idx = choose_index_v2(args.split_ratio, data_all_length) 
    
    clean_dataset.subset(ran_idx)
    data_set_clean = result['clean_train']
    data_set_clean.wrapped_dataset = clean_dataset
    data_set_clean.wrap_img_transform = train_img_transform
    benign_train_ds = data_set_clean
    
    benign_test_ds = prepro_cls_DatasetBD(
            test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_img_transform,
            ori_label_transform_in_loading=test_label_transform,
            add_details_in_preprocess=True,
        )


    adv_test_dataset = result['bd_test']
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
    
    ### 3. Load purified model

    net = generate_cls_model(
        model_name=args.model,
        num_classes=args.num_classes,
        image_size=args.img_size[0])
    net.to(device)
    
    
    
    model_dict = torch.load(folder_path + f'/defense/{args.defense_type}/defense_result.pt', map_location='cpu')
    if 'model' in model_dict:
        model_dict = model_dict['model']
    new_dict = {}
    for key in model_dict.keys():
        if 'model' in key:
            new_dict[key[6:]] = model_dict[key]
        else:
            new_dict[key] = model_dict[key]
    net.load_state_dict(new_dict)
    

    ### 4. Load poisoned data
    
    image_list = glob.glob(folder_path + '/bd_train_dataset/*/*.png')
    adv_train_dataset = CustomDataset_v2(image_list, args.attack_target, test_img_transform)

    adv_train_list = []
    for imgs in adv_train_dataset:
        adv_train_list.append(imgs)
    
    random.shuffle(adv_train_list)
    
    
    ### 5. Set up poisoned dataset for fine-tuning attack

    data_list = []
    count = 0     
    for (x, label, *addition) in benign_train_ds:
        if count < args.poison_num:
            x, label = adv_train_list[count]
            data_list.append([x, label])
            count += 1
        else:
            data_list.append([x, label])
                
    print('Number of data to conduct fine-tuning attack: ', len(data_list))

   
    train_loader = DataLoader(
            dataset = CustomDataset(data_list),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

    ### 6. Evaluate defense performace

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

    ### 6. Launch fine-tuning attack

    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum = 0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        
        batch_loss_list = []
        train_correct = 0
        train_tot = 0
        
        logging.info(f'Epoch: {epoch}')
        net.train()

        for batch_idx, (x, labels, *additional_info) in enumerate(train_loader):

            x, labels = x.to(device), labels.to(device)
            log_probs= net(x)
            
            loss = criterion(log_probs, labels.long())
            
            loss.backward()
            
            
            optimizer.step()
            optimizer.zero_grad()


            _, predicted = torch.max(log_probs, -1)
            train_correct += predicted.eq(labels).sum()
            train_tot += labels.size(0)
            batch_loss = loss.item() * labels.size(0)
            batch_loss_list.append(batch_loss)

        one_epoch_loss = sum(batch_loss_list)


        logging.info(f'Training ACC: {train_correct/train_tot} | Training loss: {one_epoch_loss}')
        logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        logging.info('-------------------------------------')

    ### 6. Evaluate and save

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
        net.eval()
        net_state = net.state_dict()
        torch.save(net_state, folder_path + f'/defense/{args.defense_type}/relearn_result.pt')
  
    
if __name__ == '__main__':
    main()
    

