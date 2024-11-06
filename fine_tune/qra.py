import sys, os
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import  pformat
import numpy as np
import torch
import random
import logging
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

import glob  
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.bd_dataset import prepro_cls_DatasetBD

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from load_data import CustomDataset, CustomDataset_v2
from utils.save_load_attack import load_attack_result
from utils.choose_index import choose_index_v2
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

class AE(nn.Module):
    def __init__(self, img_size=32):
        super().__init__()
        in_channels = 3 * img_size * img_size
        out_channels = 3 * img_size * img_size
        self.encoder_hidden_layer = nn.Linear(
            in_features=in_channels, out_features=1024
        )
        self.encoder_output_layer = nn.Linear(
            in_features=1024, out_features=1024
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=1024, out_features=1024
        )
        self.decoder_output_layer = nn.Linear(
            in_features=1024, out_features=out_channels
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.tanh(activation)
        return reconstructed

def distillation(y, labels, teacher_scores, temp=1, alpha=1):
    return nn.KLDivLoss(reduction="batchmean")(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

def save_image_from_tensor(tensor, save_path):
    tensor_np = tensor.detach().cpu().numpy()
    scaled_tensor = (tensor_np * 255).astype(np.uint8)
    image = Image.fromarray(np.transpose(scaled_tensor, (1, 2, 0)))
    image.save(save_path)

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--device', type = str)
    parser.add_argument('--attack', type = str, )
    parser.add_argument('--attack_label_trans', type=str, default='all2one',
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    
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
                        help='the data ratio for optimizing reactivating perturbation')
    
    parser.add_argument('--log', action='store_true',
                        help='record the log')
    
    parser.add_argument('--pre', action='store_true', help='load pre-trained weights')
    parser.add_argument('--save', action='store_true', help='save the model checkpoint')

    parser.add_argument('--eps', type=float, default=16/255, help='perturbation budget of qra perturbation')
    parser.add_argument('--defense_type', type=str, default='fst', help='defense type for qra')
    parser.add_argument('--clean_num_qra', type=int, default=500, help='number of clean examples for optimizing qra perturbation')
    parser.add_argument('--poison_num_qra', type=int, default=500, help='number of poisoned examples for optimizing qra perturbation')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to optimize qra')
    parser.add_argument('--alpha_qra', type=float, default=0.2, help='balance parameter between reactivating perturbation and adversarial perturbation \
                        larger alpha_qra indicates improved reactivating performance, but meanwhile leads to more adversarial component \
                        this will make the reactivating perturbation attack both the backdoor and benign models simultaneously')
    
    return parser

def main():

    ### 1. Config args, save_path, fix random seed
    
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    args.opt = 'sgd'
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
    data_set_without_tran = clean_dataset
    benign_train_ds = result['clean_train']
    benign_train_ds.wrapped_dataset = data_set_without_tran
    benign_train_ds.wrap_img_transform = test_img_transform
    print(len(benign_train_ds))
    

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
    adv_test_loader = DataLoader(
        dataset = adv_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)
    
    clean_loader = DataLoader(
            dataset = benign_test_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )


    if args.attack == 'label_consistent':
        image_list = glob.glob(folder_path + '/bd_test_dataset/*/*.png')
    else:
        image_list = glob.glob(folder_path + '/bd_train_dataset/*/*.png')
    image_list = [x for x in image_list if f'/{args.attack_target}/' not in x]
    label_list = [int(x.split('/')[-2]) for x in image_list]
    adv_train_dataset = CustomDataset_v2(image_list, label_list, test_img_transform)

    adv_train_list = []
    for imgs in adv_train_dataset:
        adv_train_list.append(imgs)
    
    random.shuffle(adv_train_list)
   

   
    if not args.pre:
          
        net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0])
    else:

        if args.model == "resnet18":        
            from torchvision.models import resnet18, ResNet18_Weights        
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device)
            net.fc = nn.Linear(in_features=512, out_features=args.num_classes, bias=True).to(args.device) 
            for _, param in net.named_parameters():
                param.requires_grad = True
        elif args.model == "resnet50":        
            from torchvision.models import resnet50, ResNet50_Weights        
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(args.device)    
            net.fc = nn.Linear(in_features=2048, out_features=args.num_classes, bias=True).to(args.device) 
            for _, param in net.named_parameters():
                param.requires_grad = True
        elif args.model == 'swin_b':
            from torchvision.models import swin_b        
            net = swin_b(weights='IMAGENET1K_V1').to(args.device)
            net.head = nn.Linear(in_features=1024, out_features=args.num_classes, bias=True).to(args.device) 
            for _, param in net.named_parameters():
                param.requires_grad = True
        elif args.model == 'swin_t':        
            from torchvision.models import swin_t        
            net = swin_t(weights='IMAGENET1K_V1').to(args.device)
            net.head = nn.Linear(in_features=768, out_features=args.num_classes, bias=True).to(args.device) 
            for _, param in net.named_parameters():
                param.requires_grad = True    
        else:        
            raise NotImplementedError(f"{args.model} is not supported")
  
  
            
    from copy import deepcopy
    
    net.to(device)
    net_purified = deepcopy(net).eval()
    net_relearn = deepcopy(net).eval()
    net_clean = deepcopy(net).eval()
    

    relearn_path = folder_path + f'/defense/{args.defense_type}/relearn_result.pt'
    defense_path = folder_path + f'/defense/{args.defense_type}/defense_result.pt'
    clean_path = f'../record/{args.dataset}/badnet/pratio_0.0-target_0-archi_{args.model}/attack_result.pt'
    model_dict = torch.load(relearn_path, map_location='cpu')
    net_relearn.load_state_dict(model_dict)
      
    model_dict = torch.load(defense_path, map_location='cpu')
    if 'model' in model_dict:
        model_dict = model_dict['model']
    net_purified.load_state_dict(model_dict)

    model_dict = torch.load(clean_path, map_location='cpu')
    if 'model' in model_dict:
        model_dict = model_dict['model']
    
    net_clean.load_state_dict(model_dict)
    
    

     
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        unnormalize_rgb = transforms.Normalize(mean=[-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], std=[1/0.247, 1/0.243, 1/0.261])
        img_size = 32
    elif args.dataset == 'cifar100' or args.dataset == 'tiny':
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        unnormalize_rgb = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        img_size = 224

    qra_data_list = []

    for idx, (x, label, *additional_info) in enumerate(benign_train_ds):
        if idx == args.clean_num_qra:
            break
        qra_data_list.append([x, [label,label]])
        
    
    for i in range(args.poison_num_qra):
        x, label = adv_train_list[i]
        qra_data_list.append([x, [args.attack_target, label]])
           
    
    poison_data = CustomDataset(qra_data_list)
    qra_data_loader = DataLoader(
            dataset = poison_data,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

    autoencoder = AE(img_size=img_size).to(device)
    
    param_list = []
    for param in autoencoder.parameters():
        param_list.append(param)
        param.requires_grad = True
    autoencoder.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(param_list, lr=1e-1,momentum = 0.9) # 5e-2
    
    
    for epoch in range(args.epochs):
        
        train_correct = 0
        train_tot = 0
        
        logging.info(f'Epoch: {epoch}')
        
        loss_all = 0
        for batch_idx, (x, labels) in enumerate(qra_data_loader):

           
            x, labels, gt_labels = x.to(device), labels[0].to(device), labels[1].to(device)

               
            trigger = torch.clamp(autoencoder(x.flatten(1))*args.eps, -args.eps, args.eps).reshape(-1, 3, img_size, img_size)
            x_unnorm = torch.clamp(unnormalize_rgb(x) + trigger, 0, 1)
            
            x_trigger = normalize(x_unnorm) 
               
            log_probs = net_purified(x_trigger)
            log_probs_t = net_relearn(x).detach()
            loss_clean = criterion(net_clean(x_trigger), gt_labels.long())
           
            loss = distillation(log_probs, labels, log_probs_t) * args.alpha_qra + loss_clean
            loss.backward()
          
           
            optimizer.step()
            optimizer.zero_grad()
            
            _, predicted = torch.max(log_probs, -1)
            train_correct += predicted.eq(labels).sum()
            train_tot += labels.size(0)
            loss_all += loss
        logging.info(f'Training ACC {train_correct / train_tot} Total Loss {loss_all}')


    os.makedirs('examples', exist_ok=True)
    autoencoder.eval()

    tot_correct = 0
    tot_count = 0
    for batch_idx, (x, labels, *add) in enumerate(clean_loader):
        x = x.to(device)
        
        
        trigger = torch.clamp(autoencoder(x.flatten(1))*args.eps, -args.eps, args.eps).reshape(-1, 3, img_size, img_size)
        x_unnorm = torch.clamp(unnormalize_rgb(x) + trigger, 0, 1)
        x_tmp = x_unnorm[0].clone()
        x = normalize(x_unnorm)

        save_image_from_tensor(x_tmp, 'examples/clean.png') # for visulation
        
        labels = labels.to(device) * 0
        log_probs = net_purified(x)

        _, predicted = torch.max(log_probs, -1)
        tot_correct += predicted.eq(labels).sum()
        tot_count += x.shape[0]
    
    print('Evaluate on purified models')
    print('QRA on Clean Samples (Acc):', tot_correct / tot_count)
    


    tot_correct = 0
    tot_count = 0
    for batch_idx, (x, labels, *add) in enumerate(adv_test_loader):
        correct = 0
        x = x.to(device)

      
        trigger = torch.clamp(autoencoder(x.flatten(1))*args.eps, -args.eps, args.eps).reshape(-1, 3, img_size, img_size)
        x_unnorm = torch.clamp(unnormalize_rgb(x) + trigger, 0, 1)
        x_tmp = x_unnorm[0].clone()
        x = normalize(x_unnorm)
       
        save_image_from_tensor(x_tmp, 'examples/poisoned.png') # for visulation

        labels = labels.to(device)
        log_probs = net_purified(x)
        
        _, predicted = torch.max(log_probs, -1)
        tot_correct += predicted.eq(labels).sum()
        tot_count += x.shape[0]
       
    print('Evaluate on purified models')
    print('QRA on Poisoned Samples (ASR):', tot_correct / tot_count)


    tot_correct = 0
    tot_count = 0
    for batch_idx, (x, labels, *add) in enumerate(clean_loader):
        x = x.to(device)
        
    
        trigger = torch.clamp(autoencoder(x.flatten(1))*args.eps, -args.eps, args.eps).reshape(-1, 3, img_size, img_size)
        x_unnorm = torch.clamp(unnormalize_rgb(x) + trigger, 0, 1)
        x_tmp = x_unnorm[0].clone()
        x = normalize(x_unnorm)
        
        labels = labels.to(device) * 0
        log_probs = net_clean(x)

        _, predicted = torch.max(log_probs, -1)
        tot_correct += predicted.eq(labels).sum()
        tot_count += x.shape[0]
       

    print('Evaluate on clean models')
    print('QRA on Clean Samples (Acc):', tot_correct / tot_count)
    
    tot_correct = 0
    tot_count = 0
    for batch_idx, (x, labels, *add) in enumerate(adv_test_loader):
        x = x.to(device)
        
        trigger = torch.clamp(autoencoder(x.flatten(1))*args.eps, -args.eps, args.eps).reshape(-1, 3, img_size, img_size)
        x_unnorm = torch.clamp(unnormalize_rgb(x) + trigger, 0, 1)
        x_tmp = x_unnorm[0].clone()
        x = normalize(x_unnorm)
       
        save_image_from_tensor(x_tmp, 'examples/clean.png')
       
       
        labels = labels.to(device)
        log_probs = net_clean(x)
       
        _, predicted = torch.max(log_probs, -1)
        tot_correct += predicted.eq(labels).sum()
        tot_count += x.shape[0]
       

    print('Evaluate on clean models')
    print('QRA on Poisoned Samples (ASR):', tot_correct / tot_count)
    

if __name__ == '__main__':
    main()
    
