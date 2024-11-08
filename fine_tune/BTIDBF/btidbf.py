import torch
from loader import Box
from models.unet_model import UNet
from evaluate import test, get_target_label
import cfg
import os
from models.mask import MaskGenerator
from copy import deepcopy
from tqdm import tqdm

def main(opt):
    device = opt.device
    box = Box(opt)
    save_path = box.get_save_path()
    if opt.dataset == 'cifar':
        folder_path = f'../../record/cifar10/{opt.attack}/pratio_{opt.pratio}-target_{opt.attack_target}-archi_{opt.model}'
    else:
        folder_path = f'../../record/{opt.dataset}/{opt.attack}/pratio_{opt.pratio}-target_{opt.attack_target}-archi_{opt.model}'
    print(f'Load from {folder_path}')
    _, _, classifier = box.get_state_dict(folder_path + '/attack_result.pt')
    classifier.to(box.device)
    cln_trainloader = box.get_dataloader(train="clean", batch_size=opt.batch_size, shuffle=True)

    bd_gen = UNet(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4)
    bd_gen.load_state_dict(torch.load(os.path.join(save_path, f"pretrain/{opt.model}_{opt.attack_name}_{opt.pratio}_init_generator.pt"), map_location=torch.device('cpu')))
    bd_gen = bd_gen.to(device)
    opt_bd = torch.optim.Adam(bd_gen.parameters(), lr=opt.gen_lr)
    bd_gen.eval()

    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()

    tmp_img = torch.ones([1, 3, opt.size, opt.size], device=device)
    tmp_feat = classifier.from_input_to_features(tmp_img).detach()
    feat_shape = tmp_feat.shape
    init_mask = torch.randn(feat_shape).to(device)
    m_gen = MaskGenerator(init_mask=init_mask, classifier=classifier)
    opt_m = torch.optim.Adam([m_gen.mask_tanh], lr=0.01)
    
    for m in range(opt.mround):
        tloss = 0
        tloss_pos_pred = 0
        tloss_neg_pred = 0
        m_gen.train()
        classifier.train()
        pbar = tqdm(cln_trainloader, desc="Decoupling Benign Features")
        for batch_idx, (cln_img, targets) in enumerate(pbar):
            opt_m.zero_grad()
            cln_img = cln_img.to(device)
            targets = targets.to(device)
            feat_mask = m_gen.get_raw_mask()
            cln_feat = classifier.from_input_to_features(cln_img)
            mask_pos_pred = classifier.from_features_to_output(feat_mask*cln_feat)
            remask_neg_pred = classifier.from_features_to_output((1-feat_mask)*cln_feat)
            mask_norm = torch.norm(feat_mask, 1)

            loss_pos_pred = ce(mask_pos_pred, targets)
            loss_neg_pred = ce(remask_neg_pred, targets)            
            loss = loss_pos_pred - loss_neg_pred

            loss.backward()
            opt_m.step()

            tloss += loss.item()
            tloss_pos_pred += loss_pos_pred.item()
            tloss_neg_pred += loss_neg_pred.item()
            pbar.set_postfix({"epoch": "{:d}".format(m), 
                              "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                              "loss_pos_pred": "{:.4f}".format(tloss_pos_pred/(batch_idx+1)),
                              "loss_neg_pred": "{:.4f}".format(tloss_neg_pred/(batch_idx+1)),
                              "mask_norm": "{:.4f}".format(mask_norm)})
            
    feat_mask = m_gen.get_raw_mask().detach()
    max_bd_feat = -float('inf')
    final_bd_gen = None

    for u in range(opt.uround):
        tloss = 0
        tloss_benign_feat = 0
        tloss_backdoor_feat = 0
        tloss_norm = 0
        m_gen.eval()
        bd_gen.train()
        classifier.eval()
        pbar = tqdm(cln_trainloader, desc="Training Backdoor Generator")
        for batch_idx, (cln_img, targets) in enumerate(pbar):
            cln_img = cln_img.to(device)
            bd_gen_img = bd_gen(cln_img)
            cln_feat = classifier.from_input_to_features(cln_img)
            bd_gen_feat = classifier.from_input_to_features(bd_gen_img)
            loss_benign_feat = mse(feat_mask*cln_feat, feat_mask*bd_gen_feat)
            loss_backdoor_feat = mse((1-feat_mask)*cln_feat, (1-feat_mask)*bd_gen_feat)
            loss_norm = mse(cln_img, bd_gen_img)

            if loss_norm > opt.norm_bound or loss_benign_feat > opt.feat_bound:
                loss = loss_norm
            else:
                loss = -loss_backdoor_feat + 0.01*loss_benign_feat

            opt_bd.zero_grad()
            loss.backward()
            opt_bd.step()
            
            tloss += loss.item()
            tloss_benign_feat += loss_benign_feat.item()
            tloss_backdoor_feat += loss_backdoor_feat.item()
            tloss_norm += loss_norm.item()
            pbar.set_postfix({"epoch": "{:d}".format(u), 
                              "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                              "loss_bengin_feat": "{:.4f}".format(tloss_benign_feat/(batch_idx+1)),
                              "loss_backdoor_feat": "{:.4f}".format(tloss_backdoor_feat/(batch_idx+1)),
                              "loss_norm": "{:.4f}".format(tloss_norm/(batch_idx+1))})
        
        if opt.use_max_bd_feat and tloss_backdoor_feat > max_bd_feat:
            max_bd_feat = tloss_backdoor_feat
            final_bd_gen = deepcopy(bd_gen)

    if opt.use_max_bd_feat and not max_bd_feat is None:
        bd_gen = final_bd_gen

    detected_tlabel = get_target_label(testloader=cln_trainloader, testmodel=classifier, box=box, midmodel=bd_gen)
    print(f"Target Label is {detected_tlabel}")
    return detected_tlabel

if __name__ == "__main__":
    opt = cfg.get_arguments().parse_args()
    opt.use_max_bd_feat = False
    detected_tlabel = main(opt)

