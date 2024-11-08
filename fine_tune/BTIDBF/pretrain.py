import torch
from loader import Box
from models.unet_model import UNet
from evaluate import test
from tqdm import tqdm
import cfg
import os

if __name__ == "__main__":
    opt = cfg.get_arguments().parse_args()
    device = opt.device
    box = Box(opt)
    save_path = box.get_save_path()
    if opt.dataset == 'cifar':
        folder_path = f'../../record/cifar10/{opt.attack}/pratio_{opt.pratio}-target_{opt.attack_target}-archi_{opt.model}'
    else:
        folder_path = f'../../record/{opt.dataset}/{opt.attack}/pratio_{opt.pratio}-target_{opt.attack_target}-archi_{opt.model}'
    print(f'Load from {folder_path}')
    _, _, classifier = box.get_state_dict(folder_path + '/attack_result.pt')

    cln_trainloader = box.get_dataloader(train="clean", batch_size=opt.batch_size, shuffle=True)
    cln_testloader = box.get_dataloader(train="test", batch_size=opt.batch_size, shuffle=False)

    generator = UNet(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=opt.gen_lr)

    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax()

    for p in range(opt.preround):
        pbar = tqdm(cln_trainloader, desc="Pretrain Generator")
        generator.train()
        classifier.eval()
        tloss = 0
        tloss_pred = 0
        tloss_feat = 0
        tloss_norm = 0
        for batch_idx, (cln_img, targets) in enumerate(pbar):
            cln_img = cln_img.to(device)
            pur_img = generator(cln_img)
            cln_feat = classifier.from_input_to_features(cln_img)
            pur_feat = classifier.from_input_to_features(pur_img)
            cln_out = classifier.from_features_to_output(cln_feat)
            pur_out = classifier.from_features_to_output(pur_feat)
            loss_pred = ce(softmax(cln_out), softmax(pur_out))
            loss_feat = mse(cln_feat, pur_feat)
            loss_norm = mse(cln_img, pur_img)

            if loss_norm > 0.1:
                loss = 1*loss_pred + 1*loss_feat + 100*loss_norm
            else:
                loss = loss_pred + 1*loss_feat + 0.01*loss_norm

            opt_g.zero_grad()
            loss.backward()
            opt_g.step()
            
            tloss += loss.item()
            tloss_pred += loss_pred.item()
            tloss_feat += loss_feat.item()
            tloss_norm += loss_norm.item()

            pbar.set_postfix({"epoch": "{:d}".format(p), 
                              "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                              "loss_pred": "{:.4f}".format(tloss_pred/(batch_idx+1)),
                              "loss_feat": "{:.4f}".format(tloss_feat/(batch_idx+1)),
                              "loss_norm": "{:.4f}".format(tloss_norm/(batch_idx+1))})
            
        if (p+1) % 10 == 0 :
            test(testloader=cln_testloader, testmodel=classifier, box=box, midmodel=generator, name="BA")
            parent_path = os.path.join(save_path, "pretrain")
            if not os.path.exists(parent_path):
                os.mkdir(parent_path)
            torch.save(generator.state_dict(), os.path.join(parent_path, f"{opt.model}_{opt.attack_name}_{opt.pratio}_init_generator.pt"))

