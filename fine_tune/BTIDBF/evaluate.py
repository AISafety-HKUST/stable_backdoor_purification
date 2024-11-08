from copy import deepcopy
import torch
import torchvision
import numpy as np
from tqdm import tqdm

def test(testloader, testmodel, box, poisoned=False, poitarget=False , midmodel = None, passlabel=None, feat_mask=None, name="BA"):
    model = deepcopy(testmodel)
    model.eval()        
    correct = 0
    total = 0

    if poisoned:
        param1, param2, _ = box.get_state_dict()

    pbar = tqdm(testloader, desc="Test")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(box.device), targets.to(box.device)
            ori_target = targets
            if poisoned:
                inputs = box.poisoned(inputs, param1, param2)

            if not midmodel is None:
                tmp_model = deepcopy(midmodel)
                tmp_model.eval()
                gnoise = 0.03 * torch.randn_like(inputs, device=box.device)
                inputs = tmp_model(inputs + gnoise)
                del tmp_model

            if poitarget:
                if box.attack_type == "all2all":
                    targets = torch.remainder(targets+1, box.num_classes).to(box.device)
                elif box.attack_type == "all2one":
                    targets = torch.ones_like(targets, device=box.device) * box.tlabel

            if not feat_mask is None:
                feat = model.from_input_to_features(inputs)
                outputs = model.from_features_to_output(feat_mask*feat)
            else:
                outputs = model(inputs)

            _, predicted = outputs.max(1)

            for i in range(inputs.shape[0]):
                if (not passlabel is None) and ori_target[i] == passlabel:
                    continue
                total += 1
                p = predicted[i]
                t = targets[i]
                if p == t:
                    correct += 1

            if total > 0:
                acc = 100.*correct/total
            else:
                acc = 0

            pbar.set_postfix({name: "{:.4f}".format(acc)})

    return 100.*correct/total

def get_target_label(testloader, testmodel, box, midmodel = None):
    model = deepcopy(testmodel)
    model.eval()        
    reg = np.zeros([box.num_classes])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(box.device), targets.to(box.device)
            if not midmodel is None:
                tmodel = deepcopy(midmodel)
                tmodel.eval()
                gnoise = 0.03 * torch.randn_like(inputs, device=box.device)
                inputs = tmodel(inputs + gnoise)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(inputs.shape[0]):
                p = predicted[i]
                reg[p] += 1
                # t = targets[i]
                # if p == t:
                #     reg[t] += 1
                    
    return np.argmax(reg)

