import torch
import torch.nn as nn
def test(model, test_data, device,multi=False):

    model.eval()
    
    metrics = {
        'test_correct': 0,
        'test_loss': 0,
        'test_total': 0
    }
    criterion = nn.CrossEntropyLoss()
    tot_tar_list = []
    cor_tar_list = []
    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(test_data):
            x = x.to(device)
            target = target.to(device)
            if multi:
                pred,_ = model(x)
            else:
                pred = model(x)
            loss = criterion(pred, target.long())

            _, predicted = torch.max(pred, -1)
            correct_mask = predicted.eq(target)
            for cor, tar in zip(correct_mask,target):
                tot_tar_list.append(int(tar.item()))
                if cor:
                    cor_tar_list.append(int(tar.item()))
            correct = correct_mask.sum()
            metrics['test_correct'] += correct.item()
            metrics['test_loss'] += loss.item() * target.size(0)
            metrics['test_total'] += target.size(0)
            
    return metrics


