import torch

def int2onehot(target, num_classes):
    batch_size = target.shape[0]
    target_onehot = torch.FloatTensor(batch_size, num_classes)
    target_onehot.zero_()
    target_onehot.scatter_(1, target, 1)
    
    return target_onehot