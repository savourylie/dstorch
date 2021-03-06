import torch
from tqdm import tqdm, tqdm_notebook

def calc_data_stats(data_loader, dtype='image', repeat=10):
    batch_size = data_loader.batch_size

    sample_size = 0
    mean_list = []
    
    try:
        _ = __IPYTHON__

    except NameError:        
        for _ in tqdm(range(repeat)):
            for data in data_loader:
                x, y = data
                x = x.view(x.size(0), x.size(1), -1)
                mean_list.append(x.mean(2).mean(0))

                sample_size += x.shape[0] 
    else:
        for _ in tqdm_notebook(range(repeat)):
            for data in data_loader:
                x, y = data
                x = x.view(x.size(0), x.size(1), -1)
                mean_list.append(x.mean(2).mean(0))

                sample_size += x.shape[0] 
        
    mean = torch.stack(mean_list).mean(dim=0)
    se = torch.stack(mean_list).std(dim=0)
    std = se * torch.sqrt(torch.tensor(float(batch_size)))

    print("=============================")
    print("Dataset size: {}".format(sample_size))
    print("Batch size: {}".format(batch_size))
    print("Mean: {}".format(mean))
    print("STD: {}".format(std))

    return mean, std