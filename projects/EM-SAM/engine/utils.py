import torch

def mask2one_hot(label, out):
    num_classes = out.shape[1]
    current_label = label.squeeze(1)  # （batch_size, 1, h, w) ---> （batch_size, h, w)
    batch_size, h, w = current_label.shape[0], current_label.shape[1], current_label.shape[2]
    one_hots = []
    for i in range(num_classes):
        tmplate = torch.ones(batch_size, h, w)  # （batch_size, h, w)
        tmplate[current_label != i] = 0
        tmplate = tmplate.view(batch_size, 1, h, w)  # （batch_size, h, w) --> （batch_size, 1, h, w)
        one_hots.append(tmplate)
    onehot = torch.cat(one_hots, dim=1).long()
    return list(onehot.unsqueeze(0))  # return list