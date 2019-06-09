import torch
from torch.autograd import grad

def calc_jacobian(model, img):
    output = model(img)

    output_reshaped = output.view(-1, output.size(1) * output.size(2) * output.size(3))

    jacobian = torch.stack([grad([output_reshaped[:, i].sum()], [img], retain_graph=True, create_graph=True)[0].view(-1, img.size(1) * img.size(2) * img.size(3))[:, i]
                            for i in range(len(output_reshaped))])

    return jacobian

# def calc_frobenius_norm(matrix):

#     return torch.sqrt((matrix ** 2).sum((1, 2))

def calc_gradient_norm(model):
	for p in model.parameters():
	            param_norm = p.grad.data.norm(2)
	            total_norm += param_norm.item() ** 2
	        total_norm = total_norm ** (1. / 2)

	return total_norm