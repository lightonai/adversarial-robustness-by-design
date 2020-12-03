from utils import compute_score, cifar10, cifar100, get_model
import torch

_, c10_test = cifar10(batch_size=100)
_, c100_test = cifar100(batch_size=100)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
useless_boolean = None


def acc_and_idx(model, binary_layer, opu, n_epochs, opu_output, opu_input, device, dataset):
    net, net_name = get_model(model=model, binary_layer=binary_layer, opu=opu, n_epochs=n_epochs, opu_output=opu_output,
                              opu_input=opu_input, sign_back=False, device=device, dataset=dataset)
    path = 'results/c' + dataset[5:] + '/models/'

    net.load_state_dict(torch.load(path + net_name + '.pt'))
    net.eval()
    if dataset == 'cifar10':
        dataloader = c10_test
    else:
        dataloader = c100_test
    acc, indexes = compute_score(net, dataloader, device=device, save_correct_labels=True)
    return acc, indexes


# Transfer experiments
print('Transfer')
acc, idx_TA_low = acc_and_idx('VGG16-OPU', True, useless_boolean, 2, 8000, 1024, device, 'cifar10')
print('Fine-tuned, low accuracy, acc', acc)
acc, idx_TA_hig = acc_and_idx('VGG16-OPU', True, useless_boolean, 5, 8000, 1024, device, 'cifar10')
print('Fine-tuned, high accuracy, acc', acc)
acc, idx_baseli = acc_and_idx('VGG-16', False, useless_boolean, 38, 512, 512, device, 'cifar10')
print('Fine-tuned, low accuracy, acc', acc)

idx0 = torch.logical_and(idx_TA_hig, idx_TA_low)
idx = torch.logical_and(idx_baseli, idx0)
print(f'Transfer: Double check: {idx.float().mean()}')
torch.save(idx, 'results/c10/transfer_indexes')

# Ablation study
print('Ablation')
acc, idx_DFA = acc_and_idx('Ablation', False, False, 19, 8000, 1024, device, 'cifar10')
print('DFA', acc)
acc, idx_BIN = acc_and_idx('Ablation', True, False, 19, 8000, 1024, device, 'cifar10')
print('BIN', acc)
acc, idx_RP = acc_and_idx('Ablation', False, True, 22, 8000, 1024, device, 'cifar10')
print('RP', acc)
acc, idx_OPU = acc_and_idx('Ablation', True, True, 19, 8000, 1024, device, 'cifar10')
print('OPU', acc)

idx0 = torch.logical_and(idx_DFA, idx_BIN)
idx1 = torch.logical_and(idx_RP, idx_OPU)
idx = torch.logical_and(idx0, idx1)
print(f'Ablation: Double check: {idx.float().mean()}')
torch.save(idx, 'results/c10/ablation_indexes')

# Transfer experiments CIFAR100
print('Transfer CIFAR100')
acc, idx_TA_low = acc_and_idx('VGG16-OPU', True, useless_boolean, 4, 8000, 1024, device, 'cifar100')
print('Fine-tuned, low accuracy, acc', acc)
acc, idx_TA_hig = acc_and_idx('VGG16-OPU', True, useless_boolean, 5, 8000, 1024, device, 'cifar100')
print('Fine-tuned, high accuracy, acc', acc)
acc, idx_baseli = acc_and_idx('VGG-16', False, useless_boolean, 105, 512, 512, device, 'cifar100')
print('Fine-tuned, low accuracy, acc', acc)

idx0 = torch.logical_and(idx_TA_hig, idx_TA_low)
idx = torch.logical_and(idx_baseli, idx0)
print(f'Transfer: Double check: {idx.float().mean()}')
torch.save(idx, 'results/c100/transfer_indexes')
