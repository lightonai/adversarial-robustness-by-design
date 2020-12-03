import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np

from vgg16_models import cifar_vgg16, ablation_cifar_vgg16


def get_model(model, binary_layer, opu, n_epochs, dataset, opu_output, opu_input, sign_back=False, device='cpu'):
    print(f'You are creating model: {model}')
    net, file_name = None, f'model_{model}_nepochs_{n_epochs}'
    f'res__model_{model}_nepochs_{n_epochs}_bl__{binary_layer}'

    if model == 'VGG-16':
        net = cifar_vgg16(opu=False, binary_layer=False, opu_features=512, opu_input=512, device=device, dataset=dataset,
                          sign_back=sign_back)
        file_name = file_name + f'_bl__{binary_layer}_opu_{False}'

    if model == 'VGG16-OPU':
        net = cifar_vgg16(opu=True, binary_layer=binary_layer, opu_features=opu_output, opu_input=opu_input,
                          sign_back=sign_back, device=device, dataset=dataset)
        file_name = file_name + f'_bl__{binary_layer}_opu_{True}_oo_{opu_output}_oi_{opu_input}'

    if model == 'VGG16-R-OPU':
        net = cifar_vgg16(opu=True, binary_layer=True, opu_features=opu_output, opu_input=opu_input,
                          sign_back=sign_back, real_opu=True, device=device, dataset=dataset)
        file_name = file_name + f'_bl__{True}_opu_{True}_oo_{opu_output}_oi_{opu_input}'

    if model == 'Ablation':
        net = ablation_cifar_vgg16(opu=opu, binary_layer=binary_layer, opu_features=8000, opu_input=1024, dataset=dataset,
                                   sign_back=sign_back)
        file_name = f'model_{ablation_model_name(binary_layer, opu)}_nepochs_{n_epochs}'
        file_name = file_name + f'_bl__oo_{opu_output}_oi_{opu_input}'

    return net.to(device), file_name


def compute_score(net, dl, device, save_correct_labels=False, indexes=None):
    net_training_mode = net.training
    if save_correct_labels:
        print('dl size must be divisible by batch size! e.g. for cifar10 test set -> batch size = 100.')
        correct_indexes = torch.zeros(dl.batch_size * len(dl))
    if net_training_mode:
        net.eval()
    with torch.no_grad():
        score, n = 0, 0
        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            if indexes is not None:
                indexes_batch = indexes[i * dl.batch_size: (i + 1) * dl.batch_size]
                x, y = x[indexes_batch], y[indexes_batch]
            y_hat = torch.argmax(net(x), dim=1)
            score += (y_hat == y).float().sum()
            n += x.shape[0]
            if save_correct_labels:
                correct_indexes[i * x.shape[0]:(i + 1) * x.shape[0]] = y_hat == y
    if net_training_mode:  # if model was originally in train mode you don't want to change that.
        net.train()
    if save_correct_labels:
        return (score / n).item() * 100, correct_indexes
    else:
        return (score / n).item() * 100


def cifar10(batch_size, num_workers=0, subsample=None):
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    t = transforms.Compose([transforms.ToTensor(), norm])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=t)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=t)
    train_shuffle, train_sampler = True, None
    if subsample:
        indices = list(range(len(trainset)))
        np.random.seed(1234)
        np.random.shuffle(indices)
        train_idx = indices[:subsample]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_shuffle = False

    c10_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=train_shuffle,
                                            num_workers=num_workers, sampler=train_sampler)
    c10_test = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return c10_train, c10_test


def cifar100(batch_size, num_workers=0, subsample=None):
    norm = transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))
    t = transforms.Compose([transforms.ToTensor(), norm])
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
         transforms.ToTensor(), norm])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=t)
    train_shuffle, train_sampler = True, None
    if subsample:
        indices = list(range(len(trainset)))
        np.random.seed(1234)
        np.random.shuffle(indices)
        train_idx = indices[:subsample]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_shuffle = False
    c100_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=train_shuffle,
                                             num_workers=num_workers, sampler=train_sampler)
    c100_test = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return c100_train, c100_test


def return_path_to_folder(dataset):
    if dataset == 'cifar10':
        path = 'results/c10/'
    else:
        path = 'results/c100/'
    return path


def ablation_model_name(bin, opu):
    if bin:
        if opu:
            model_name = 'OPU'
        else:
            model_name = 'BIN'
    else:
        if opu:
            model_name = 'RP'
        else:
            model_name = 'DFA'
    print(model_name)
    return model_name
