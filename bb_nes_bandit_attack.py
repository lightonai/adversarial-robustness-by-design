import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from black_box_attack_utils import make_adversarial_examples
from utils import get_model, return_path_to_folder, cifar10, cifar100


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class Classifier(nn.Module):
    def __init__(self, model):
        super(Classifier, self).__init__()
        self.norm = Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])

        self.model = model

    def forward(self, x):
        return self.model(self.norm(x))


def perfect_dataloader(model, test_dl, device, batch_size):
    # taking the well predicted images of test set for Vanilla
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 12}

    correctly_classified_indexes = []

    with torch.no_grad():
        for i, (x, y) in enumerate(test_dl):
            inputs, labels = x.to(device), y.to(device)
            outputs = model(inputs).to(device)
            _, predicted = torch.max(outputs.data, 1)
            correctly_classified_indexes.append(np.where((predicted == labels).double().cpu().numpy())[0])

    correctly_classified_indexes = list(itertools.chain(*correctly_classified_indexes))
    test_dl_perfect = torch.utils.data.Subset(test_dl.dataset, correctly_classified_indexes)
    test_loader_perfect = torch.utils.data.DataLoader(test_dl_perfect, **params)
    return test_loader_perfect


def bandit_attack(model, test_dl, device, fd_eta, nes, exploration, gradient_iters):
    success = np.array([])
    budget = np.array([])

    for images, labels in tqdm(test_dl):
        images = images.to(device)
        labels = labels.to(device)
        adv_ex = make_adversarial_examples(images, labels, model, nes=nes,
                                           loss="xent", mode="linf", epsilon=8. / 256, max_queries=15000,
                                           gradient_iters=gradient_iters, fd_eta=fd_eta, image_lr=0.01, online_lr=0.1,
                                           exploration=exploration, prior_size=16, targeted=False,
                                           log_progress=True, device=device)

        success = np.concatenate((success, adv_ex["success"]))
        budget = np.concatenate((budget, adv_ex["elapsed_budget"]))

    return success, budget


def upload_model(path_to_folder, model, dataset, binary_layer, opu_output, opu_input, n_epochs, device):
    """Uploads model to perform the attack on"""

    model, file_name = get_model(model=model, dataset=dataset, binary_layer=binary_layer, opu_output=opu_output,
                                 opu_input=opu_input, n_epochs=n_epochs, device=device, opu=False)

    model.load_state_dict(torch.load(path_to_folder + 'models/' + file_name + '.pt'))
    model = Classifier(model)
    model.eval()
    return model.to(device), file_name


def pars_args():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("-model", default='VGG-16', type=str, help='Name of the trained net',
                        choices=['VGG-16', 'VGG16-OPU', 'VGG16-R-OPU', 'Ablation'])
    parser.add_argument("-dataset", default='cifar10', type=str, help='Name of dataset',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument("-n_epochs", default=55, type=int, help='Number of epochs')
    parser.add_argument("-gradient_iters", default=1, type=int, help='Gradient iters')
    parser.add_argument("-exploration", default=0.1, type=float, help='exploration')
    parser.add_argument("-batch_size", default=1024, type=int, help='Batch size')
    parser.add_argument("-opu_output", default=512, type=int, help='Dimension of OPU output')
    parser.add_argument("-opu_input", default=512, type=int, help='Dimension of OPU input')
    # boolean switches.
    parser.add_argument("-binary_layer", default=False, action='store_true', help='Binary layer is active')
    parser.add_argument("-nes", default=False, action='store_true', help='If bb is NES')
    args = parser.parse_args("")
    return args


if __name__ == '__main__':
    import pathlib

    args = pars_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_to_folder = return_path_to_folder(args.dataset)
    path = path_to_folder + 'black_box/'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    if args.nes:
        attack = 'NES_'
    else:
        attack = 'Bandit_'

    if args.dataset == 'cifar10':
        _, test_dl = cifar10(batch_size=100)
    else:
        _, test_dl = cifar100(batch_size=100)

    model, model_name = upload_model(path_to_folder=path_to_folder, model=args.model, dataset=args.dataset,
                                     binary_layer=args.binary_layer, opu_output=args.opu_output,
                                     opu_input=args.opu_input, n_epochs=args.n_epochs, device=device)

    perfect_dl = perfect_dataloader(model, test_dl, device, args.batch_size)

    for fd_eta in [0.1, 0.5, 1.]:
        success, budget = bandit_attack(model, perfect_dl, device=device, fd_eta=fd_eta, nes=args.nes,
                                        exploration=args.exploration, gradient_iters=args.gradient_iters)
        budget_sort = np.sort(budget[success == 1])
        cum_succ_rate = np.arange(len(budget_sort)) / len(success)
        file_name = attack + model_name + f'_bs_{args.batch_size}_maxquery_{15000}_fdeta_{fd_eta}'
        np.save(path + file_name, success=success, budget=budget, budget_sort=budget_sort,
                cum_succ_rate=cum_succ_rate)
