import argparse
import numpy as np
import torch

from utils import get_model, compute_score, cifar10, cifar100, return_path_to_folder


class TransferAttack:
    def __init__(self, model, dataset, n_epochs, opu_output, opu_input, indexes, binary_layer, opu):
        """ Class that performs the trasnfer attack experiments. If indexes are given the transfer attacks are
        evaluated only on samples that were well classified by the studied model."""
        self.model_name = model
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.opu_output = opu_output
        self.opu_input = opu_input
        self.indexes = indexes
        self.binary_layer = binary_layer
        self.opu = opu

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.path_to_folder = return_path_to_folder(self.dataset)
        self.model, self.file_name = self._upload_model()
        self.model.eval()

        if args.indexes:
            self.indexes = torch.load(args.indexes)
        else:
            self.indexes = None

    def my_name(self):
        self.file_name = 'TR_' + self.file_name
        if self.indexes is not None:
            self.file_name = self.file_name + '_idx'
        return self.file_name

    def transfer_attack(self, attack, epsilon):
        dataloader = self._upload_dataloader(attack, epsilon)

        correct_transfer, n = 0, 0

        for i, (x, target) in enumerate(dataloader):
            x, target = x.to(self.device), target.to(self.device)
            if self.indexes is not None:
                indexes_batch = self.indexes[i * dataloader.batch_size: (i + 1) * dataloader.batch_size]
                x, target = x[indexes_batch], target[indexes_batch]

            n += x.shape[0]
            y_hat = torch.argmax(self.model(x), dim=1)
            correct_transfer += (y_hat == target).float().sum()

        final_acc = correct_transfer / n
        return final_acc.item()

    def _upload_model(self):
        """Uploads model to perform the attack on"""

        model, file_name = get_model(model=self.model_name, binary_layer=self.binary_layer, opu_output=self.opu_output,
                                     opu_input=self.opu_input, sign_back=False, n_epochs=self.n_epochs,
                                     device=self.device, dataset=self.dataset, opu=self.opu)

        model.load_state_dict(torch.load(self.path_to_folder + 'models/' + file_name + '.pt'))
        model.eval()
        return model.to(self.device), file_name

    def _upload_dataloader(self, attack, eps):
        """Create dataloader for previously saved perturbed images"""
        dataset = self.path_to_folder.split('/')[1]
        perturbed_img_folder = 'adversarial_data/' + dataset + '/'
        X, y = torch.load(perturbed_img_folder + f'{attack}_{eps}'), torch.load(perturbed_img_folder + f'labels_{eps}')
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
        return dataloader


def pars_args():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("-model", default='VGG-16', type=str, help='Name of the model',
                        choices=['VGG-16', 'VGG16-OPU', 'VGG16-R-OPU', 'Ablation'])
    parser.add_argument("-dataset", default='cifar10', type=str, help='dataset', choices=['cifar10', 'cifar100'])
    parser.add_argument("-n_epochs", default=120, type=int, help='Number of epochs the model has been trained')
    parser.add_argument("-opu_output", default=512, type=int, help='Dimension of OPU output')
    parser.add_argument("-opu_input", default=512, type=int, help='Dimension of OPU output')
    parser.add_argument("-indexes", default=None, type=str, help='Path/file_name to correct indexes')
    # boolean switches.
    parser.add_argument("-binary_layer", default=False, action='store_true', help='If model uses torch.sign()')
    parser.add_argument("-opu", default=False, action='store_true', help='Needed for ablation models')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import pathlib

    args = pars_args()
    attack_transfer = TransferAttack(model=args.model, dataset=args.dataset, n_epochs=args.n_epochs,
                                     opu_output=args.opu_output, opu_input=args.opu_input, indexes=args.indexes,
                                     binary_layer=args.binary_layer, opu=args.opu)

    if args.dataset == 'cifar10':
        _, test_dl = cifar10(batch_size=100)
    else:
        _, test_dl = cifar100(batch_size=100)

    if args.indexes:
        msg = 'Test score={}. With indexes accuracy has to be 100%'
        print(msg.format(
            compute_score(attack_transfer.model, test_dl, attack_transfer.device, indexes=torch.load(args.indexes))))
    else:
        msg = 'Test score={}, to be compared to eps=0 attack'
        print(msg.format(compute_score(attack_transfer.model, test_dl, attack_transfer.device)))

    epsilons = [0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1]
    fgsm_acc, pgd_acc = list(), list()

    for eps in epsilons:
        fgsm_acc.append(attack_transfer.transfer_attack('fgsm', eps))
        pgd_acc.append(attack_transfer.transfer_attack('pgd', eps))
        print(f'{eps},  fgsm = {fgsm_acc[-1]}, pgd = {pgd_acc[-1]}')

    file_name = attack_transfer.my_name()
    path = attack_transfer.path_to_folder + 'transfer/'

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    np.savez(path + file_name, fgsm=fgsm_acc, pgd=pgd_acc, eps=epsilons)
