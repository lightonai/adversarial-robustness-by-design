import argparse
import numpy as np
import torch

from utils import get_model, compute_score, cifar10, cifar100, return_path_to_folder


class AdversarialAttack:
    def __init__(self, model, dataset, n_epochs, step_size, opu_output, opu_input, pgd_attacks_iterations, binary_layer,
                 save_images, sign_back, opu):
        """ Class that performs the adversarial attack (both FGSM and PGD), if save_images is True it saves the images
        for further experiments; i.e. transfer attack"""
        self.model_name = model
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.step_size = step_size
        self.opu_output = opu_output
        self.opu_input = opu_input
        self.pgd_attacks_iterations = pgd_attacks_iterations
        self.binary_layer = binary_layer
        self.save_images = save_images
        self.sign_back = sign_back
        self.opu = opu

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path_to_folder = return_path_to_folder(args.dataset)
        self.model, self.file_name = self._upload_model()

    def attack(self, dataloader, epsilon):
        correct_fgsm, correct_PGD, n = 0, 0, 0
        if self.save_images:
            saved_FGSM, saved_PGD = torch.zeros(10000, 3, 32, 32), torch.zeros(10000, 3, 32, 32)
            saved_labels = torch.zeros(10000)
            dataset = self.path_to_folder.split('/')[1]
            images_path = 'adversarial_data/' + dataset + '/'
            pathlib.Path(images_path).mkdir(parents=True, exist_ok=True)

        sanity_check = True
        for i, (x, target) in enumerate(dataloader):
            x, target = x.to(self.device), target.to(self.device)
            if sanity_check:
                print(f'- Check origianl clamp, x max = {x.max()}, x min = {x.min()}')

            n += x.shape[0]
            x.requires_grad = True
            perturbed_x_fgsm = self._projected_gradient_descent(x=x, y=target, eps=epsilon, sanity_check=sanity_check,
                                                                num_steps=1, clamp=(x.min().item(), x.max().item()),
                                                                step_size=self.step_size)
            perturbed_x_PGD = self._projected_gradient_descent(x=x, y=target, eps=epsilon, sanity_check=sanity_check,
                                                               num_steps=self.pgd_attacks_iterations,
                                                               clamp=(x.min().item(), x.max().item()),
                                                               step_size=self.step_size)

            if self.save_images:
                saved_FGSM[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = perturbed_x_fgsm.to('cpu')
                saved_PGD[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = perturbed_x_PGD.to('cpu')
                saved_labels[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = target.to('cpu')

            y_fgsm, y_PGD = self.model(perturbed_x_fgsm), self.model(perturbed_x_PGD)
            y_fgsm, y_PGD = torch.argmax(y_fgsm, dim=1), torch.argmax(y_PGD, dim=1)
            correct_fgsm += (y_fgsm == target).float().sum()
            correct_PGD += (y_PGD == target).float().sum()
            sanity_check = False

        final_acc_fgsm = correct_fgsm / n
        final_acc_PGD = correct_PGD / n

        if self.save_images:
            torch.save(saved_FGSM, images_path + f'fgsm_{epsilon}')
            torch.save(saved_PGD, images_path + f'pgd_{epsilon}')
            torch.save(saved_labels, images_path + f'labels_{epsilon}')

        return final_acc_fgsm.item(), final_acc_PGD.item()

    def _projected_gradient_descent(self, x, y, eps, sanity_check, num_steps, step_size, clamp=(-1, 1)):
        """Performs the projected gradient descent attack on a batch of images."""
        x_adv = x.clone().detach().requires_grad_(True).to(x.device)

        for i in range(num_steps):
            _x_adv = x_adv.clone().detach().requires_grad_(True)

            prediction = self.model(_x_adv)
            self.model.zero_grad()
            loss = torch.nn.functional.cross_entropy(prediction, y)
            loss.backward()

            with torch.no_grad():
                if num_steps == 1:
                    x_adv += eps * _x_adv.grad.data.sign()
                    if eps == 0 and sanity_check:
                        print(f'-Sanity check after grad adding: {torch.equal(x_adv, x)}')
                else:
                    x_adv += _x_adv.grad.data.sign() * step_size
                    x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)

            x_adv = x_adv.clamp(*clamp)
            if eps == 0 and sanity_check:
                print(f'- Sanity check after clamping: {torch.equal(x_adv, x)}')
            sanity_check = False
        return x_adv.detach()

    def _upload_model(self):
        """Uploads model to perform the attack on"""

        model, file_name = get_model(model=self.model_name, binary_layer=self.binary_layer, opu_output=self.opu_output,
                                     opu_input=self.opu_input, sign_back=self.sign_back, n_epochs=self.n_epochs,
                                     device=self.device, dataset=self.dataset, opu=self.opu)

        model.load_state_dict(torch.load(self.path_to_folder + 'models/' + file_name + '.pt'))
        print(file_name)
        model.eval()
        return model.to(self.device), file_name


def pars_args():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("-model", default='VGG16-R-OPU', type=str, help='Name of the model',
                        choices=['VGG-16', 'VGG16-OPU', 'VGG16-R-OPU', 'Ablation'])
    parser.add_argument("-dataset", default='cifar10', type=str, help='dataset', choices=['cifar10', 'cifar100'])
    parser.add_argument("-n_epochs", default=120, type=int, help='Number of epochs the model has been trained')
    parser.add_argument("-step_size", default=0.01, type=float,
                        help='Learning rate with which the model has been trained')
    parser.add_argument("-opu_output", default=512, type=int, help='Dimension of OPU output')
    parser.add_argument("-opu_input", default=512, type=int, help='Dimension of OPU output')
    parser.add_argument("-pgd_attacks_iterations", default=50, type=int, help='Number of iterations for PGD attack.')
    # boolean switches.
    parser.add_argument("-binary_layer", default=False, action='store_true', help='To activate binary layer')
    parser.add_argument("-save_images", default=False, action='store_true', help='Saves images for TA')
    parser.add_argument("-sign_back", default=False, action='store_true', help='Replace sign by identity for BDSM')
    parser.add_argument("-opu", default=False, action='store_true', help='Needed for ablation models')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import pathlib

    args = pars_args()
    adversarial_attack = AdversarialAttack(args.model, args.dataset, args.n_epochs, args.step_size, args.opu_output,
                                           args.opu_input, args.pgd_attacks_iterations, args.binary_layer,
                                           args.save_images,
                                           args.sign_back, args.opu)
    if args.dataset == 'cifar10':
        _, test_dl = cifar10(batch_size=100, num_workers=8)  # if it gives trouble remove num_workers
    else:
        _, test_dl = cifar100(batch_size=100, num_workers=8)  # same as above

    print(
        f'Test score={compute_score(adversarial_attack.model, test_dl, adversarial_attack.device)}, to be compared to eps=0 attack')
    epsilons = [0, .01, .02, .03, .04, .05]
    if args.save_images:
        epsilons = epsilons + [0.06, 0.07, 0.08, 0.09, 0.1]  # images are saved for transfer attack experiments.
    accuracies = list()

    for eps in epsilons:
        acc_test = adversarial_attack.attack(test_dl, eps)
        accuracies.append(acc_test)
        print(eps, accuracies[-1])

    file_name = 'AA_' + adversarial_attack.file_name
    path = adversarial_attack.path_to_folder + 'attacks/'
    if args.sign_back:
        file_name = file_name + '_sign_back'

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    np.savez(path + file_name, accuracies=accuracies, eps=epsilons)
