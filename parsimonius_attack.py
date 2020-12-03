import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets

from utils import get_model, return_path_to_folder
from black_box_attack_utils import ParsimoniousAttack


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class Classifier(nn.Module):
    def __init__(self, model, dataset_name):
        super(Classifier, self).__init__()
        if dataset_name == "cifar10":
            self.norm = Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
        elif dataset_name == "cifar100":
            self.norm = Normalize(mean=[0.507, 0.486, 0.440],
                                  std=[0.267, 0.256, 0.276])

        self.model = model

    def forward(self, x):
        return self.model(self.norm(x))


def eval_parsimonious(dataset_name, model, model_name, device):
    classifier = Classifier(model, dataset_name)
    classifier.to(device)
    classifier.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            "data/cifar10", train=False, download=True, transform=transform)
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            "data/cifar100", train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset, num_workers=4,
                                              batch_size=1,
                                              shuffle=False)

    success = np.array([])
    budget = np.array([])
    i = 0
    correctly_classified = 0

    for images, labels in test_loader:
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)

            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            correctly_classified_indexes = np.where(
                (predicted == labels).double().cpu().numpy())[0]
            correctly_classified += len(correctly_classified_indexes)

            if len(correctly_classified_indexes) > 0:
                images, labels = images.cpu(), labels.cpu()
                attack = ParsimoniousAttack(classifier,
                                            targeted=False,
                                            loss_func="xent",
                                            max_queries=20000,
                                            epsilon=8. / 256,
                                            batch_size=64,
                                            block_size=4,
                                            no_hier=False,
                                            max_iters=1)
                bla = attack.perturb(images[correctly_classified_indexes],
                                     labels[correctly_classified_indexes])
                success = np.concatenate((success, bla["success"]))
                budget = np.concatenate((budget, bla["elapsed_budget"]))
                image_adv = bla["image_adv"]
                _, label_adv = torch.max(classifier(image_adv.to(device)), 1)

                i += 1

    return success, budget


def upload_model(path_to_folder, model_name, binary_layer, opu_output, opu_input, n_epochs, device, dataset):
    """Uploads model to perform the attack on"""

    model, file_name = get_model(model=model_name, binary_layer=binary_layer, opu_output=opu_output,
                                 opu_input=opu_input, n_epochs=n_epochs, device=device, dataset=dataset,
                                 opu=False, sign_back=False)

    model.load_state_dict(torch.load(path_to_folder + 'models/' + file_name + '.pt'))
    model.eval()
    return model.to(device), file_name


def pars_args():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("-model", default='VGG-16', type=str, help='Name of the trained net',
                        choices=['VGG-16', 'VGG16-OPU', 'VGG16-R-OPU', 'Ablation'])
    parser.add_argument("-dataset", default='cifar10', type=str, help='Name of dataset',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument("-n_epochs", default=55, type=int, help='Number of epochs')
    parser.add_argument("-opu_output", default=512, type=int, help='Dimension of OPU output')
    parser.add_argument("-opu_input", default=512, type=int, help='Dimension of OPU input')
    # boolean switches.
    parser.add_argument("-binary_layer", default=False, action='store_true', help='Binary layer is active')
    args = parser.parse_args("")
    return args


if __name__ == '__main__':
    import pathlib

    args = pars_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_to_folder = return_path_to_folder(args.dataset)
    path = path_to_folder + 'black_box/'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    model, model_name = upload_model(path_to_folder=path_to_folder, model_name=args.model, dataset=args.dataset,
                                     binary_layer=args.binary_layer, opu_output=args.opu_output,
                                     opu_input=args.opu_input, n_epochs=args.n_epochs, device=device)

    success, budget = eval_parsimonious(dataset_name=args.dataset, model=model, model_name=model_name, device=device)
    np.savez(path + f'parsimonius_{model_name}', success=success, budget=budget)
