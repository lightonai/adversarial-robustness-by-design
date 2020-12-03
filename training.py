import argparse
import torch

from utils import get_model, compute_score, cifar10, cifar100, return_path_to_folder


def training(net, train_dl, test_dl, device, n_epochs, optimizer, is_scheduler, milestones):
    if is_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, threshold=0.01)
        if milestones:
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, [45, 60], gamma=0.5)
    print(f'- Total number of epochs = {n_epochs}')
    for e in range(n_epochs):
        loss_stat = 0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
            loss_stat += loss.item()

        loss_stat = loss_stat / len(train_dl)
        if is_scheduler:
            scheduler.step(loss_stat)
            if milestones:
                scheduler2.step()
        train_score = compute_score(net, train_dl, device)
        test_score = compute_score(net, test_dl, device)
        print(
            f'Epoch {e + 1}, loss = {loss_stat:.3f}, train = {train_score:.2f}%, '
            f'test = {test_score:.2f}% lr = {optimizer.param_groups[0]["lr"]}')

    return net


def features_loading(net, dataset, device, features_model):
    " loads the features of a given model, it freezes them during training time to fine-tune the classifier"

    conv_features, _ = get_model(model='Ablation', binary_layer=False, opu=False, n_epochs='_', dataset=dataset,
                                 opu_output=8000, opu_input=1024, sign_back=False, device=device)

    conv_features.load_state_dict(torch.load(path_to_folder + 'models/' + features_model + '.pt'))
    conv_features.eval()
    conv_features = conv_features.state_dict()
    for name, param in net.named_parameters():
        if name.split('.')[0] == 'features':
            param.data = conv_features[name]
            param.requires_grad = False

    print('- Robust features loaded!')

    return net


def exp(model, n_epochs, opu, batch_size, binary_layer, lr, optimizer, weight_decay, smoke_test,
        opu_output, opu_input, is_scheduler, dataset, features_model, milestones):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'- Current torch.device is: {device}')

    net, file_name = get_model(model=model, binary_layer=binary_layer, n_epochs=n_epochs, opu_output=opu_output,
                               opu_input=opu_input, device=device, dataset=dataset, opu=opu)

    if features_model is not None:
        net = features_loading(net, dataset, device, features_model)

    if optimizer == 'SGD':
        print(f'- Optimizer = {optimizer}, Starting lr = {lr}, Momentum = {0.9}, Weight decay = {weight_decay}')
        optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        print(f'- Optimizer = {optimizer}, Starting lr = {lr}, Weight decay = {weight_decay}')
        optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)

    if smoke_test:
        train_samples = 5000
    else:
        train_samples = None

    if dataset == 'cifar10':
        train_dl, test_dl = cifar10(batch_size=batch_size, num_workers=16, subsample=train_samples)
    if dataset == 'cifar100':
        train_dl, test_dl = cifar100(batch_size=batch_size, num_workers=16, subsample=train_samples)

    net = training(net.to(device), train_dl, test_dl, device, n_epochs, optimizer, is_scheduler, milestones)
    train_acc = compute_score(net, train_dl, device)
    test_acc = compute_score(net, test_dl, device)
    print(f'- Train acc {train_acc:.2f}%, Test acc {test_acc:.2f}%')
    return train_acc, test_acc, net, file_name


def pars_args():
    parser = argparse.ArgumentParser('parameters')
    # str
    parser.add_argument("-model", default='VGG16-OPU', type=str, help='Name of the trained net',
                        choices=['VGG-16', 'VGG16-OPU', 'VGG16-R-OPU', 'Ablation'])
    parser.add_argument("-optimizer", default='SGD', type=str, help='Optimizer choice', choices=['SGD', 'ADAM'])
    parser.add_argument("-dataset", default='cifar10', type=str, help='dataset', choices=['cifar10', 'cifar100'])
    parser.add_argument("-features_model", default=None, type=str, help='model name to take features from')
    # int
    parser.add_argument("-n_epochs", default=100, type=int, help='Number of epochs')
    parser.add_argument("-batch_size", default=128, type=int, help='Batch size')
    parser.add_argument("-opu_output", default=512, type=int, help='Dimension of OPU output')
    parser.add_argument("-opu_input", default=512, type=int, help='Dimension of OPU output')
    parser.add_argument("-model_index", default=None, type=int, help='To save multiple models with same hyperparams')
    parser.add_argument("-seed", default=None, type=int, help='Torch/numpy seed to ensure experiments reproducibility')
    # float
    parser.add_argument("-lr", default=1e-3, type=float, help='Learning rate')
    parser.add_argument("-weight_decay", default=0.005, type=float, help='Weight decay for SGD')
    # boolean switches.
    parser.add_argument("-smoke-test", default=False, action='store_true', help='Reduce number of training samples')
    parser.add_argument("-is_scheduler", default=True, action='store_false', help='Deactivates scheduler')
    parser.add_argument("-binary_layer", default=False, action='store_true', help='Binary layer is active')
    parser.add_argument("-opu", default=False, action='store_true', help='Needed for ablation models')
    parser.add_argument("-milestones", default=False, action='store_true', help='Needed for ablation models')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import numpy as np
    import pathlib

    args = pars_args()
    model_index = None

    if args.model_index:
        model_index = args.model_index
    if args.seed is not None:
        print(f'- Manual seed = {args.seed}')
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    del args.model_index
    del args.seed

    path_to_folder = return_path_to_folder(args.dataset)
    pathlib.Path(path_to_folder + 'accuracies').mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_to_folder + 'models').mkdir(parents=True, exist_ok=True)

    train_accuracy, test_accuracy, model, file_name = exp(model=args.model, n_epochs=args.n_epochs,
                                                          batch_size=args.batch_size,
                                                          binary_layer=args.binary_layer, lr=args.lr, opu=args.opu,
                                                          optimizer=args.optimizer,
                                                          weight_decay=args.weight_decay, smoke_test=args.smoke_test,
                                                          opu_output=args.opu_output, opu_input=args.opu_input,
                                                          is_scheduler=args.is_scheduler, dataset=args.dataset,
                                                          features_model=args.features_model,
                                                          milestones=args.milestones)

    if model_index:
        file_name = file_name + f'__{model_index}'

    np.savez(path_to_folder + 'accuracies/' + file_name, train_accuracy=train_accuracy, test_accuracy=test_accuracy)
    torch.save(model.state_dict(), path_to_folder + 'models/' + file_name + '.pt')
    print('model and accuracies saved.')
