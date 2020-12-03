import itertools
import heapq
import math
import numpy as np
import sys
import torch
from torch.nn.modules import Upsample


def replicate_input(x):
    return torch.as_tensor(x).detach().clone()


def to_one_hot(y, num_classes=10):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    y = replicate_input(y).view(-1, 1)

    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot


def norm(t):
    assert len(t.shape) == 4
    norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float() * 1e-8
    return norm_vec


###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###


def eg_step(x, g, lr):
    real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
    pos = real_x * torch.exp(lr * g)
    neg = (1 - real_x) * torch.exp(-lr * g)
    new_x = pos / (pos + neg)
    return new_x * 2 - 1


def linf_step(x, g, lr):
    return x + lr * torch.sign(g)


def l2_prior_step(x, g, lr):
    new_x = x + lr * g / norm(g)
    norm_new_x = norm(new_x)
    norm_mask = (norm_new_x < 1.0).float()
    return new_x * norm_mask + (1 - norm_mask) * new_x / norm_new_x


def gd_prior_step(x, g, lr):
    return x + lr * g


def l2_image_step(x, g, lr):
    return x + lr * g / norm(g)


##
# Projection steps for l2 and linf constraints:
# All take the form of func(new_x, old_x, epsilon)
##


def l2_proj(image, eps):
    orig = image.clone()

    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps * delta / norm(delta)) * out_of_bounds_mask
        x += new_x * (1 - out_of_bounds_mask)
        return x

    return proj


def linf_proj(image, eps):
    orig = image.clone()

    def proj(new_x):
        return orig + torch.clamp(new_x - orig, -eps, eps)

    return proj


##
# Main functions
##


def make_adversarial_examples(image, true_label, model_to_fool,
                              nes=False, loss="xent", mode="linf", epsilon=8. / 256, max_queries=50000,
                              gradient_iters=1, fd_eta=0.1, image_lr=0.0001, online_lr=100,
                              exploration=0.01, prior_size=15, targeted=False,
                              log_progress=True, device='cpu'):
    '''
    The main process for generating adversarial examples with priors.
    '''

    with torch.no_grad():
        # Initial setup
        batch_size = image.size(0)
        total_queries = torch.zeros(batch_size).to(device)
        upsampler = Upsample(size=(image.size(2), image.size(2)))
        prior = torch.zeros(batch_size, 3, prior_size, prior_size).to(device)

        dim = prior.nelement() / batch_size
        prior_step = gd_prior_step if mode == 'l2' else eg_step
        image_step = l2_image_step if mode == 'l2' else linf_step
        proj_maker = l2_proj if mode == 'l2' else linf_proj
        proj_step = proj_maker(image, epsilon)

        # Loss function
        if targeted:
            criterion = -torch.nn.CrossEntropyLoss(reduction='none')
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')

        losses = criterion(model_to_fool(image), true_label)

        # Original classifications
        orig_images = image.clone()
        orig_classes = model_to_fool(image).argmax(1).to(device)
        correct_classified_mask = (orig_classes == true_label).to(device).float()
        total_ims = correct_classified_mask.to(device).sum()
        not_dones_mask = correct_classified_mask.to(device).clone()

        t = 0

        while not torch.any(total_queries > max_queries):
            t += gradient_iters * 2
            if t >= max_queries:
                break
            if not nes:
                # Updating the prior:
                # Create noise for exporation, estimate the gradient, and take a PGD step
                exp_noise = exploration * torch.randn_like(prior) / (dim ** 0.5)
                exp_noise = exp_noise.to(device)
                # Query deltas for finite difference estimator
                q1 = upsampler(prior + exp_noise)
                q2 = upsampler(prior - exp_noise)
                # Loss points for finite difference estimator
                l1 = criterion(model_to_fool(image + fd_eta * q1 / norm(q1)), true_label)  # L(prior + c*noise)
                l2 = criterion(model_to_fool(image + fd_eta * q2 / norm(q2)), true_label)  # L(prior - c*noise)
                # Finite differences estimate of directional derivative
                est_deriv = (l1 - l2) / (fd_eta * exploration)
                # 2-query gradient estimate
                est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise
                # Update the prior with the estimated gradient

                prior = prior_step(prior, est_grad, online_lr)

            else:
                prior = torch.zeros_like(image)
                for _ in range(gradient_iters):
                    exp_noise = torch.randn_like(image) / (dim ** 0.5)
                    exp_noise = exp_noise.to(device)
                    est_deriv = (criterion(model_to_fool(image + fd_eta * exp_noise), true_label) -
                                 criterion(model_to_fool(image - fd_eta * exp_noise), true_label)) / fd_eta
                    prior += est_deriv.view(-1, 1, 1, 1) * exp_noise

                # Preserve images that are already done,
                # Unless we are specifically measuring gradient estimation
                prior = prior * not_dones_mask.view(-1, 1, 1, 1)

            # Update the image:
            # take a pgd step using the prior
            new_im = image_step(image, upsampler(prior * correct_classified_mask.to(device).view(-1, 1, 1, 1)),
                                image_lr)
            image = proj_step(new_im)

            image = torch.clamp(image, 0, 1)

            # Continue query count
            total_queries += 2 * gradient_iters * not_dones_mask
            not_dones_mask = not_dones_mask * ((model_to_fool(image).argmax(1) == true_label).to(device).float())

            # Logging stuff
            if loss == "xent":
                new_losses = criterion(model_to_fool(image), true_label).to(device)
            elif loss == "cw":
                output = model_to_fool(image)
                output = torch.nn.functional.softmax(output)
                y_onehot = to_one_hot(true_label, num_classes=image.shape[1])
                real = (y_onehot.float().to(device) * output.float().to(device))
                real = real.sum(dim=1)
                other = ((1.0 - y_onehot.float().to(device)) * output.float().to(device) - (
                        y_onehot.float().to(device) * 10000000)
                         ).max(1)[0]

                new_losses = torch.log(real + 1e-10) - torch.log(other + 1e-10)
            success_mask = (1 - not_dones_mask) * correct_classified_mask
            num_success = success_mask.to(device).sum()
            current_success_rate = (num_success / correct_classified_mask.sum()).to(device).item()
            success_queries = ((success_mask * total_queries).sum() / num_success).to(device).item()
            not_done_loss = ((new_losses * not_dones_mask).to(device).sum() / not_dones_mask.sum()).item()
            max_curr_queries = total_queries.max().to(device).item()
            if log_progress:
                print("Queries: %d | Success rate: %f | Average queries: %f" % (
                    max_curr_queries, current_success_rate, success_queries))

            if current_success_rate == 1.0:
                break
        if batch_size == 1:
            return {"image_adv": image.cpu().numpy(),
                    "prediction": model_to_fool(image).argmax(1),
                    "elapsed_budget": total_queries.cpu().numpy()[0],
                    "success": success_mask.cpu().numpy()[0] == True}
        return {
            'average_queries': success_queries,
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
            'success_rate': current_success_rate,
            'images_orig': orig_images.cpu().numpy(),
            'images_adv': image.cpu().numpy(),
            'all_queries': total_queries.cpu().numpy(),
            'correctly_classified': correct_classified_mask.cpu().numpy(),
            'success': list(success_mask.cpu().numpy()),
            "elapsed_budget": list(total_queries.cpu().numpy()),

        }


class LocalSearchHelper(object):
    """A helper for local search algorithm.
    Note that since heapq library only supports min heap, we flip the sign of loss function.
    """

    def __init__(self, classifier, epsilon, max_iters, targeted, loss_func):
        """Initalize local search helper.
        Args:
          model: TensorFlow model
          loss_func: str, the type of loss function
          epsilon: float, the maximum perturbation of pixel value
        """
        # Hyperparameter setting
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.targeted = targeted
        self.loss_func = loss_func

        # Network setting

        self.classifier = classifier

        # probs = tf.nn.softmax(self.logits)
        # batch_num = tf.range(0, limit=tf.shape(probs)[0])
        # indices = tf.stack([batch_num, self.y_input], axis=1)
        # ground_truth_probs = tf.gather_nd(params=probs, indices=indices)
        # top_2 = tf.nn.top_k(probs, k=2)
        # max_indices = tf.where(tf.equal(top_2.indices[:, 0], self.y_input), top_2.indices[:, 1], top_2.indices[:, 0])
        # max_indices = tf.stack([batch_num, max_indices], axis=1)
        # max_probs = tf.gather_nd(params=probs, indices=max_indices)

    def _losses(self, classifier, inputs, label):
        with torch.no_grad():
            inputs = torch.tensor(inputs).cuda()
            logits = classifier(inputs)
            _, preds = torch.max(logits.data, 1)
        if self.targeted:
            if self.loss_func == 'xent':
                losses = torch.nn.functional.cross_entropy(
                    logits.cpu(), torch.tensor(label, dtype=torch.long), reduction='none')
            elif self.loss_func == 'cw':
                y_onehot = to_one_hot(torch.tensor(label, dtype=torch.long), num_classes=logits.shape[1])
                output = torch.nn.functional.softmax(logits.cpu())
                real = (y_onehot.float().cpu() * output.float().cpu())
                real = real.sum(dim=1)
                other = ((1.0 - y_onehot.float().cpu()) * output.float().cpu() - (y_onehot.float().cpu() * 10000000)
                         ).max(1)[0]
                losses = -torch.log(real + 1e-10) + torch.log(other + 1e-10)
            else:
                print('Loss function must be xent or cw')
                sys.exit()
        else:
            if self.loss_func == 'xent':
                losses = -torch.nn.functional.cross_entropy(logits.cpu(),
                                                            torch.tensor(label, dtype=torch.long), reduction='none')
            elif self.loss_func == 'cw':
                y_onehot = to_one_hot(torch.tensor(label, dtype=torch.long), num_classes=logits.shape[1])
                output = torch.nn.functional.softmax(logits.cpu())
                real = (y_onehot.float().cpu() * output.float().cpu())
                real = real.sum(dim=1)
                other = ((1.0 - y_onehot.float().cpu()) * output.float().cpu() - (y_onehot.float().cpu() * 10000000)
                         ).max(1)[0]
                losses = torch.log(real + 1e-10) - torch.log(other + 1e-10)
            else:
                print('Loss function must be xent or cw')
                sys.exit()
        return losses.cpu(), preds.cpu()

    def _perturb_image(self, image, noise):
        """Given an image and a noise, generate a perturbed image.
        First, resize the noise with the size of the image.
        Then, add the resized noise to the image.
        Args:
          image: numpy array of size [1, 299, 299, 3], an original image
          noise: numpy array of size [1, 256, 256, 3], a noise
        Returns:
          adv_iamge: numpy array with size [1, 299, 299, 3], a perturbed image
        """
        upsampler = Upsample(size=(self.width, self.height))
        noise = upsampler(torch.tensor(noise))
        adv_image = np.clip(image + noise, 0., 1.)
        return adv_image

    def _flip_noise(self, noise, block):
        """Flip the sign of perturbation on a block.
        Args:
          noise: numpy array of size [1, 256, 256, 3], a noise
          block: [upper_left, lower_right, channel], a block
        Returns:
          noise: numpy array of size [1, 256, 256, 3], an updated noise
        """
        noise_new = np.copy(noise)
        upper_left, lower_right, channel = block
        noise_new[0, channel, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]] *= -1
        return noise_new

    def perturb(self, image, noise, label, blocks):
        """Update a noise with local search algorithm.
        Args:
          image: numpy array of size [1, 299, 299, 3], an original image
          noise: numpy array of size [1, 256, 256, 3], a noise
          label: numpy array of size [1], the label of image (or target label)
          blocks: list, a set of blocks
        Returns:
          noise: numpy array of size [1, 256, 256, 3], an updated noise
          num_queries: int, the number of queries
          curr_loss: float, the value of loss function
          success: bool, True if attack is successful
        """
        # Class variables
        self.width = image.shape[2]
        self.height = image.shape[3]

        w_n = int(2 ** (np.floor(np.log(self.width) / np.log(2))))
        h_n = int(2 ** (np.floor(np.log(self.height) / np.log(2))))
        # Local variables
        priority_queue = []
        num_queries = 0

        # Check if a block is in the working set or not
        A = np.zeros([len(blocks)], np.int32)
        for i, block in enumerate(blocks):
            upper_left, _, channel = block
            x = upper_left[0]
            y = upper_left[1]
            # If the sign of perturbation on the block is positive,
            # which means the block is in the working set, then set A to 1
            if noise[0, channel, x, y] > 0:
                A[i] = 1

        # Calculate the current loss
        image_batch = self._perturb_image(image, noise)
        label_batch = np.copy(label)
        losses, preds = self._losses(self.classifier, image_batch, label_batch)
        num_queries += 1
        curr_loss = losses[0]

        # Early stopping
        if self.targeted:
            if preds == label:
                return noise, num_queries, curr_loss, True
        else:
            if preds != label:
                return noise, num_queries, curr_loss, True

        # Main loop
        for _ in range(self.max_iters):
            # Lazy greedy insert
            indices, = np.where(A == 0)

            batch_size = 100
            num_batches = int(math.ceil(len(indices) / batch_size))

            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, len(indices))

                image_batch = np.zeros([bend - bstart, 3, self.width, self.height], np.float32)
                noise_batch = np.zeros([bend - bstart, 3, w_n, h_n], np.float32)
                label_batch = np.tile(label, bend - bstart)

                for i, idx in enumerate(indices[bstart:bend]):
                    noise_batch[i:i + 1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i + 1, ...] = self._perturb_image(image, noise_batch[i:i + 1, ...])

                losses, preds = self._losses(self.classifier, image_batch, label_batch)

                # Early stopping
                success_indices, = np.where(preds == label) if self.targeted else np.where(preds != label)
                if len(success_indices) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = losses[success_indices[0]]
                    num_queries += success_indices[0] + 1
                    return noise, num_queries, curr_loss, True
                num_queries += bend - bstart

                # Push into the priority queue
                for i in range(bend - bstart):
                    idx = indices[bstart + i]
                    margin = losses[i] - curr_loss
                    heapq.heappush(priority_queue, (margin, idx))

            # Pick the best element and insert it into the working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                curr_loss += best_margin
                noise = self._flip_noise(noise, blocks[best_idx])
                A[best_idx] = 1

            # Add elements into the working set
            while len(priority_queue) > 0:
                # Pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)

                # Re-evalulate the element
                image_batch = self._perturb_image(
                    image, self._flip_noise(noise, blocks[cand_idx]))
                label_batch = np.copy(label)
                losses, preds = self._losses(self.classifier, image_batch, label_batch)

                num_queries += 1
                margin = losses[0] - curr_loss

                # If the cardinality has not changed, add the element
                if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin > 0:
                        break
                    # Update the noise
                    curr_loss = losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 1
                    # Early stopping
                    if self.targeted:
                        if preds == label:
                            return noise, num_queries, curr_loss, True
                    else:
                        if preds != label:
                            return noise, num_queries, curr_loss, True
                # If the cardinality has changed, push the element into the priority queue
                else:
                    heapq.heappush(priority_queue, (margin, cand_idx))

            priority_queue = []

            # Lazy greedy delete
            indices, = np.where(A == 1)

            batch_size = 100
            num_batches = int(math.ceil(len(indices) / batch_size))

            for ibatch in range(num_batches):
                bstart = ibatch * batch_size
                bend = min(bstart + batch_size, len(indices))

                image_batch = np.zeros([bend - bstart, 3, self.width, self.height], np.float32)
                noise_batch = np.zeros([bend - bstart, 3, w_n, h_n], np.float32)
                label_batch = np.tile(label, bend - bstart)

                for i, idx in enumerate(indices[bstart:bend]):
                    noise_batch[i:i + 1, ...] = self._flip_noise(noise, blocks[idx])
                    image_batch[i:i + 1, ...] = self._perturb_image(image, noise_batch[i:i + 1, ...])

                losses, preds = self._losses(self.classifier, image_batch, label_batch)

                # Early stopping
                success_indices, = np.where(preds == label) if self.targeted else np.where(preds != label)
                if len(success_indices) > 0:
                    noise[0, ...] = noise_batch[success_indices[0], ...]
                    curr_loss = losses[success_indices[0]]
                    num_queries += success_indices[0] + 1
                    return noise, num_queries, curr_loss, True
                num_queries += bend - bstart

                # Push into the priority queue
                for i in range(bend - bstart):
                    idx = indices[bstart + i]
                    margin = losses[i] - curr_loss
                    heapq.heappush(priority_queue, (margin, idx))

            # Pick the best element and remove it from the working set
            if len(priority_queue) > 0:
                best_margin, best_idx = heapq.heappop(priority_queue)
                curr_loss += best_margin
                noise = self._flip_noise(noise, blocks[best_idx])
                A[best_idx] = 0

            # Delete elements into the working set
            while len(priority_queue) > 0:
                # pick the best element
                cand_margin, cand_idx = heapq.heappop(priority_queue)

                # Re-evalulate the element
                image_batch = self._perturb_image(
                    image, self._flip_noise(noise, blocks[cand_idx]))
                label_batch = np.copy(label)

                losses, preds = self._losses(self.classifier, image_batch, label_batch)

                num_queries += 1
                margin = losses[0] - curr_loss

                # If the cardinality has not changed, remove the element
                if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
                    # If there is no element that has negative margin, then break
                    if margin >= 0:
                        break
                    # Update the noise
                    curr_loss = losses[0]
                    noise = self._flip_noise(noise, blocks[cand_idx])
                    A[cand_idx] = 0
                    # Early stopping
                    if self.targeted:
                        if preds == label:
                            return noise, num_queries, curr_loss, True
                    else:
                        if preds != label:
                            return noise, num_queries, curr_loss, True
                # If the cardinality has changed, push the element into the priority queue
                else:
                    heapq.heappush(priority_queue, (margin, cand_idx))

            priority_queue = []

        return noise, num_queries, curr_loss, False


class ParsimoniousAttack(object):
    """Parsimonious attack using local search algorithm"""

    def __init__(self, classifier, epsilon, max_queries, loss_func,
                 batch_size, block_size, no_hier, max_iters, targeted):
        """Initialize attack.
        Args:
          model: TensorFlow model
          args: arguments
        """
        # Hyperparameter setting
        self.loss_func = loss_func
        self.max_queries = max_queries
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.block_size = block_size
        self.no_hier = no_hier
        self.max_iters = max_iters
        self.targeted = targeted
        # Create helper
        self.local_search = LocalSearchHelper(classifier, self.epsilon, self.max_iters, self.targeted, self.loss_func)

    def _perturb_image(self, image, noise):
        """Given an image and a noise, generate a perturbed image.
        First, resize the noise with the size of the image.
        Then, add the resized noise to the image.
        Args:
          image: numpy array of size [1, w, h, 3], an original image
          noise: numpy array of size [1, 2**n, 2**n, 3], a noise
        Returns:
          adv_iamge: numpy array of size [1, 299, 299, 3], an perturbed image
        """
        upsampler = Upsample(size=(self.width, self.height))
        noise = upsampler(torch.tensor(noise))

        adv_image = image + noise
        adv_image = np.clip(adv_image, 0., 1.)
        return adv_image

    def _split_block(self, upper_left, lower_right, block_size):
        """Split an image into a set of blocks.
        Note that a block consists of [upper_left, lower_right, channel]
        Args:
          upper_left: [x, y], the coordinate of the upper left of an image
          lower_right: [x, y], the coordinate of the lower right of an image
          block_size: int, the size of a block
        Returns:
          blocks: list, the set of blocks
        """
        blocks = []
        xs = np.arange(upper_left[0], lower_right[0], block_size)
        ys = np.arange(upper_left[1], lower_right[1], block_size)
        for x, y in itertools.product(xs, ys):
            for c in range(3):
                blocks.append([[x, y], [x + block_size, y + block_size], c])
        return blocks

    def perturb(self, image, label, index=0):
        """Perturb an image.
        Args:
          image: numpy array of size [1, 299, 299, 3], an original image
          label: numpy array of size [1], the label of the image (or target label)
          index: int, the index of the image
        Returns:
          adv_image: numpy array of size [1, 299, 299, 3], an adversarial image
          num_queries: int, the number of queries
          success: bool, True if attack is successful
        """
        # Set random seed by index for the reproducibility
        np.random.seed(index)

        # Class variables
        self.width = image.shape[2]
        self.height = image.shape[3]

        w_n = int(2 ** (np.floor(np.log(self.width) / np.log(2))))
        h_n = int(2 ** (np.floor(np.log(self.height) / np.log(2))))

        # Local variables
        adv_image = np.copy(image)
        num_queries = 0
        block_size = self.block_size
        upper_left = [0, 0]
        lower_right = [w_n, h_n]

        # Split an image into a set of blocks
        blocks = self._split_block(upper_left, lower_right, block_size)

        # Initialize a noise to -epsilon
        noise = -self.epsilon * np.ones([1, 3, w_n, h_n], dtype=np.float32)

        # Construct a batch
        num_blocks = len(blocks)
        batch_size = self.batch_size if self.batch_size > 0 else num_blocks
        curr_order = np.random.permutation(num_blocks)

        # Main loop
        while True:
            # Run batch
            num_batches = int(math.ceil(num_blocks / batch_size))
            for i in range(num_batches):
                # Pick a mini-batch
                bstart = i * batch_size
                bend = min(bstart + batch_size, num_blocks)
                blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
                # Run local search algorithm on the mini-batch
                noise, queries, loss, success = self.local_search.perturb(
                    image, noise, label, blocks_batch)
                num_queries += queries
                print("Block size: {}, batch: {}, loss: {:.4f}, num queries: {}".format(
                    block_size, i, loss, num_queries))
                # If query count exceeds the maximum queries, then return False
                if num_queries > self.max_queries:
                    res = {"image_adv": adv_image,
                           "elapsed_budget": [num_queries],
                           "success": [False]}
                    return res
                # Generate an adversarial image
                adv_image = self._perturb_image(image, noise)
                # If attack succeeds, return True
                if success:
                    res = {"image_adv": adv_image,
                           "elapsed_budget": [num_queries],
                           "success": [True]}
                    return res
            # If block size >= 2, then split the iamge into smaller blocks and reconstruct a batch
            if not self.no_hier and block_size >= 2:
                block_size = int(block_size / 2)
                blocks = self._split_block(upper_left, lower_right, block_size)
                num_blocks = len(blocks)
                batch_size = self.batch_size if self.batch_size > 0 else num_blocks
                curr_order = np.random.permutation(num_blocks)

            # Otherwise, shuffle the order of the batch
            else:
                curr_order = np.random.permutation(num_blocks)
