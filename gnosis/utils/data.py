import hydra
from torch.utils.data import random_split
import torchvision
import torch
import math
from upcycle import cuda
from gnosis.distillation.classification import reduce_ensemble_logits
import copy
from torch.utils.data import TensorDataset, DataLoader
import random
import os

from torchvision.datasets.folder import ImageFolder


def get_loaders(config):
    train_transform, test_transform = get_augmentation(config)
    if config.dataset.name == 'tiny_imagenet':
        train_dataset = ImageFolder(root=os.path.join(config.dataset.root_dir, 'train'), transform=train_transform)
        test_dataset = ImageFolder(root=os.path.join(config.dataset.root_dir, 'val'), transform=test_transform)
    else:
        train_dataset = hydra.utils.instantiate(config.dataset.init, train=True, transform=train_transform)
        test_dataset = hydra.utils.instantiate(config.dataset.init, train=False, transform=test_transform)

    if config.dataset.shuffle_train_targets.enabled:
        random.seed(config.dataset.shuffle_train_targets.seed)
        num_shuffled = int(len(train_dataset) * config.dataset.shuffle_train_targets.ratio)
        shuffle_start = random.randint(0, len(train_dataset) - num_shuffled)
        target_copy = train_dataset.targets[shuffle_start:shuffle_start + num_shuffled]
        random.seed(config.dataset.shuffle_train_targets.seed)  # for backwards-compatibility
        random.shuffle(target_copy)
        train_dataset.targets[shuffle_start:shuffle_start + num_shuffled] = target_copy

    subsample_ratio = config.dataset.subsample.ratio
    if subsample_ratio < 1.0:
        train_splits = split_dataset(train_dataset, subsample_ratio,
                                     config.dataset.subsample.seed)
        train_dataset = train_splits[config.dataset.subsample.split]
    else:
        train_splits = [train_dataset]
    if config.trainer.eval_dataset == 'val':
        train_dataset, test_dataset = split_dataset(train_dataset, 0.8)

    train_loader = hydra.utils.instantiate(config.dataloader, dataset=train_dataset)
    test_loader = hydra.utils.instantiate(config.dataloader, dataset=test_dataset)

    return train_loader, test_loader, train_splits


def split_dataset(dataset, ratio, seed=None):
    num_total = len(dataset)
    num_split = int(num_total * ratio)
    gen = torch.Generator() if seed is None else torch.Generator().manual_seed(seed)
    return random_split(dataset, [num_split, num_total - num_split], gen)


def get_augmentation(config):
    assert 'augmentation' in config.keys()
    if config.augmentation.transforms_list is not None:
        transforms_list = [hydra.utils.instantiate(config.augmentation[name])
                           for name in config.augmentation["transforms_list"]]
        if 'random_apply' in config.augmentation.keys() and config.augmentation.random_apply.p < 1:
            transforms_list = [
                hydra.utils.instantiate(config.augmentation.random_apply, transforms=transforms_list)]
    else:
        transforms_list = []

    normalize_transforms = [
        torchvision.transforms.ToTensor(),
    ]
    if config.augmentation.normalization == 'zscore':
        # mean subtract and scale to unit variance
        normalize_transforms.append(
            torchvision.transforms.Normalize(config.dataset.statistics.mean_statistics,
                                             config.dataset.statistics.std_statistics)
        )
    elif config.augmentation.normalization == 'unitcube':
        # rescale values to [-1, 1]
        min_vals = config.dataset.statistics.min
        max_vals = config.dataset.statistics.max
        offset = [0.5 * (min_val + max_val) for min_val, max_val in zip(min_vals, max_vals)]
        scale = [(max_val - min_val) / 2 for max_val, min_val in zip(max_vals, min_vals)]
        normalize_transforms.append(
            torchvision.transforms.Normalize(offset, scale)
        )

    train_transform = torchvision.transforms.Compose(transforms_list + normalize_transforms)
    test_transform = torchvision.transforms.Compose(normalize_transforms)
    return train_transform, test_transform


def make_real_teacher_data(train_dataset, teacher, batch_size):
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    inputs, targets, teacher_logits = [], [], []
    teacher.eval()
    for input_batch, target_batch in train_loader:
        input_batch = cuda.try_cuda(input_batch)
        with torch.no_grad():
            batch_logits = teacher(input_batch)  # [batch_size, num_teachers, ... ]
            batch_logits = reduce_ensemble_logits(batch_logits)
        inputs.append(input_batch.cpu())
        targets.append(target_batch.cpu())
        teacher_logits.append(batch_logits.cpu())

    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)
    teacher_logits = torch.cat(teacher_logits, dim=0)
    return inputs, targets, teacher_logits


def make_synth_teacher_data(generator, teacher, dataset_size, batch_size):
    if dataset_size == 0:
        return None
    num_rounds = math.ceil(dataset_size / batch_size)
    synth_inputs, teacher_labels, teacher_logits = [], [], []
    teacher.eval()
    generator.eval()
    for _ in range(num_rounds):
        with torch.no_grad():
            input_batch = generator.sample(batch_size)
            logit_batch = teacher(input_batch)
            logit_batch = reduce_ensemble_logits(logit_batch)
        label_batch = logit_batch.argmax(dim=-1)

        synth_inputs.append(input_batch.cpu())
        teacher_logits.append(logit_batch.cpu())
        teacher_labels.append(label_batch.cpu())

    synth_inputs = torch.cat(synth_inputs, dim=0)[:dataset_size]
    synth_targets = torch.cat(teacher_labels, dim=0)[:dataset_size]
    synth_logits = torch.cat(teacher_logits, dim=0)[:dataset_size]
    return synth_inputs, synth_targets, synth_logits


def get_distill_loaders(config, train_loader, synth_data):
    num_real = len(train_loader.dataset)
    num_synth = 0 if synth_data is None else synth_data[0].size(0)
    real_ratio = num_real / (num_real + num_synth)
    real_batch_size = math.ceil(real_ratio * config.dataloader.batch_size)
    synth_batch_size = config.dataloader.batch_size - real_batch_size
    train_loader = DataLoader(train_loader.dataset, shuffle=True, batch_size=real_batch_size)
    if num_synth == 0:
        return train_loader, None
    synth_loader = DataLoader(TensorDataset(*synth_data), shuffle=True, batch_size=synth_batch_size)
    return train_loader, synth_loader


def get_logits(model, data_loader):
    model.eval()
    logits = []
    for minibatch in data_loader:
        input_batch = cuda.try_cuda(minibatch[0])
        with torch.no_grad():
            logit_batch = model(input_batch)
            if logit_batch.dim() == 3:
                logit_batch = reduce_ensemble_logits(logit_batch)
            logits.append(logit_batch.cpu())
    return torch.cat(logits, dim=0)


def save_logits(config, student, teacher, generator, logger):
    print('==== saving logits ====')
    config = copy.deepcopy(config)
    config.augmentation.transforms_list = None  # no data augmentation for evaluation
    config.dataloader.shuffle = False
    _, test_loader, train_splits = get_loaders(config)
    distill_splits = [train_splits[i] for i in config.distill_loader.splits]
    distill_loader = hydra.utils.instantiate(config.distill_loader, teacher=teacher,
                                             datasets=distill_splits, synth_sampler=generator)

    student_train = get_logits(student, distill_loader)
    logger.save_obj(student_train, 'student_train_logits.pkl')
    teacher_train = get_logits(teacher, distill_loader)
    logger.save_obj(teacher_train, 'teacher_train_logits.pkl')
    del student_train, teacher_train, distill_loader

    student_test = get_logits(student, test_loader)
    logger.save_obj(student_test, 'student_test_logits.pkl')
    teacher_test = get_logits(teacher, test_loader)
    logger.save_obj(teacher_test, 'teacher_test_logits.pkl')
    del student_test, teacher_test, test_loader

    # if synth_data is None:
    #     return None
    # synth_loader = DataLoader(TensorDataset(*synth_data), shuffle=False,
    #                           batch_size=config.dataloader.batch_size)
    # student_synth = get_logits(student, synth_loader)
    # logger.save_obj(student_synth, 'student_synth_logits.pkl')
    # teacher_synth = get_logits(teacher, synth_loader)
    # logger.save_obj(teacher_synth, 'teacher_synth_logits.pkl')
