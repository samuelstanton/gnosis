import hydra
from torch.utils.data import random_split
import torchvision
import torch
import math
from upcycle import cuda
from gnosis.distillation.classification import reduce_ensemble_logits
import copy
from torch.utils.data import TensorDataset, DataLoader


def get_loaders(config):
    train_transform, test_transform = get_augmentation(config)
    train_dataset = hydra.utils.instantiate(config.dataset.init, train=True, transform=train_transform)
    test_dataset = hydra.utils.instantiate(config.dataset.init, train=False, transform=test_transform)

    subsample_ratio = config.dataset.subsample_ratio
    if subsample_ratio > 0.:
        train_dataset, _ = split_dataset(train_dataset, subsample_ratio)
        test_dataset, _ = split_dataset(test_dataset, subsample_ratio)
    if config.trainer.eval_dataset == 'val':
        train_dataset, test_dataset = split_dataset(train_dataset, 0.8)

    train_loader = hydra.utils.instantiate(config.dataloader, dataset=train_dataset)
    test_loader = hydra.utils.instantiate(config.dataloader, dataset=test_dataset)

    return train_loader, test_loader


def split_dataset(dataset, ratio):
    num_total = len(dataset)
    num_split = int(num_total * ratio)
    return random_split(dataset, [num_split, num_total - num_split])


def get_augmentation(config):
    assert 'augmentation' in config.keys()
    if config.augmentation.transforms_list is not None:
        transforms_list = [hydra.utils.instantiate(config.augmentation[name])
                           for name in config.augmentation["transforms_list"].split(",")]
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


def get_distill_loader(config, teacher, train_dataset, synth_data):
    real_data = make_real_teacher_data(train_dataset, teacher, batch_size=config.dataloader.batch_size)
    if synth_data is not None:
        full_data = [torch.cat([r, s], dim=0) for r, s in zip(real_data, synth_data)]
        full_dataset = TensorDataset(*full_data)
    else:
        full_dataset = TensorDataset(*real_data)
    dataloader = DataLoader(full_dataset, batch_size=config.dataloader.batch_size,
                                             shuffle=True)
    return dataloader


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


def save_logits(config, student, teacher, synth_data, logger):
    print('==== saving logits ====')
    config = copy.deepcopy(config)
    config.augmentation.transforms_list = None  # no data augmentation for evaluation
    config.dataloader.shuffle = False
    train_loader, test_loader = get_loaders(config)

    student_train = get_logits(student, train_loader)
    logger.save_obj(student_train, 'student_train_logits.pkl')
    teacher_train = get_logits(teacher, train_loader)
    logger.save_obj(teacher_train, 'teacher_train_logits.pkl')
    del student_train, teacher_train, train_loader

    student_test = get_logits(student, test_loader)
    logger.save_obj(student_test, 'student_test_logits.pkl')
    teacher_test = get_logits(teacher, test_loader)
    logger.save_obj(teacher_test, 'teacher_test_logits.pkl')
    del student_test, teacher_test, test_loader

    synth_loader = DataLoader(TensorDataset(*synth_data), shuffle=False,
                              batch_size=config.dataloader.batch_size)
    student_synth = get_logits(student, synth_loader)
    logger.save_obj(student_synth, 'student_synth_logits.pkl')
    teacher_synth = get_logits(teacher, synth_loader)
    logger.save_obj(teacher_synth, 'teacher_synth_logits.pkl')
