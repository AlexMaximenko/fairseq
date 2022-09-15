from torch.utils.data import WeightedRandomSampler
import torch
from copy import copy
from torch.utils.data import SubsetRandomSampler, DistributedSampler


def get_group_batch_sizes(batch_size, weightes):
    """Splits batch into parts according to provided weights."""
    sub_batches = []
    sum_weightes = sum(weightes)
    for weight in weightes:
        sub_batches.append(round(batch_size * weight / sum_weightes))
    sub_batches[-1] = batch_size - sum(sub_batches[:-1])
    return sub_batches


def sample_groups_batch(group_sizes, group_iters):
    """Samples from each group_iter group_size elements.

    If iter for some group is exhausted then Exception is suppressed
    and sampling goes to the next group.
    """
    batch = []
    for size, group_iter in zip(group_sizes, group_iters):
        for _ in range(size):
            try:
                batch.append(next(group_iter))
            except StopIteration:
                break
    return batch


class WeightedGroupsBatchSampler(torch.utils.data.BatchSampler):
    """Each batch has reserved slots for samples from each group.

    These slots are distributed according to specified weights. Samples are drawn until one of the groups exhausts.
    """

    def __init__(self, groups, weights=None, sub_batches=None, shuffle=True, generator_seed=1, *args, **kwargs):
        super(WeightedGroupsBatchSampler, self).__init__(sampler=None, *args, **kwargs)
        self.weights = weights
        self.groups = groups
        self.shuffle = shuffle
        self.seed = generator_seed

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if weights is not None:
            assert sub_batches is None, "you can only specify weigths or subbatches"
            assert len(groups) == len(weights), "groups num: {}, weights num: {}".format(len(groups), len(weights))
            self.sub_batches = get_group_batch_sizes(self.batch_size, self.weights)
        else:
            self.sub_batches = sub_batches
            assert len(groups) == len(sub_batches)
            assert sum(self.sub_batches) == self.batch_size
        self.group_samplers = [
            SubsetRandomSampler(group, generator=generator) if self.shuffle else group
            for group in groups
        ]

    def __call__(self, dataset, epoch):
        self.update(epoch)
        return self

    def update(self, epoch):
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)

        self.group_samplers = [
            SubsetRandomSampler(group, generator=generator) if self.shuffle else group
            for group in self.groups
        ]

    def __iter__(self):
        group_iters = [iter(sub_sampler) for sub_sampler in self.group_samplers]

        # Iterate over batches
        while True:
            batch = sample_groups_batch(self.sub_batches, group_iters)
            if len(batch) == 0:
                break
            elif len(batch) < self.batch_size:
                if not self.drop_last:
                    yield batch
                break
            else:
                yield batch

    def __len__(self):
        if self.drop_last:
            return min(
                len(group_sampler) // group_size
                for group_sampler, group_size in zip(
                    self.group_samplers, self.sub_batches
                )
            )
        else:
            return min(
                (len(group_sampler) + group_size - 1) // group_size
                for group_sampler, group_size in zip(
                    self.group_samplers, self.sub_batches
                )
            )

    def get_last_batch_underflow(self):
        if self.drop_last:
            return 0
        else:
            num_iters = len(self)
            last_underflow = [
                max(0, sub_batch * num_iters - len(sub_sampler))
                for sub_sampler, sub_batch in zip(self.group_samplers, self.sub_batches)
            ]
            return sum(last_underflow)