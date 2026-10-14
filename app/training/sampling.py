from torch.utils.data import BatchSampler


class CompleteBatchSampler(BatchSampler):
    """Merge a trailing singleton into the previous batch rather than omit an image."""

    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is not None:
                if len(batch) == 1:
                    yield pending + batch
                    pending = None
                    continue
                yield pending
            pending = batch
        if pending is not None:
            yield pending

    def __len__(self):
        n = len(self.sampler)
        return (n + self.batch_size - 1) // self.batch_size - int(
            n % self.batch_size == 1 and n > 1
        )
