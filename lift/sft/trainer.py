from copy import deepcopy
from random import shuffle
from torch.utils.data import Sampler, DataLoader
from transformers import Trainer
from typing import List


class LIFTSFTSampler(Sampler):
    def __init__(self, batch_size: int, batch_ids: List, num_article_epochs: int, num_article_qa_epochs: int):
        super().__init__()
        self.batch_size = batch_size
        self.batch_ids = deepcopy(batch_ids)
        self.num_article_epochs = num_article_epochs
        self.num_article_qa_epochs = num_article_qa_epochs
        self.size = 0
        for article_ids, article_qa_ids in self.batch_ids:
            self.size += (article_ids[1] - article_ids[0] + self.batch_size - 1) // self.batch_size * self.num_article_epochs
            self.size += (article_qa_ids[1] - article_qa_ids[0] + self.batch_size - 1) // self.batch_size * self.num_article_epochs
    
    def __iter__(self):
        shuffle(self.batch_ids)
        batch_ids = []
        for article_ids, article_qa_ids in self.batch_ids:
            article_ids = list(range(article_ids[0], article_ids[1]))
            batch_ids += [article_ids[i:i + self.batch_size] for i in range(0, len(article_ids), self.batch_size)] * self.num_article_epochs
            article_qa_ids = list(range(article_qa_ids[0], article_qa_ids[1]))
            batch_ids += [article_qa_ids[i:i + self.batch_size] for i in range(0, len(article_qa_ids), self.batch_size)] * self.num_article_qa_epochs
        for batch in batch_ids:
            yield batch
    
    def __len__(self):
        return self.size


class LIFTSFTTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        
        from transformers.trainer_utils import seed_worker

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": seed_worker,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "batch_sampler": LIFTSFTSampler(self._train_batch_size, train_dataset.batch_ids, train_dataset.num_article_epochs, train_dataset.num_article_qa_epochs),
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
