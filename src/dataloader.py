import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataListLoader
from pathlib import PosixPath

from .dataset import EmailEUCore


class GraphData(pl.LightningDataModule):

    def __init__(
        self,
        graph_path: PosixPath,
        dataset_path: PosixPath,
        embbeddings_path: PosixPath,
        root_path: str = "dataset",
        split: str = 'predefined',
        random_state: int = 345
        
    ):
        super().__init__()

        self._dataset = self._load(
            dataset_path=dataset_path,
            embbeddings_path=embbeddings_path,
            graph_path=graph_path,
            split=split,
            root_path=root_path,
            random_state=random_state
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader()

    def val_dataloader(self) -> DataLoader:
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        return self._dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader()

    @property
    def num_node_features(self) -> int:
        return self._dataset.num_node_features

    @property
    def num_classes(self) -> int:
        return self._dataset.num_classes

    @property
    def data(self) -> Data:
        return self._dataset[0]

    @staticmethod
    def _load(
        dataset_path: PosixPath,
        embbeddings_path: PosixPath,
        graph_path: PosixPath,
        split: str,
        root_path: str,
        random_state: int
    ) -> Dataset:
        
        return EmailEUCore(
            root_path,
            graph_path=graph_path,
            data_path=dataset_path,
            embbeddings_path=embbeddings_path,
            split=split,
            random_state=random_state
        )

    def _dataloader(self) -> DataLoader:
        # We can use the same DataLoader for all data splits, as there are
        # masks in the Data object that we will use for selecting the
        # appropriate nodes set. Moreover, we can set `shuffle=False` for all
        # splits, because we have only one `Data` object (there is nothing
        # to shuffle). Notice that we use PyTorch-Geometric's custom data loader
        # object, because the default PyTorch one does not know how to collate
        # `Data` objects in a batch.
        return DataListLoader(
            dataset=self._dataset,
            batch_size=1,
            shuffle=False,
            num_workers=12
        )
