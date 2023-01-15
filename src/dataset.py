import os
from typing import Callable, List, Optional

import torch
import pandas as pd

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

from networkx import Graph
from gensim.models import KeyedVectors
from torch_geometric.utils.convert import from_networkx
import networkx as nx
from sklearn.model_selection import train_test_split

class EmailEUCore(InMemoryDataset):
    r"""A modified e-mail communication network of a large European research
    institution, taken from the `"Local Higher-order Graph Clustering"
    <https://www-cs.stanford.edu/~jure/pubs/mappr-kdd17.pdf>`_ paper.
    Nodes indicate members of the institution.
    An edge between a pair of members indicates that they exchanged at least
    one email.
    Node labels indicate membership to one of the 37 departments.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(
        self,
        root: str,
        embbeddings_path: str,
        data_path: str,
        graph_path: str,
        split: str = "predefined",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        random_state: int = 345,
        num_train_per_class: int = 5
    ):
        self._embbeddings_path = embbeddings_path
        self._graph_path = graph_path
        self._data_path = data_path
        self._random_state = random_state
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.split = split
        
        assert self.split in ['predefined', 'full']

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])
   
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    @property
    def processed_dir(self) -> str:
        return self.root
    
    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        y = self.data.y
        if y is None:
            return 0
        elif y.numel() == y.size(0):
            return torch.unique(y).numel()
        else:
            return self.data.y.size(-1)
    
    def process(self):
        
        labels = pd.read_csv(self._data_path, sep = " ")
        vectors = KeyedVectors.load_word2vec_format(
            self._embbeddings_path,
            binary=False
        )
        
        G = nx.read_gpickle(self._graph_path)
        attribs = list(G.nodes[0].keys())
        
        
        if "department_id" not in attribs:
            raise ValueError("department_id or user_id not found in node attributes")
        
        if "user_id" not in list(labels.columns):
            raise ValueError("user_id not found in train_data")
            
        train_ids, test_ids = train_test_split(
            list(labels["user_id"]),
            test_size=0.5,
            stratify=list(labels["department_id"]), 
            random_state=self._random_state
        )
        
        test_ids, valid_ids = train_test_split(
            test_ids,
            test_size=0.5,
            stratify=[labels[labels["user_id"] == y]["department_id"].iloc[0] for y in test_ids], 
            random_state=self._random_state
        )
        
        
        nx.set_node_attributes(G, False, "train_mask")
        nx.set_node_attributes(G, False, "test_mask")
        nx.set_node_attributes(G, False, "val_mask")
        
        
        for v, data in G.nodes(data=True):
            embbeding_id = labels[labels["user_id"] == v]["embbedding_id"].iloc[0]
            data["y"] = data.pop("department_id")
            data["x"] = vectors[embbeding_id]
            
            if v in train_ids:
                data["train_mask"] = True
            elif v in valid_ids:
                data["val_mask"] = True
            else:
                data["test_mask"] = True
            

        data = from_networkx(G)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])