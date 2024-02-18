# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
import torch as th
from torch.utils.data import Dataset


class CsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sep: str,
        header: Optional[int] = 0,
        encoding: Optional[str] = "utf-8",
        columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        dtypes: Optional[Dict[str, Type]] = None,
        shuffle: Optional[bool] = True,
    ):
        super().__init__()

        self.__df = pd.read_csv(
            csv_path, sep=sep, header=header, dtype=dtypes, encoding=encoding
        )

        if shuffle:
            self.__df = self.__df.iloc[th.randperm(len(self.__df))]

        assert target_column in self.__df.columns
        self.__target_columns = target_column

        self.__class_to_idx = {
            c: i
            for i, c in enumerate(self.__df[self.__target_columns].unique())
        }

        if columns is None:
            self.__columns = list(
                set(self.__df.columns) - {self.__target_columns}
            )
            assert len(self.__columns) > 0
        else:
            assert all(c in self.__df.columns for c in columns)
            self.__columns = columns

    def __len__(self) -> int:
        return len(self.__df)

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:
        x = th.tensor(self.__df[self.__columns].iloc[index, :].tolist())
        y = th.tensor(
            self.__class_to_idx[self.__df[self.__target_columns].iloc[index]]
        )

        return x, y
