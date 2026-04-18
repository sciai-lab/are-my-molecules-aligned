import copy
import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, random_split
from torch_geometric.datasets.qm9 import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import radius_graph
from torchvision.transforms import transforms
from lightning.pytorch.utilities import CombinedLoader


class QM9DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        selected_features: list[int] = [0, 7],  # Dipole moment and internal energy at T=0K
        batch_size: int = 64,
        radial_cutoff: bool = True,
        radial_cutoff_radius: float = 5.0,
        self_loops: bool = False,
        num_workers: int = 0,
        exclude_edge_attr: bool = False,
        use_right_split: bool = True,
        custom_transforms: Optional[torch.nn.Module] = None,
        use_custom_transforms_in_split: list[str] = ("train", "val", "test"),
        use_additional_untransformed_val_loader: bool = False,
        use_additional_untransformed_test_loader: bool = False,
        seed_for_split: Optional[int] = None,
        use_first_pc_as_target: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.transform_sample = _transform_sample(
            selected_features, radial_cutoff, radial_cutoff_radius, self_loops, exclude_edge_attr, use_first_pc_as_target=use_first_pc_as_target
        )

        if custom_transforms is not None:
            self.transform_sample_with_custom = transforms.Compose([self.transform_sample, custom_transforms])
            self.use_custom_transforms_in_split = use_custom_transforms_in_split
        else:
            self.use_custom_transforms_in_split = ()

        self.batch_size_per_device = batch_size

        self.radial_cutoff = radial_cutoff
        self.radial_cutoff_radius = radial_cutoff_radius
        self.self_loops = self_loops
        self.exclude_edge_attr = exclude_edge_attr
        self.use_right_split = use_right_split
        self.seed_for_split = seed_for_split
        self.use_additional_untransformed_val_loader = use_additional_untransformed_val_loader
        self.use_additional_untransformed_test_loader = use_additional_untransformed_test_loader

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = QM9(root=self.hparams.data_dir, transform=self.transform_sample)

            if self.seed_for_split is not None:
                print(f"Using seed {self.seed_for_split} for dataset split.")
                generator = torch.Generator().manual_seed(self.seed_for_split)
            else:
                generator = torch.Generator()

            if self.use_right_split:
                dataset_length = len(dataset)

                self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset, lengths=[110000, 10000, dataset_length - 120000],
                    generator=generator,
                )
            else:
                # split the test set into validation and test sets
                self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset, lengths=self.hparams.split,
                    generator=generator,
                )

        for split in self.use_custom_transforms_in_split:
            # apply custom transforms only to the training set

            # first deepcopy the dataset to avoid modifying the original dataset
            dataset = getattr(self, f"data_{split}").dataset
            getattr(self, f"data_{split}").dataset = copy.deepcopy(dataset)

            getattr(self, f"data_{split}").dataset.transform = self.transform_sample_with_custom
            # print(f"Applied custom transforms to {split} set.", self.transform_sample_with_custom)

    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

        if self.use_additional_untransformed_val_loader:
            # get val dataset without custom transforms
            val_data2 = copy.deepcopy(self.data_val)
            val_data2.dataset.transform = self.transform_sample
            val_loader2 = DataLoader(
                dataset=val_data2,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                shuffle=False,
            )

            val_loader = CombinedLoader(
                {"val_with_custom": val_loader, "val_without_custom": val_loader2},
                mode="sequential",
            )

        return val_loader

    def test_dataloader(self):
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
        if self.use_additional_untransformed_test_loader:
            # get test dataset without custom transforms
            test_data2 = copy.deepcopy(self.data_test)
            test_data2.dataset.transform = self.transform_sample
            test_loader2 = DataLoader(
                dataset=test_data2,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                shuffle=False,
            )

            test_loader = CombinedLoader(
                {"test_with_custom": test_loader, "test_without_custom": test_loader2},
                mode="sequential",
            )
        return test_loader

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


class _transform_sample(torch.nn.Module):
    def __init__(
        self,
        selected_features: list[int],
        radial_cutoff: bool = False,
        radial_cutoff_radius: float = 5.0,
        self_loops: bool = False,
        exclude_edge_attr: bool = False,
        use_first_pc_as_target: bool = False,
    ):
        super().__init__()
        self.selected_features = selected_features
        self.radial_cutoff = radial_cutoff
        self.radial_cutoff_radius = radial_cutoff_radius
        self.self_loops = self_loops
        self.exclude_edge_attr = exclude_edge_attr
        self.use_first_pc_as_target = use_first_pc_as_target

    def forward(self, data):
        data.y = data.y[:, self.selected_features]

        if self.radial_cutoff:
            old_edge_index = data.edge_index  # [2, E]
            old_edge_attr = data.edge_attr  # [E, F]
            data.old_edge_attr = old_edge_attr.clone()
            data.old_edge_index = old_edge_index.clone()

            new_edge_index = radius_graph(
                data.pos, r=self.radial_cutoff_radius, batch=data.batch, loop=self.self_loops
            )  # [2, E']
            new_edge_attr = torch.zeros(
                new_edge_index.shape[1], old_edge_attr.shape[1] + 1
            )  # [E', F+1]

            # create mask for the old edges that are also in the new edges
            t_edge_index = new_edge_index.t()  # [E', 2]
            t_old_edge_index = old_edge_index.t()  # [E, 2]
            mask = (t_edge_index[:, None] == t_old_edge_index[None, :]).all(-1).any(-1)  # [E']

            # copy the data from the old edges to the new edges
            new_edge_attr[mask] = torch.cat(
                [old_edge_attr, torch.zeros(old_edge_attr.shape[0], 1)], dim=1
            )

            # add a new edge attribute to the newly created edges
            one_vector = torch.zeros(old_edge_attr.shape[1] + 1)
            one_vector[-1] = 1
            new_edge_attr[~mask] = one_vector

            data.edge_index = new_edge_index
            data.edge_attr = new_edge_attr

        # the z can have values of 1, 6, 7, 8, 9
        # now we want to one-hot encode it into data.x
        data.z_original = data.z.clone()
        data.z = data.z - 1
        data.z[data.z != 0] = data.z[data.z != 0] - 4
        data.x = torch.nn.functional.one_hot(data.z.long(), num_classes=5).float()

        if self.exclude_edge_attr:
            data.edge_attr = None

        # calculate the mean of the positions
        data.mean = torch.mean(data.pos, dim=0).unsqueeze(0)

        return data
