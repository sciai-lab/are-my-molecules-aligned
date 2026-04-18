from functools import partial
from typing import Any, Dict, Tuple

import torch
import torch_geometric
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.regression import mae
from torch.nn import Module
from torch_geometric.nn import (
    LayerNorm,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torchvision.ops import MLP
from torch_geometric.nn import MessagePassing

from tensor_frames.nn.embedding.radial import BesselEmbedding
from tensor_frames.nn.embedding.axial import AxisWiseBesselEmbedding
from tensor_frames.reps.tensorreps import TensorReps
from tensor_frames.nn.envelope import EnvelopePoly
from tensor_frames.nn.pointnet.pointnet import PointNetEncoder
from tensor_frames.utils.point_sampling import CustomPointSampler
from tensor_frames.nn.embedding.angular import TrivialAngularEmbedding, compute_edge_vec
from tensor_frames.nn.mlp import MLPWrapped
from tensor_frames.lframes import IdentityLFrames


class QM9AlignedLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_function: torch.nn.Module,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "loss_function"])

        self.net = net

        self.criterion = loss_function
        self.sigmoid = torch.nn.Sigmoid()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_bin_acc = BinaryAccuracy()
        self.val_bin_acc = BinaryAccuracy()
        self.test_bin_acc = BinaryAccuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        pre_logits = self.forward(batch)

        loss = self.criterion(pre_logits, batch.is_transformed)  # + orthogonality_loss * 0.001
        return loss, pre_logits

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, pre_logits = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )
        self.train_bin_acc(self.sigmoid(pre_logits), batch.is_transformed.int())
        self.log(
            "train/acc",
            self.train_bin_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, pre_logits = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )
        self.val_bin_acc(self.sigmoid(pre_logits), batch.is_transformed.int())
        self.log(
            "val/acc",
            self.val_bin_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )

        # log the learning rate
        lightning_optimizer = self.optimizers()  # self = your model
        for i, param_group in enumerate(lightning_optimizer.optimizer.param_groups):
            self.log(
                "learning_rate",
                param_group["lr"],
                on_step=False,
                on_epoch=True,
                batch_size=batch.batch_size,
            )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, pre_logits = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        self.test_bin_acc(self.sigmoid(pre_logits), batch.is_transformed.int())
        self.log(
            "test/acc",
            self.test_bin_acc,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            if self.hparams.scheduler.func == torch.optim.lr_scheduler.ChainedScheduler:
                schedulers = self.hparams.scheduler.keywords["schedulers"]

                tmp_schedulers = []

                for scheduler in schedulers:
                    tmp_schedulers.append(scheduler(optimizer=optimizer))

                scheduler = torch.optim.lr_scheduler.ChainedScheduler(tmp_schedulers)
            else:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/acc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


# Explicit configuration (converted from the previous YAML config)
DEFAULT_CONFIG: Dict[str, Any] = {}


class NodeEmbeddingV8(MessagePassing):
    def __init__(
        self,
        input_tensor_reps,
        tensor_reps,
        embed_dim,
    ):
        super().__init__()
        self.input_tensor_reps = input_tensor_reps
        self.in_dim = input_tensor_reps.dim
        self.tensor_reps = tensor_reps

        self.lin_1 = torch.nn.Linear(self.in_dim, self.tensor_reps.dim)
        self.lin_2 = torch.nn.Linear(embed_dim, self.tensor_reps.dim)

    def forward(self, x, edge_index, edge_embedding):
        self_emb = self.lin_1(x)
        prop = self.propagate(edge_index, edge_embedding=edge_embedding, size=(x.size(0), x.size(0)))
        return self_emb + prop

    def message(self, edge_embedding):
        return self.lin_2(edge_embedding)

class EdgeEmbedding(Module):
    def __init__(
        self,
        out_dim,
        radial_dim,
        angular_dim,
        hidden_layers,
        edge_attr_dim=0,
        activation=torch.nn.SiLU,
        dropout=0.0,
        **mlp_kwargs
    ):
        super().__init__()
        self.out_dim = out_dim
        self.radial_dim = radial_dim
        self.angular_dim = angular_dim
        self.edge_attr_dim = edge_attr_dim
        self.hidden_layers = hidden_layers.copy()
        self.hidden_layers.append(out_dim)
        self.mlp = MLP(
            radial_dim + angular_dim + edge_attr_dim,
            self.hidden_layers,
            activation_layer=activation,
            **mlp_kwargs
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, radial_embedding, angular_embedding, edge_attr):
        if angular_embedding is None:
            edge_embedding = radial_embedding
        else:
            edge_embedding = torch.cat([radial_embedding, angular_embedding], dim=1)

        if edge_attr is not None:
            edge_embedding = torch.cat([edge_embedding, edge_attr], dim=1)

        return self.dropout(self.mlp(edge_embedding))

class MoleculeNet(Module):
    def __init__(
        self,
        radial_module: Module,
        angular_module: Module,
        embedding_module: Module,
        layers: torch.nn.ModuleList,
        global_out: bool,
        pooling_method: str,
        output_tensor_reps: TensorReps,
        edge_embedding_module: Module | None = None,
    ):
        super().__init__()

        self.radial_module = radial_module
        self.angular_module = angular_module
        self.embedding_module = embedding_module
        self.layers = layers
        self.global_out = global_out
        self.pooling_method = pooling_method
        self.output_tensor_reps = output_tensor_reps
        self.edge_embedding_module = edge_embedding_module

    def forward(self, sample):
        lframes = IdentityLFrames()(pos=sample.pos)  # identity local frames are equivalent to not using local frames at all
        edge_vec = compute_edge_vec(sample.pos, sample.edge_index, lframes=lframes)
        radial_out = self.radial_module(edge_vec=edge_vec)
        angular_out = self.angular_module(edge_vec=edge_vec)

        if self.edge_embedding_module is not None:
            edge_embedding = self.edge_embedding_module(
                radial_embedding=radial_out,
                angular_embedding=angular_out,
                edge_attr=sample.edge_attr,
            )

        if isinstance(self.embedding_module, NodeEmbeddingV8):
            x = self.embedding_module(
                x=sample.x,
                edge_index=sample.edge_index,
                edge_embedding=edge_embedding,
            )
        else:
            raise NotImplementedError("Only NodeEmbeddingV8 is supported.")

        for i, layer in enumerate(self.layers):
            if isinstance(layer, PointNetEncoder):
                x, pos, batch, lframes, cached_layer_outputs = layer(
                    x=x,
                    pos=sample.pos,
                    batch=sample.batch,
                    lframes=lframes,
                )
            elif isinstance(layer, LayerNorm):
                x = layer(x, batch=sample.batch)
            else:
                x = layer(x)

        if self.global_out is not None:
            if self.pooling_method == "add":
                x = global_add_pool(x, sample.batch)
            elif self.pooling_method == "mean":
                x = global_mean_pool(x, sample.batch)
            elif self.pooling_method == "max":
                x = global_max_pool(x, sample.batch)

        return x


def build_default_net(radial_cutoff: float = 5.0):
    """Construct the default `MoleculeNet` using the tensor_frames building blocks
    mirroring the original YAML configuration.

    This function imports tensor_frames and project-specific modules lazily so the
    file can be imported even if those packages are not installed; an ImportError
    will be raised when the function is called and dependencies are missing.
    """

    # radial and angular modules
    num_radial_frequencies = 2
    num_angular_frequencies = 1
    radial = BesselEmbedding(num_frequencies=num_radial_frequencies, cutoff=radial_cutoff, envelope=EnvelopePoly(p=5))
    angular = AxisWiseBesselEmbedding(num_frequencies=num_angular_frequencies)

    # edge embedding
    edge_embedding = EdgeEmbedding(
        out_dim=4,
        radial_dim=num_radial_frequencies,
        angular_dim=3 * num_angular_frequencies,  # axis-wise angular embedding has 3 components per frequency
        hidden_layers=[4],
        edge_attr_dim=5,
        dropout=0.0,
    )

    # embedding module
    input_tensor_reps = TensorReps(tensor_reps="5x0n")
    tensor_reps = TensorReps(tensor_reps="4x0n")
    feature_dim = tensor_reps.dim
    embedding_module = NodeEmbeddingV8(input_tensor_reps=input_tensor_reps, tensor_reps=tensor_reps, embed_dim=edge_embedding.out_dim)

    # layers / modules list
    modules = []

    pn_encoder = PointNetEncoder(
        list_in_reps=[tensor_reps],
        list_hidden_channels=[[4]],
        sam_out_reps=tensor_reps,
        list_r=[10],
        list_center_sampler=[CustomPointSampler(ratio=1.0)],
        radial_module_type="gaussian",
        radial_module_kwargs={"num_gaussians": 2, "is_learnable": False},
        list_conv_kwargs={
            "aggr": "max",
            "concatenate_receiver_features_in_mlp1": True,
            "angular_module": TrivialAngularEmbedding(normalize=True),
            "activation_layer": torch.nn.SiLU,
            "norm_layer": torch_geometric.nn.norm.BatchNorm,
        },
    )
    modules.append(pn_encoder)

    # append LayerNorm:
    modules.append(torch_geometric.nn.LayerNorm(feature_dim))

    mlp = MLPWrapped(
        in_channels=feature_dim,
        hidden_channels=[4, 2, 1],
        activation_layer=torch.nn.SiLU,
    )
    modules.append(mlp)
    layers = torch.nn.ModuleList(modules)

    output_tensor_reps = TensorReps(tensor_reps="1x0n")
    net = MoleculeNet(
        radial_module=radial,
        angular_module=angular,
        embedding_module=embedding_module,
        edge_embedding_module=edge_embedding,
        layers=layers,
        output_tensor_reps=output_tensor_reps,
        global_out=True,
        pooling_method="add",
    )

    return net

# Optimizer builder: call with `params` to get an optimizer instance
def _default_optimizer_builder():
    return partial(torch.optim.AdamW, lr=5e-4, weight_decay=5e-3)

# Scheduler builder: call with `optimizer` and optional `max_epochs`
def _default_scheduler_builder(max_epochs=100):
    # Two-phase scheduler: Linear warmup then cosine annealing
    linear = partial(torch.optim.lr_scheduler.LinearLR, total_iters=5)
    cosine = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=max_epochs, eta_min=1e-6)
    return partial(torch.optim.lr_scheduler.ChainedScheduler, schedulers=[linear, cosine])

def build_default_classifier(net: torch.nn.Module = None, loss_function: torch.nn.Module = torch.nn.BCEWithLogitsLoss(), max_epochs: int = 100) -> QM9AlignedLitModule:
    """Helper to construct a `QM9AlignedLitModule` using the explicit defaults.

    :param net: The neural network module to train.
    :param loss_function: If provided, overrides the default loss function.
    :returns: An initialized `QM9AlignedLitModule` with optimizer and scheduler builders wired in.
    """
    if net is None:
        net = build_default_net()
    
    optimizer = _default_optimizer_builder()
    scheduler = _default_scheduler_builder(max_epochs=max_epochs)

    module = QM9AlignedLitModule(
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
    )
    return module
