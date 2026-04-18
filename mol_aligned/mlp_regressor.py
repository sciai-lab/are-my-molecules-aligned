from functools import partial
from typing import Any, Dict, Tuple

import torch
import torch_geometric
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MeanSquaredError
from torchmetrics.regression import mae

from tensor_frames.nn.mlp import MLPWrapped


class QM9RegressionLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
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
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_function: torch.nn.Module = torch.nn.MSELoss(),
        use_channelwise_l2_test_metric: bool = False,
        output_dim: int = 1,
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

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mae = mae.MeanAbsoluteError()
        self.val_mae = mae.MeanAbsoluteError()
        self.test_mae = mae.MeanAbsoluteError()
        self.test_mse = MeanSquaredError()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.use_channelwise_l2_test_metric = use_channelwise_l2_test_metric

    def forward(self, batch) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(batch.x)

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

        logits = self.forward(batch)

        loss = self.criterion(logits, batch.y)
        return loss, logits

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits = self.model_step(batch)

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
        self.train_mae(logits, batch.y)
        self.log(
            "train/mae",
            self.train_mae,
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

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits = self.model_step(batch)

        if dataloader_idx == 0:
            val_str = "val"
        else:
            val_str = f"val_{dataloader_idx}"

        # update and log metrics
        self.val_loss(loss)
        self.log(
            f"{val_str}/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )
        self.val_mae(logits, batch.y)
        self.log(
            f"{val_str}/mae",
            self.val_mae,
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
                prog_bar=True,
                batch_size=batch.batch_size,
            )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def on_test_epoch_start(self):
        self.sample_count = 0
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits = self.model_step(batch)

        if dataloader_idx == 0:
            test_str = "test"
        else:
            test_str = f"test_{dataloader_idx}"

        # update and log metrics
        self.test_loss(loss)
        self.log(
            f"{test_str}/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )
        self.test_mae(logits, batch.y)
        self.log(
            f"{test_str}/mae",
            self.test_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )

        self.test_mse(logits, batch.y)
        self.log(
            f"{test_str}/mse",
            self.test_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )

        if self.use_channelwise_l2_test_metric:
            if self.sample_count == 0:
                self.channelwise_squared_errors = torch.zeros(
                    logits.shape[1], device=logits.device
                )
            squared_errors = (logits - batch.y) ** 2
            self.channelwise_squared_errors += squared_errors.sum(dim=0)
            self.sample_count += logits.shape[0]

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        if self.use_channelwise_l2_test_metric:
            # print the channel-wise L2 test errors
            channelwise_l2_errors = self.channelwise_squared_errors / self.sample_count
            for i, l2_error in enumerate(channelwise_l2_errors):
                print(f"Test L2 error for channel {i}: {l2_error.item():.6f}")
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
                    "monitor": "val/mae",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

# Optimizer builder: call with `params` to get an optimizer instance
def _default_optimizer_builder():
    return partial(torch.optim.Adam, lr=1e-3)

# Scheduler builder: call with `optimizer` and optional `max_epochs`
def _default_scheduler_builder(max_epochs=100):
    # Two-phase scheduler: Linear warmup then cosine annealing
    linear = partial(torch.optim.lr_scheduler.LinearLR, total_iters=5)
    cosine = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=max_epochs, eta_min=1e-6)
    return partial(torch.optim.lr_scheduler.ChainedScheduler, schedulers=[linear, cosine])

def build_default_regressor(net: torch.nn.Module = None, loss_function: torch.nn.Module = torch.nn.MSELoss(), max_epochs: int = 100) -> QM9RegressionLitModule:
    """Helper to construct a `QM9RegressionLitModule` using the explicit defaults.

    :param net: The neural network module to train.
    :param loss_function: If provided, overrides the default loss function.
    :returns: An initialized `QM9RegressionLitModule` with optimizer and scheduler builders wired in.
    """
    if net is None:
        net = MLPWrapped(in_channels=6, hidden_channels=[256, 256, 256, 256, 1], activation_layer=torch.nn.SiLU)
    
    optimizer = _default_optimizer_builder()
    scheduler = _default_scheduler_builder(max_epochs=max_epochs)

    module = QM9RegressionLitModule(
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
    )
    return module
