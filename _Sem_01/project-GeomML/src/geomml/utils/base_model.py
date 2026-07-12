from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Base class for all models in GeomML.

    Contract
    --------
    forward(batch) -> dict
        Must always return a dictionary containing all outputs required for
        inference and loss computation.

    loss_fn(outputs, batch) -> Tensor
        Computes a scalar loss from the outputs returned by forward()
        and the input batch.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch) -> dict[str, torch.Tensor]:
        """
        Run the model.

        Parameters
        ----------
        batch
            Input mini-batch.

        Returns
        -------
        dict
            Dictionary of model outputs.

        Examples
        --------
        Single-task model:
            {
                "y": prediction
            }

        Multi-task model:
            {
                "repr": h,
                "y": y_pred,
                "dipole": dipole_pred,
                "polar": polar_pred,
            }
        """
        raise NotImplementedError

    @abstractmethod
    def loss_fn(
        self,
        outputs: dict[str, torch.Tensor],
        batch,
    ) -> torch.Tensor:
        """
        Compute a scalar training loss.

        Parameters
        ----------
        outputs
            Dictionary returned by forward().
        batch
            Original batch.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        raise NotImplementedError
    
    
    @torch.no_grad()
    def predict(self, batch) -> dict[str, torch.Tensor]:
        """
        Inference helper.
        """
        self.eval()
        return self(batch)