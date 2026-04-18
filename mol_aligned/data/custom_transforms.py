import math
import numbers
import random
from typing import List, Tuple, Union

import e3nn
import numpy as np
import torch
import torch_geometric.transforms as T
from e3nn.o3 import rand_matrix
from torch_geometric.data import Data
from torch_geometric.nn import fps
from mol_aligned.orientations import compute_pca


def transform_attributes(
    data: Data, matrix: torch.Tensor, transform_attrs: List[str], attrs_repr: List[str]
):
    # Apply the same transform to the target
    for attr, rep in zip(transform_attrs, attrs_repr):
        if rep == "vector":
            attr_data = getattr(data, attr)
            setattr(data, attr, attr_data @ matrix.T)
        else:
            raise NotImplementedError("only vector representation is supported")

    return data


class RandomJitter(T.BaseTransform):
    def __init__(
        self,
        sigma_max: float = 0.01,
        clip: float = 0.05,
        p: float = 0.5,
        sigma_min: float | None = None,
    ):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.clip = clip
        self.p = p

    def forward(self, data: Data):
        if random.random() < 1 - self.p:
            return data

        if self.sigma_min is not None:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
        else:
            sigma = self.sigma_max
        noise = torch.randn_like(data.pos) * sigma
        if self.clip is not None:
            noise = torch.clamp(noise, -self.clip, self.clip)
        data.pos += torch.randn_like(data.pos) * self.sigma_max

        return data
    

class UniformRandomRotate(T.BaseTransform):
    """An example on how to write a custom transform."""

    def __init__(
        self,
        transform_attrs: List[str] = None,
        attrs_repr: List[str] = None,
        center_around_origin: bool = True,
        add_is_transformed_flag: bool = False,
        p: float = 1.0,
        deterministic: bool = False,
    ):
        self.matrix = None
        self.center_around_origin = center_around_origin
        self.add_is_transformed_flag = add_is_transformed_flag
        self.transform_attrs = transform_attrs
        self.attrs_repr = attrs_repr
        self.p = p
        self.deterministic = deterministic
        if self.transform_attrs is not None:
            assert len(self.transform_attrs) == len(
                self.attrs_repr
            ), "transform_attrs and attrs_repr must have the same length"

    def forward(self, data, use_stored_matrix=False):
        # use the pos to get a random seed:
        if self.deterministic:
            seed = data.idx.item()
            random.seed(seed)
            torch.manual_seed(seed)

        if self.center_around_origin:
            # Center the data around the origin before applying the rotation
            data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)

        if random.random() > self.p:
            # do not apply the transform with probability 1 - p

            # transform pos with identity matrix
            matrix = rand_matrix()
            matrix = torch.eye(3, dtype=matrix.dtype, device=data.pos.device)
            data = T.LinearTransformation(matrix)(data)

            if self.add_is_transformed_flag:
                data.is_transformed = torch.tensor([[0.]], device=data.pos.device)
            return data
        
        if use_stored_matrix:
            assert self.matrix is not None, "matrix is not stored"
            matrix = self.matrix
        else:
            matrix = rand_matrix()
            self.matrix = matrix


        data = T.LinearTransformation(matrix)(data)
        if self.transform_attrs is not None:
            data = transform_attributes(data, matrix, self.transform_attrs, self.attrs_repr)

        if self.add_is_transformed_flag:
            data.is_transformed = torch.tensor([[1.]], device=data.pos.device)

        return data

    def __repr__(self):
        # Return a string representation of the transform
        return f"{self.__class__.__name__}"


class UniformRandomRotateAngleBounded(T.BaseTransform):
    def __init__(
        self,
        transform_attrs: List[str] = None,
        attrs_repr: List[str] = None,
        center_around_origin: bool = True,
        add_is_transformed_flag: bool = False,
        p: float = 1.0,
        deterministic: bool = False,
        max_angle: float = 10.0,
        min_angle: float = 0.0,
    ):
        self.matrix = None
        self.center_around_origin = center_around_origin
        self.add_is_transformed_flag = add_is_transformed_flag
        self.transform_attrs = transform_attrs
        self.attrs_repr = attrs_repr
        self.p = p
        self.deterministic = deterministic
        self.min_angle = min_angle
        self.max_angle = max_angle
        if self.transform_attrs is not None:
            assert len(self.transform_attrs) == len(
                self.attrs_repr
            ), "transform_attrs and attrs_repr must have the same length"

    def rejection_sampling(self, max_angle, min_angle, batch_size=10000, max_iter=1000):
        # if max_angle is 0, simply return identity matrix
        if max_angle == 0.0:
            return torch.eye(3)

        for _ in range(max_iter):
            matrices = rand_matrix(batch_size)
            angles = torch.acos((matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2] - 1) / 2)
            angles = angles * 180.0 / math.pi
            mask = (angles <= max_angle) & (angles >= min_angle)
            if mask.sum() > 0:
                return matrices[mask][0]
            
        raise RuntimeError("Failed to sample a rotation matrix within the angle bounds.")

    def forward(self, data, use_stored_matrix=False):
        # use the pos to get a random seed:
        if self.deterministic:
            seed = data.idx
            random.seed(seed)
            torch.manual_seed(seed)

        if self.center_around_origin:
            # Center the data around the origin before applying the rotation
            data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)

        if random.random() > self.p:
            # do not apply the transform with probability 1 - p

            # transform pos with identity matrix
            matrix = rand_matrix()
            matrix = torch.eye(3, dtype=matrix.dtype, device=data.pos.device)
            data = T.LinearTransformation(matrix)(data)

            if self.add_is_transformed_flag:
                data.is_transformed = torch.tensor([[0.]], device=data.pos.device)
            return data
        
        if use_stored_matrix:
            assert self.matrix is not None, "matrix is not stored"
            matrix = self.matrix
        else:
            matrix = self.rejection_sampling(max_angle=self.max_angle, min_angle=self.min_angle)
            matrix = matrix.to(data.pos.device)
            self.matrix = matrix


        data = T.LinearTransformation(matrix)(data)
        if self.transform_attrs is not None:
            data = transform_attributes(data, matrix, self.transform_attrs, self.attrs_repr)

        if self.add_is_transformed_flag:
            data.is_transformed = torch.tensor([[1.]], device=data.pos.device)

        return data

    def __repr__(self):
        # Return a string representation of the transform
        return f"{self.__class__.__name__}"


class ReplaceXWithPCAVectors(T.BaseTransform):
    """Replaces the 'x' attribute with the PCA of the 'pos' attribute."""

    def __init__(self, use_mass_weighting: bool = False, use_random_sign: bool = False):
        self.use_mass_weighting = use_mass_weighting
        self.use_random_sign = use_random_sign

    def forward(self, data: Data) -> Data:
        eigvecs = compute_pca(data, use_mass_weighting=self.use_mass_weighting)

        # randomly flip the sign of the eigenvectors to remove ambiguity
        if self.use_random_sign:
            eigvecs = eigvecs * torch.sign(torch.randn(3, device=data.pos.device)).view(3, 1)

        # 4. Replace 'x' with the first two PCA vectors
        data.x = eigvecs[:2, :].reshape(1, 6)  # [1, 6] for two vectors in 3D

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"