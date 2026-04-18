import torch
from torch_geometric.data import Data, Batch
from pyscf.data.elements import MASSES
from tqdm import tqdm
from torch_geometric.utils import scatter
from torch_geometric.utils._scatter import scatter_argmax
from torch_geometric.nn import radius

def compute_pca(data: Data, use_mass_weighting: bool = False, fix_orientation: bool = True, orientation_method: str = "max", ensure_right_handed: bool = True):
    pos = data.pos
    if pos.size(1) < 3:
        raise ValueError(
            f"pos has {pos.size(1)} dimensions, but 3 are needed."
        )
    # Suppose sample.pos is [N, 3]
    N, d = pos.shape

    if use_mass_weighting:
        masses = torch.tensor([MASSES[int(z)] for z in data.z_original]).view(-1, 1)
        total_mass = masses.sum()
    else:
        masses = torch.ones((N, 1), device=pos.device)
        total_mass = N

    # 1. Center the positions
    com = (masses * pos).sum(dim=0, keepdim=True) / total_mass
    pos_centered = pos - com

    # 2. Compute covariance matrix (3x3)
    cov = pos_centered.T @ (masses * pos_centered) / total_mass

    # 3. Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)  # symmetric -> eigh is stable

    # Sort by eigenvalue descending (principal components first)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvecs = eigvecs.T

    if fix_orientation:
        if orientation_method == "max":
            # point each principal component in the direction of the atom with the largest projection
            for i in range(2):
                eigvec = eigvecs[i]
                projections = pos_centered @ eigvec  # (N,)
                argmax_abs_projections = torch.argmax(torch.abs(projections))
                sign_of_max_projection = torch.sign(projections[argmax_abs_projections])
                # break ties randomly (happens for collinear points):
                if sign_of_max_projection == 0:
                    sign_of_max_projection = torch.sign(torch.rand(1) - 0.5)
                eigvecs[i] *= sign_of_max_projection
        elif orientation_method == "count":
            # point each principal component in the direction in which most atoms lie
            for i in range(2):
                eigvec = eigvecs[i]
                projections = pos_centered @ eigvec  # (N,)
                summed_projection_signs = torch.sign(projections).sum()
                # break ties randomly
                if summed_projection_signs == 0:
                    summed_projection_signs = torch.sign(torch.rand(1) - 0.5)
                eigvecs[i] *= torch.sign(summed_projection_signs)
        else:
            raise ValueError(f"Unknown orientation_method: {orientation_method}")

    if ensure_right_handed:
        if torch.det(eigvecs) < 0:
            eigvecs[2, :] *= -1
        assert torch.allclose(torch.det(eigvecs), torch.tensor(1.0), atol=1e-5), f"{torch.det(eigvecs), eigvecs}, Eigenvectors do not form a right-handed system"

    # check that the eigvecs are orthogonal and have unit length
    assert torch.allclose(
        eigvecs.T @ eigvecs, torch.eye(3), atol=1e-6
    ), "Eigenvectors are not orthogonal"
    assert torch.allclose(
        torch.linalg.norm(eigvecs, dim=1), torch.ones(3), atol=1e-6
    ), "Eigenvectors do not have unit length"

    return eigvecs

def compute_pca_batched(batch: Batch, use_mass_weighting: bool = False, fix_orientation: bool = True, orientation_method: str = "max", ensure_right_handed: bool = True):
    """
    Compute PCA of the positions, optionally weighed by atomic mass, for each molecule in a batch.

    Args:
        batch: Batch of molecules (torch_geometric Batch object).
        use_mass_weighting: Whether to weigh positions by atomic mass.
        fix_orientation: Whether to fix the orientation of the principal components, so that they point in the direction of the largest projection or the direction in which most atoms lie.
        orientation_method:
        ensure_right_handed:

    Returns:

    """
    pos = batch.pos
    if pos.size(1) < 3:
        raise ValueError(
            f"pos has {pos.size(1)} dimensions, but 3 are needed."
        )
    # Suppose sample.pos is [N, 3]
    N, d = pos.shape

    if use_mass_weighting:
        masses = torch.tensor([MASSES[int(z)] for z in batch.z_original]).view(-1, 1)
        total_masses = scatter(masses, batch.batch, dim=0, reduce='sum')
    else:
        masses = torch.ones((N, 1), device=pos.device)
        total_masses = scatter(masses, batch.batch, dim=0, reduce='sum')

    # 1. Center the positions
    coms = scatter(masses * pos, batch.batch, dim=0, reduce='sum') / total_masses
    pos_centered = pos - coms[batch.batch]

    relative_masses = masses / total_masses[batch.batch]
    cov_matrices = scatter(
            pos_centered.unsqueeze(-1) * (relative_masses * pos_centered).unsqueeze(-2),
            batch.batch,
            dim=0,
        )

    # compute the PCA:
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrices)

    # choose the directions to be o3 equivariant:
    eigenvectors = eigenvectors.transpose(-1, -2)  # (N_graphs, n_vec, dim_vec)

    # sort eigenvectors based on eigenvalues:
    sorted_idx = torch.argsort(eigenvalues, dim=-1, descending=True)
    eigenvectors = eigenvectors[torch.arange(eigenvectors.size(0))[:, None], sorted_idx, :]

    if fix_orientation:
        if orientation_method == "max":
            # point each principal component in the direction of the atom with the largest projection
            dots = torch.einsum("bk,bvk->bv", pos_centered, eigenvectors[batch.batch, :2])  # (N, 2)
            argmax_abs_dots = scatter_argmax(torch.abs(dots), batch.batch, dim=0) # (N_graphs, 2)
            sign_of_max_dots = torch.sign(dots[argmax_abs_dots, torch.arange(dots.size(1))])  # (N_graphs, 2)
            # break ties randomly (happens for collinear points):
            sign_of_max_dots[sign_of_max_dots == 0] = torch.sign(torch.rand(sign_of_max_dots[sign_of_max_dots == 0].shape, device=pos.device) - 0.5)
            eigenvectors[:, :2, :] *= sign_of_max_dots.unsqueeze(-1)
        elif orientation_method == "count":
            # point each principal component in the direction in which most atoms lie
            dots = torch.einsum("bk,bvk->bv", pos_centered, eigenvectors[batch.batch, :2])  # (N, 2)
            sum_sign_dots = scatter(torch.sign(dots), batch.batch, dim=0, reduce='sum') # (N_graphs, 2)
            # break ties randomly
            sum_sign_dots[sum_sign_dots == 0] = torch.sign(torch.rand(sum_sign_dots[sum_sign_dots == 0].shape, device=pos.device) - 0.5)
            eigenvectors[:, :2, :] *= torch.sign(sum_sign_dots).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown orientation_method: {orientation_method}")

    if ensure_right_handed:
        determinants = torch.det(eigenvectors)
        flip_mask = determinants < 0
        eigenvectors[flip_mask, 2, :] *= -1
        assert torch.allclose(torch.det(eigenvectors), torch.ones(eigenvectors.size(0), device=eigenvectors.device), atol=1e-6), "Eigenvectors do not form a right-handed system"
    
    # check that the eigvecs are orthogonal and have unit length
    assert torch.allclose(
        torch.bmm(eigenvectors, eigenvectors.transpose(-1, -2)),
        torch.eye(3, device=eigenvectors.device).unsqueeze(0),
        atol=1e-6,
    ), "Eigenvectors are not orthogonal"
    assert torch.allclose(
        torch.linalg.norm(eigenvectors, dim=-1), torch.ones((eigenvectors.size(0), 3), device=eigenvectors.device), atol=1e-6
    ), "Eigenvectors do not have unit length"

    return eigenvectors


def compute_inertia_eigenvectors(data: Data, use_mass_weighting: bool = False, sort_descending: bool = False, 
                                 ensure_right_handed: bool = True) -> torch.Tensor:
    """
    Compute principal axes (eigenvectors) of the moment of inertia tensor for each molecule.
    Returns a list of (3,3) numpy arrays (eigenvectors, sorted by eigenvalue).
    """
    pos = data.pos
    if pos.size(1) < 3:
        raise ValueError(
            f"pos has {pos.size(1)} dimensions, but 3 are needed."
        )
    # Suppose sample.pos is [N, 3]
    N, d = pos.shape

    if use_mass_weighting:
        masses = torch.tensor([MASSES[int(z)] for z in data.z_original]).view(-1, 1)
        total_mass = masses.sum()
    else:
        masses = torch.ones((N, 1), device=pos.device)
        total_mass = N

    # 1. Center the positions
    com = (masses * pos).sum(dim=0, keepdim=True) / total_mass
    pos_centered = pos - com

    I = torch.zeros((3, 3))
    for i in range(len(masses)):
        x, y, z = pos_centered[i]
        x, y, z = x.item(), y.item(), z.item()
        m = masses[i].item()
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)
        I[0, 1] -= m * x * y
        I[0, 2] -= m * x * z
        I[1, 2] -= m * y * z
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    
    eigvals, eigvecs = torch.linalg.eigh(I)
    
    # Sort by eigenvalue descending (principal components first)
    idx = torch.argsort(eigvals, descending=sort_descending)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvecs = eigvecs.T

    if ensure_right_handed:
        if torch.det(eigvecs) < 0:
            eigvecs[2, :] *= -1

    # check that the eigvecs are orthogonal and have unit length
    assert torch.allclose(
        eigvecs.T @ eigvecs, torch.eye(3), atol=1e-6
    ), "Eigenvectors are not orthogonal"
    assert torch.allclose(
        torch.linalg.norm(eigvecs, dim=1), torch.ones(3), atol=1e-6
    ), "Eigenvectors do not have unit length"

    return eigvecs


def compute_angle_distance(axes_i, axes_j, consider_sign_flips=True):
    # Compute angle distance between two sets of axes (3x3 matrices)
    # axes_i: (m, 3, 3) or (1, 3, 3)
    # axes_j: (m, 3, 3)
    signs = [[1,1],[1,-1],[-1,1],[-1,-1]]
    # assert axes_i.shape == axes_j.shape, "axes_i and axes_j must have the same shape"
    m = axes_i.shape[0]
    if consider_sign_flips:
        angles = torch.full((m,), float('inf'), dtype=torch.float32, device=axes_i.device)
        for sign in signs:
            axes_i_signed = axes_i.clone()
            axes_i_signed[:, 0] *= sign[0]
            axes_i_signed[:, 1] *= sign[1]
            axes_i_signed[:, 2] *= torch.linalg.det(axes_i_signed)[:, None]
            axes_j_signed = axes_j.clone()
            axes_j_signed[:, 2] *= torch.linalg.det(axes_j_signed)[:, None]
            # Compute rotation matrices
            axes_i_exp = axes_i_signed.unsqueeze(1)  # (1, 1, 3, 3)
            axes_j_exp = axes_j_signed.unsqueeze(0)  # (1, block, 3, 3)
            Rmat = torch.matmul(axes_j_exp, axes_i_exp.transpose(-2, -1))  # (1, block, 3, 3)
            traces = Rmat.diagonal(dim1=-2, dim2=-1).sum(-1).squeeze(0)  # (block,)
            cos_theta = (traces - 1) / 2
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angles = torch.minimum(angles, torch.acos(cos_theta))
    else:
        # Compute rotation matrices
        axes_i_exp = axes_i.unsqueeze(1)  # (m, 1, 3, 3)
        axes_j_exp = axes_j.unsqueeze(0)  # (1, m, 3, 3)
        Rmat = torch.matmul(axes_j_exp, axes_i_exp.transpose(-2, -1))  # (m, m, 3, 3)
        traces = Rmat.diagonal(dim1=-2, dim2=-1).sum(-1)  # (m, m)
        cos_theta = (traces - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angles = torch.acos(cos_theta)
    return angles

def compute_single_angle_distance(axes_i, axes_j, consider_sign_flips=True):
    return compute_angle_distance(axes_i.view(1,3,3), axes_j.view(1,3,3), consider_sign_flips=consider_sign_flips).item()

def compute_pairwise_angle_distance(eigenvecs, block_size=10000, device='cpu', consider_sign_flips=True):
    # Blockwise vectorized pairwise angle calculation, compute on GPU, store on CPU

    # check that all rot have determinant +1
    assert torch.allclose(torch.det(eigenvecs), torch.ones(len(eigenvecs)), atol=1e-6), "Not all rotation matrices have determinant +1"

    n = len(eigenvecs)

    angles = torch.zeros((n, n), dtype=torch.float32)  # CPU
    num_blocks = (n + block_size - 1) // block_size
    for i_block in tqdm(range(num_blocks), desc='Block rows'):
        i_start = i_block * block_size
        i_end = min((i_block + 1) * block_size, n)
        axes_i = eigenvecs[i_start:i_end].to(device)  # (b1, 3, 3)
        for j_block in range(i_block, num_blocks):
            j_start = j_block * block_size
            j_end = min((j_block + 1) * block_size, n)
            axes_j = eigenvecs[j_start:j_end].to(device)  # (b2, 3, 3)
            block_angles = compute_angle_distance(axes_i, axes_j, consider_sign_flips=consider_sign_flips)  # (b1, b2)
            angles[i_start:i_end, j_start:j_end] = block_angles.float().cpu()
            if i_block != j_block:
                angles[j_start:j_end, i_start:i_end] = block_angles.T  # symmetry
    print(f"Blockwise torch-parallelized distance matrix computed for eigenvectors (vectorized, block size {block_size}). Stored on CPU.")

    return angles

def compute_single_row_of_distance_matrix(eigenvecs, index, block_size=10000, device='cpu', consider_sign_flips=True):
    # Compute a single row of the pairwise angle distance matrix, compute on GPU, store on CPU

    assert torch.allclose(torch.det(eigenvecs), torch.ones(len(eigenvecs)), atol=1e-6), "Not all rotation matrices have determinant +1"

    n = len(eigenvecs)

    axes_i = eigenvecs[index:index+1].to(device)  # (1, 3, 3)

    row_angles = torch.zeros(n, dtype=torch.float32)  # CPU
    num_blocks = (n + block_size - 1) // block_size
    for j_block in tqdm(range(num_blocks), desc='Blockwise single row'):
        j_start = j_block * block_size
        j_end = min((j_block + 1) * block_size, n)
        block_angles = torch.full((j_end - j_start,), float('inf'), dtype=torch.float32, device=device)
        axes_j = eigenvecs[j_start:j_end].to(device)  # (block, 3, 3)
        block_angles = compute_angle_distance(axes_i, axes_j, consider_sign_flips=consider_sign_flips)  # (block,)
        row_angles[j_start:j_end] = block_angles.float().cpu()
    print(f"Blockwise single row of distance matrix computed for index {index} (block size {block_size}). Stored on CPU.")

    return row_angles