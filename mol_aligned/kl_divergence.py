import numpy as np
from scipy.special import digamma
import torch
from tqdm.auto import tqdm

from mol_aligned.orientations import compute_pairwise_angle_distance

def estimate_so3_kl(angles_block, k=5, block_size=10000, verbose=True):
    """
    Computes Geometric Entropy and KL Divergence to Uniform on SO(3).
    """
    N = angles_block.shape[0]
    dists = angles_block.cpu().numpy()  # Convert to numpy for distance computation

    dists.sort(axis=1)
    
    # 2. k-NN Estimation
    theta_k = dists[:, k]
    
    # Volume of geodesic ball on SO(3): V(w) = 8*pi*(w - sin(w))
    volumes = 8 * np.pi * (theta_k - np.sin(theta_k))
    
    # Log volumes (add epsilon for stability)
    log_volumes = np.log(volumes + 1e-15)
    
    # 3. Compute Entropy
    h_est = digamma(N) - digamma(k) + np.mean(log_volumes)
    
    # 4. Compute KL Divergence to Uniform
    # Max Entropy of SO(3) is log(Total Volume) = log(8 * pi^2)
    max_entropy = np.log(8 * np.pi**2)
    kl_div = max_entropy - h_est
        
    return kl_div