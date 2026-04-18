import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import markers
# from src.utils.colors import COLORS
import matplotlib.transforms as mtransforms
from mol_aligned.utils.plot_coordinate_axes import plot_coordinate_axes


IBM_COLORS = ["#648fff", "#ffb000", "#dc267f", "#785ef0", "#fe6100", "#000000", "#ffffff"]
COLORS = IBM_COLORS

# set color cycle for matplotlib
def set_mpl_color_cycle():
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)

# function to reset to default matplotlib color cycle
def reset_mpl_color_cycle():
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=mpl.rcParamsDefault['axes.prop_cycle'].by_key()['color'])

set_mpl_color_cycle()

# rot_mat = np.array([
#     [-1, 0, 0],
#     [0, 1, 0],
#     [0, 0, -1]
# ])

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

def add_scatter_legend(ax, base_colors, with_marked=True, **legend_kwargs):
    colors = base_colors
    # First three entries (PC 1, PC 2, PC 3) as circles
    handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=c,
                   markeredgecolor=c, markersize=5, label=f"PC {i+1}")
        for i, c in enumerate(colors)
    ]

    # Custom entry: three circular markers side by side
    multi_circles = tuple(
        plt.Line2D([0], [0], marker='o', color='none',
                   markerfacecolor=c, markeredgecolor='black',
                   markeredgewidth=2, markersize=np.sqrt(100))
        for c in colors
    )

    # Add it to handles with a single label
    if with_marked:
        handles.append(multi_circles)
    labels = [f"PC {i+1}" for i in range(3)] + ["most common\norientation"] if with_marked else []

    # Legend with HandlerTuple to group markers
    loc = legend_kwargs.pop('loc', 'upper right')
    # ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc=loc, **legend_kwargs)
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)},
              loc='upper left', bbox_to_anchor=(0.9, 1.0), borderaxespad=0., **legend_kwargs)


def axes_to_long_lat(axes):
    """
    Convert Cartesian axes (x, y, z) to spherical coordinates (longitude, latitude)

    Args:
        axes: (..., 3)

    Returns:
        longitueds: (...)
        lattitudes: (...)
    """
    longitudes = np.arctan2(axes[..., 1], axes[..., 0])
    latitudes = np.arcsin(axes[..., 2])
    return longitudes, latitudes

def plot_rotation_matrices_mollweide(
        rotations: torch.Tensor,
        marked_axes: torch.Tensor = None,
        color_by=None,
        cmap='viridis',
        cbar_label=None,
        joint: bool = True,
        size: float = 4.0,
        suptitle=None,
        legend=True,
        legend_kwargs=None,
        plot_marked_axes=True,
        q=0.1,
        **kwargs
):
    """
    Make scatter plots of rotation matrix axes on a Mollweide projection of the full sphere.

    This function interprets each (3, 3) matrix as a set of three row vectors.
    - Vector 1: matrix[0, :]
    - Vector 2: matrix[1, :]
    - Vector 3: matrix[2, :]

    Args:
        rotations: (N, 3, 3) tensor of rotation matrices.
        color_by: (N,) tensor of scalar values for coloring points when joint is False.
        joint: If True, plot all three axes on a single plot (colored R, G, B).
               If False, create separate plots for each axis, colored by `color_by`.
        size: Figure size parameter.
        **kwargs: Additional keyword arguments for the scatter plot (e.g., alpha, s).
    """
    # Extract the three sets of row vectors as specified in the request
    rotations = rotations #@ rot_mat
    axes_vectors = [rotations[:, i, :] for i in range(3)]

    if marked_axes is not None:
        if marked_axes.ndim == 2:
            marked_axes = marked_axes[None]
        # compute longitudes and latitudes for marked axes
        # marked_axes = marked_axes.permute(0, 2, 1)
        marked_long, marked_lat = axes_to_long_lat(marked_axes)

    if joint:
        # Create a single plot for all three axes
        fig, ax = plt.subplots(1, 1, figsize=(size * 1.8, size), subplot_kw={'projection': 'mollweide'})
        # colors = np.eye(3)
        base_colors = np.array(COLORS[:3])
        # base_colors = np.array(['r', 'g', 'b'])
        # labels = ['PC 1', 'PC 2', 'PC 3']

        longitudes = []
        latitudes = []
        for i in range(3):
            subset_axes = axes_vectors[i].cpu().numpy()
            if subset_axes.shape[0] == 0:
                continue
            # Convert Cartesian axes (x, y, z) to spherical coordinates (longitude, latitude)
            long, lat = axes_to_long_lat(subset_axes)
            longitudes.append(long)
            latitudes.append(lat)

        # c = colors[None].repeat(axes_vectors[0].shape[0]).reshape(3, -1).T
        c = base_colors[None].repeat(axes_vectors[0].shape[0])

        longitudes = np.concatenate(longitudes)
        latitudes = np.concatenate(latitudes)
        # Shuffle for better visualization of dense areas
        perm = np.random.permutation(len(longitudes))

        # Scatter plot
        ax.scatter(longitudes[perm], latitudes[perm], alpha=kwargs.get('alpha', 0.5), s=kwargs.get('s', 2),
                   lw=0, c=c[perm], rasterized=True, zorder=1)

        # Configure plot aesthetics
        # ax.set_title(f"Joint Plot of Rotation Matrix Rows ({rotations.shape[0]} matrices)")

        # ax.grid(True, linestyle="--", alpha=1)
        ax.grid(True, linestyle="--", c='k', lw=0.5, alpha=1, zorder=-1000)
        ax.set_xticks(np.array([-180, -90, 0, 90, 180]) * np.pi/180, labels=["", "(0,-1,0)", "(1,0,0)", "(0,1,0)", ""])
        ax.set_yticks(np.array([90, 60, 30, 0, -30, -60, -90]) * np.pi/180, labels=["(0,0,1)", "", "", "(-1,0,0)", "", "", "(0,0,-1)"])

        # shift x ticks
        dx, dy = 0, -21  # shift: 0 in x, -15 pixels in y
        offset = mtransforms.ScaledTranslation(dx/72., dy/72., fig.dpi_scale_trans)
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + offset)

        if marked_axes is not None:
            markers = ['o', '^', '*', 's', 'v', 'x', '+']
            assert len(marked_axes) <= len(markers)
            for i, (long, lat) in enumerate(zip(marked_long, marked_lat)):
                ax.scatter(long, lat, c=base_colors, marker=markers[i], lw=2, s=100, edgecolors='k', zorder=10)

            if plot_marked_axes:
                # plot the first marked axis
                img = plot_coordinate_axes(marked_axes[:1].numpy(), colors=list(base_colors))
                # pad image with zeros on the top
                img = np.pad(img, ((3, 0), (0, 0), (0, 0)), mode='constant', constant_values=255)

                # fig2, ax2 = plt.subplots()
                # ax2.imshow(img)
                # fig2.show()
                # inset_ax = inset_axes(ax, width="15%", height="30%", loc='lower right')
                inset_ax = ax.inset_axes([0.93, -0.1, 0.3, 0.55], zorder=-2000)
                inset_ax.imshow(img)
                inset_ax.axis('off')


        if legend:
            legend_kwargs = dict() if legend_kwargs is None else legend_kwargs
            add_scatter_legend(ax, base_colors, with_marked=marked_axes is not None, **legend_kwargs)
        # ax.legend()

    else:  # Not joint, create 3 separate plots
        assert marked_axes is None, f'marked_axes must be None if joint is false, but got {marked_axes}'
        if color_by is None:
            raise ValueError("`color_by` must be provided when `joint` is False.")

        num_plots = 3
        fig, axes_plots = plt.subplots(num_plots, 1, figsize=(size * 2, size * num_plots),
                                        subplot_kw={'projection': 'mollweide'})

        # Ensure axes_plots is always a list for consistent indexing
        if num_plots == 1:
            axes_plots = [axes_plots]

        titles = [f"PC {i}" for i in range(3)]

        # Prepare color mapping once for all plots
        c_data = color_by.cpu().float().numpy()
        cmap = plt.get_cmap(cmap)

        # Calculate color limits based on quantiles, with fallback for uniform data
        vmin, vmax = np.quantile(c_data, [q, 1-q])
        if vmin == vmax: # Handle case where quantile values are identical
            vmin, vmax = c_data.min(), c_data.max()

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        c_rgba = sm.to_rgba(c_data)

        for i in range(num_plots):
            ax = axes_plots[i]
            subset_axes = axes_vectors[i].cpu().numpy()

            if subset_axes.shape[0] == 0:
                ax.set_title(f"{titles[i]}\n(No data)")
                ax.grid(True)
                continue

            # Convert Cartesian to spherical coordinates
            longitude = np.arctan2(subset_axes[:, 1], subset_axes[:, 0])
            latitude = np.arcsin(subset_axes[:, 2])

            # Shuffle points
            perm = np.random.permutation(len(longitude))

            # Scatter plot
            ax.scatter(longitude[perm], latitude[perm], alpha=kwargs.get('alpha', 0.5), s=kwargs.get('s', 2),
                       lw=0, c=c_rgba[perm], rasterized=True, zorder=1)

            # ax.grid(True, linestyle="--", c='k', lw=0.5, alpha=1, zorder=-1000)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_xticks(np.array([-180, -90, 0, 90, 180]) * np.pi/180, labels=["", "(0,-1,0)", "(1,0,0)", "(0,1,0)", ""])
            # ax.set_yticks(np.array([90, 60, 30, 0, -30, -60, -90]) * np.pi/180, labels=["(0,0,1)", "", "", "(-1,0,0)", "", "", "(0,0,-1)"])

            # shift x ticks
            dx, dy = 0, -21  # shift: 0 in x, -15 pixels in y
            offset = mtransforms.ScaledTranslation(dx/72., dy/72., fig.dpi_scale_trans)
            for label in ax.get_xticklabels():
                label.set_transform(label.get_transform() + offset)

            ax.set_title(f"{titles[i]}")

            # Add colorbar to middle axis=
            sm.set_array([]) # Necessary to ensure the colorbar uses the correct norm
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            if cbar_label is not None:
                cbar.set_label(cbar_label)
            if i != 1:
                # hide the colorbar
                cbar.ax.cla()
                cbar.ax.axis('off')

            # Configure plot aesthetics
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])

    if suptitle is not None:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return fig