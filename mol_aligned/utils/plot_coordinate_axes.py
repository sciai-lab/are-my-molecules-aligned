from .colors import COLORS
import pyvista as pv
import pyvista
import numpy as np

def get_plotter():
    pl = pv.Plotter(off_screen=True, notebook=False, image_scale=1)
    pl.camera_position = "zx"
    pl.enable_parallel_projection()
    return pl

import matplotlib.font_manager as fm
import matplotlib as mpl

default_font_name = mpl.rcParams['font.sans-serif'][0]
MATPLOTLIB_FONT_PATH = fm.findfont(default_font_name)

def get_local_frames_mesh_dict(
    origins: np.ndarray,
    bases: np.ndarray,
    scale: float = 1,
    axes_radius_scale: float = 0.03,
    cone_scale: float = 2.0,
    cone_aspect_ratio: float = 2,
    resolution: int = 20,
    axes_colors: np.ndarray = None,
) -> dict:
    """Create a mesh dict for a set of local frames, given by their origin and basis vectors.

    Args:
        origins: The origins of the local frames, shape (n, 3).
        bases: The basis vectors of the local frames, shape (n, 3=n_vectors, 3=n_axes).
        scale: The scale of the local frames.
        axes_radius_scale: The radius of the cylinders representing the axes.
        cone_scale: The scale of the cones representing the axes.
        cone_aspect_ratio: The aspect ratio of the cones representing the axes.
        resolution: The resolution of the cylinders and cones.

    Returns:
        A dictionary with keyword arguments to pass to :meth:`pyvista.Plotter.add_mesh`.
    """

    if axes_colors is None:
        axes_colors = ["#ff0000", "#00ff00", "#0000ff"]

    assert (
        bases.shape[2] == 3 and 0 < bases.shape[1] <= 3
    ), f"Invalid shape of bases: {bases.shape}. Must be (n, 1|2|3, 3)."
    assert origins.shape[1] == 3, f"Invalid shape of origins: {origins.shape}. Must be (n, 3)."
    assert origins.shape[0] == bases.shape[0], (
        f"Incompatible shapes of origins and bases: {origins.shape} and {bases.shape}. "
        f"Must be (n, 3) and (n, 1|2|3, 3)."
    )

    mesh_elements = []

    axes_radius = scale * axes_radius_scale
    cone_radius = axes_radius * cone_scale
    cone_height = cone_radius * cone_aspect_ratio * 2

    for origin, bases in zip(origins, bases):
        for color_id, basis_vector in enumerate(bases):
            cylinder_height = scale * np.linalg.norm(basis_vector, axis=-1) - cone_height
            if cylinder_height < 0:
                cylinder_height = 0
            basis_vector_length = np.linalg.norm(basis_vector)
            basis_vector = basis_vector / basis_vector_length
            dest = np.asarray(origin) + np.asarray(basis_vector) * np.asarray(cylinder_height)

            cylinder = pv.Cylinder(
                center=(origin + dest) / 2,  # one quarter of the way from i to j
                direction=dest - origin,
                radius=axes_radius,
                height=cylinder_height,
                resolution=resolution,
            )
            cylinder["color_ids"] = np.ones(cylinder.n_points) * color_id
            mesh_elements.append(cylinder)

            # add a cone at the tip of the arrow
            this_cone_height = min(cone_height, basis_vector_length)
            cone = pv.Cone(
                center=np.asarray(origin) + np.asarray(basis_vector) * (np.asarray(cylinder_height) + np.asarray(this_cone_height) / 2.),
                direction=basis_vector,
                height=this_cone_height,
                radius=cone_radius,
                resolution=resolution,
            )
            cone = cone.extract_geometry()
            cone["color_ids"] = np.ones(cone.n_points) * color_id
            mesh_elements.append(cone)

    merged_mesh = pv.MultiBlock(mesh_elements).combine().extract_surface()
    add_mesh_kwargs = dict(
        mesh=merged_mesh,
        smooth_shading=True,
        diffuse=0.5,
        specular=0.5,
        ambient=0.5,
        clim=(0, len(axes_colors) - 1),
        cmap=axes_colors,
        show_scalar_bar=False,
    )
    return add_mesh_kwargs

def crop_whitespace(img, tol=5):
    """
    Crop away white borders from an RGB image.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, 3), dtype uint8.
    tol : int, optional
        Tolerance for "white" (default 5). A pixel is white if all channels > 255 - tol.

    Returns
    -------
    cropped : np.ndarray
        Cropped image (may be smaller than input).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected an RGB image of shape (H, W, 3).")

    # Create mask of non-white pixels
    mask = (img < 255 - tol).any(axis=2)

    # If all white, just return original
    if not mask.any():
        return img

    # Find bounding box of non-white pixels
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slice end is exclusive

    return img[y0:y1, x0:x1]

def plot_coordinate_axes(
    axes: np.ndarray,
    plotter: pyvista.Plotter = None,
    res: int = 512,
    annotate_axes: bool = True,
    annotate_pcs: bool = True,
    colors: str = 'colorblind',
    autocrop: bool = True,
):
    """
    Plot a set of axes in 3D.
    Args:
        axes: (3, 3) axes or (n, 3, 3) array of n sets of 3 axes.
        plotter: pyvista plotter to use. If None, a new one will be created.
        res: Resolution of the output image before whitespace cropping. Better leave on default, messes with text scaling otherwise.
        annotate_axes: Whether to annotate the standard axes with their coordinates.
        annotate_pcs: Whether to annotate the axes with PC1, PC2, PC3.
        colors: colors to use for the axes. Can be 'colorblind' (default) or 'rgb', or a list of colors.
        autocrop: Whether to crop whitespace from the output image.

    Returns:
        img: The output image as a numpy array of shape (~res, ~res, 3).

    """
    if isinstance(colors, str):
        if colors == 'colorblind':
            colors = COLORS[:3]
        elif colors == 'rgb':
            colors = ["#FF0000", "#00FF00", "#0000FF"]
        else:
            raise ValueError(f'Invalid colors: {colors}')
    if plotter is None:
        plotter = get_plotter()
    if axes.ndim == 2:
        axes = axes[None]
    frame_mesh = get_local_frames_mesh_dict(
        origins = np.zeros((1, 3)),
        bases=axes,
        scale=2,
        axes_colors=colors,
    )
    global_frame_mesh = get_local_frames_mesh_dict(
        origins = np.zeros((2, 3)),
        bases=np.eye(3)[None] * np.array([-1, 1])[:, None, None],
        scale=2.5,
        axes_colors=["000000"]*3,
        cone_scale=0,
        axes_radius_scale=0.01,
    )

    plotter.add_mesh(**frame_mesh)
    plotter.add_mesh(**global_frame_mesh, opacity=0.1)

    font_settings = dict(font_file=MATPLOTLIB_FONT_PATH, font_size=int(33*res/512))

    # plotter.add_point_labels(np.eye(3) * 1, ['a', '(0,1,0)', '(0,0,1)'], point_size=20, text_color='black')
    text_dist = 2.7 if not annotate_pcs else 2.9
    if annotate_axes:
        plotter.add_point_labels(np.eye(3)[:2] * text_dist, ['(1,0,0)', '(0,1,0)'], point_size=20, text_color='black', show_points=False, justification_horizontal='center', justification_vertical='top', fill_shape=False, shape=None, always_visible=True, **font_settings)
        plotter.add_point_labels(np.eye(3)[-1:] * text_dist, ['(0,0,1)'], point_size=20, text_color='black', show_points=False, justification_horizontal='center', justification_vertical='bottom', fill_shape=False, shape=None, always_visible=True, **font_settings)

    if len(axes) == 1 and annotate_pcs:
        plotter.add_point_labels(axes[0] * 2.3, ['PC 1', 'PC 2', 'PC 3'], point_size=20, text_color='black', show_points=False, justification_horizontal='center', justification_vertical='center', fill_shape=False, shape=None, always_visible=True, **font_settings)


    plotter.camera_position="iso"
    plotter.reset_camera(bounds=0.8 * text_dist * np.stack([np.full(3, -1), np.full(3, 1)], axis=1).flatten())

    img = plotter.show(screenshot=True, window_size=(res, res))
    # crop whitespace
    if autocrop:
        img = crop_whitespace(img, tol=0)
    return img
