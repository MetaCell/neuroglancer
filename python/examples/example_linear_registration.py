import argparse
import threading
import webbrowser
from copy import deepcopy
from time import ctime, time

import neuroglancer
import neuroglancer.cli
import numpy as np
import scipy.ndimage

MESSAGE_DURATION = 5  # seconds
NUM_DEMO_DIMS = 2  # Currently can be 2D or 3D
AFFINE_NUM_DECIMALS = 4

MARKERS_SHADER = """
#uicontrol vec3 fixedPointColor color(default="#00FF00")
#uicontrol vec3 movingPointColor color(default="#0000FF")
#uicontrol float pointSize slider(min=1, max=16, default=6)
void main() {
    if (int(prop_group()) == 0) {
        setColor(fixedPointColor);
    } else {
        setColor(movingPointColor);
    }
    setPointMarkerSize(pointSize);
}
"""


def debounce(wait: float):
    def decorator(fn):
        timer = None

        def debounced(*args, **kwargs):
            nonlocal timer

            if timer is not None:
                timer.cancel()

            timer = threading.Timer(wait, lambda: fn(*args, **kwargs))
            timer.start()

        return debounced

    return decorator


def fit_model(fixed_points: np.ndarray, moving_points: np.ndarray):
    """
    Choose the appropriate model based on number of points and dimensions.

    Inspired by https://github.com/AllenInstitute/render-python/blob/master/renderapi/transform/leaf/affine_models.py
    """
    assert fixed_points.shape == moving_points.shape
    N, D = fixed_points.shape

    if N == 1:
        return translation_fit(fixed_points, moving_points)
    if N == 2:
        return rigid_or_similarity_fit(fixed_points, moving_points, rigid=True)
    if N == 3 and D == 2:
        return affine_fit(fixed_points, moving_points)
    if N == 3 and D > 2:
        return rigid_or_similarity_fit(fixed_points, moving_points, rigid=False)
    return affine_fit(fixed_points, moving_points)


# See https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
# and https://math.nist.gov/~JBernal/kujustf.pdf
def rigid_or_similarity_fit(
    fixed_points: np.ndarray, moving_points: np.ndarray, rigid: bool = True
):
    N, D = fixed_points.shape

    # Remove translation aspect to first determine rotation/scale
    X = fixed_points - fixed_points.mean(axis=0)
    Y = moving_points - moving_points.mean(axis=0)

    # Cross-covariance
    sigma = (Y.T @ X) / N

    # SVD - Unitary matrix, Diagonal, conjugate transpose of unitary matrix
    U, S, Vt = np.linalg.svd(sigma)  # Sigma ≈ U diag(S) V*

    d = np.ones(D)
    if np.linalg.det(U @ Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt

    # Scale
    if rigid:
        s = 1.0
    else:
        var_src = (X**2).sum() / N  # sum of variances across dims
        s = (S * d).sum() / var_src

    # Translation
    t = Y - s * (R @ X)

    # Homogeneous (D+1)x(D+1)
    T = np.zeros((D, D + 1))
    T[:D, :D] = s * R
    T[:, -1] = -1 * np.diagonal(t)

    affine = np.round(T, decimals=AFFINE_NUM_DECIMALS)
    return affine


def translation_fit(fixed_points: np.ndarray, moving_points: np.ndarray):
    N, D = fixed_points.shape

    estimated_translation = np.mean(moving_points - fixed_points, axis=0)

    affine = np.zeros((D, D + 1))
    affine[:, :D] = np.eye(D)
    affine[:, -1] = estimated_translation

    affine = np.round(affine, decimals=AFFINE_NUM_DECIMALS)
    return affine


def affine_fit(fixed_points: np.ndarray, moving_points: np.ndarray):
    N, D = fixed_points.shape

    # Target values (B) is a D * N array
    # Input values (A) is a D * N, (D * (D + 1)) array
    # Output estimation is a (D * (D + 1)) array
    A = np.zeros(((D * N), D * (D + 1)))
    for i in range(N):
        for j in range(D):
            start_index = j * D
            end_index = (j + 1) * D
            A[D * i + j, start_index:end_index] = moving_points[i]
            A[D * i + j, D * D + j] = 1
    B = fixed_points.flatten()

    # The estimated affine transform params will be flattened
    # and there will be D * (D + 1) of them
    # Format is x1, x2, ..., b1, b2, ...
    tvec, res, rank, sd = np.linalg.lstsq(A, B)

    # Put the flattened version back into the matrix
    affine = np.zeros((D, D + 1))
    for i in range(D):
        start_index = i * D
        end_index = start_index + D
        affine[i, :D] = tvec[start_index:end_index]
        affine[i, -1] = tvec[D * D + i]

    # Round to close decimals
    affine = np.round(affine, decimals=AFFINE_NUM_DECIMALS)
    return affine


def transform_points(affine: np.ndarray, points: np.ndarray):
    # Apply the affine transform to the points
    transformed = np.zeros_like(points)
    padded = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
    for i in range(len(points)):
        transformed[i] = affine @ padded[i]
    return transformed


# Only used if no data provided
def _create_demo_data(size: int | tuple = 60, radius: float = 20):
    data_size = (size,) * NUM_DEMO_DIMS if isinstance(size, int) else size
    data = np.zeros(data_size, dtype=np.uint8)
    if NUM_DEMO_DIMS == 2:
        yy, xx = np.indices(data.shape)
        center = np.array(data.shape) / 2
        circle_mask = (xx - center[1]) ** 2 + (yy - center[0]) ** 2 < radius**2
        data[circle_mask] = 255
        return data
    zz, yy, xx = np.indices(data.shape)
    center = np.array(data.shape) / 2
    sphere_mask = (xx - center[2]) ** 2 + (yy - center[1]) ** 2 + (
        zz - center[0]
    ) ** 2 < radius**2
    data[sphere_mask] = 255
    return data


# Only used if no data provided
def _create_demo_fixed_image():
    return neuroglancer.ImageLayer(
        source=[
            neuroglancer.LayerDataSource(neuroglancer.LocalVolume(_create_demo_data()))
        ]
    )


# Only used if no data provided
def _create_demo_moving_image():
    if NUM_DEMO_DIMS == 2:
        desired_output_matrix_homogenous = [
            [0.8, 0, 0],
            [0, 0.2, 0],
            [0, 0, 1],
        ]
    else:
        desired_output_matrix_homogenous = [
            [0.8, 0, 0, 0],
            [0, 0.2, 0, 0],
            [0, 0, 0.9, 0],
            [0, 0, 0, 1],
        ]
    inverse_matrix = np.linalg.inv(desired_output_matrix_homogenous)
    transformed = scipy.ndimage.affine_transform(
        _create_demo_data(),
        matrix=inverse_matrix,
    )
    print("target demo affine", inverse_matrix)
    return neuroglancer.ImageLayer(
        source=[neuroglancer.LayerDataSource(neuroglancer.LocalVolume(transformed))]
    )


def change_coord_names(dims: neuroglancer.CoordinateSpace, name_mod):
    return neuroglancer.CoordinateSpace(
        names=[n + name_mod for n in dims.names],
        units=dims.units,
        scales=dims.scales,
    )


def create_dimensions(viewer_dims: neuroglancer.CoordinateSpace, indices=None):
    names = viewer_dims.names
    units = viewer_dims.units
    scales = viewer_dims.scales
    if indices is not None:
        names = [names[i] for i in indices]
        units = [units[i] for i in indices]
        scales = [scales[i] for i in indices]

    return neuroglancer.CoordinateSpace(names=names, units=units, scales=scales)


class LinearRegistrationWorkflow:
    def __init__(self, args):
        starting_state = args.state
        self.two_coord_spaces = not args.single_coord_space
        self.annotations_name = args.annotations_name
        self.status_timers = {}
        self.stored_points = [[], []]
        self.stored_moving_dims = {}
        self.moving_layer_names = []
        self.input_dim_names = []
        self.output_dim_names = []
        self.stored_group_number = -1
        self.affine = None
        self.co_ords_ready = False
        self.ready = False
        self.last_updated_print = -1
        self.viewer = neuroglancer.Viewer()
        self.viewer.shared_state.add_changed_callback(
            lambda: self.viewer.defer_callback(self.on_state_changed)
        )

        if starting_state is None:
            self._add_demo_data_to_viewer()
        else:
            self.viewer.set_state(starting_state)

        self._set_status_message(
            "help",
            "Place fixed (reference) layers in the left hand panel, and moving layers (to be registered) in the right hand panel. Then press 't' once you have completed this setup.",
        )
        with self.viewer.txn() as s:
            self.setup_two_panel_layout(s)
        self.setup_viewer_actions()

    def update(self):
        """Primary update loop, called whenever the viewer state changes."""
        current_time = time()
        if current_time - self.last_updated_print > 5:
            print(f"Viewer states are successfully syncing at {ctime()}")
            self.last_updated_print = current_time
        # TODO make ready a status instead of two vars
        # TODO overall update the class attributes at the end to cleaner
        if self.co_ords_ready and not self.ready:
            with self.viewer.txn() as s:
                self.setup_registration_layers(s)
        if self.ready:
            if not self.two_coord_spaces:
                self.automatically_group_markers_and_update()
            self.update_affine()
            self._clear_status_messages()

    def setup_viewer(self):
        self.setup_second_coord_space()
        self._set_status_message(
            "help",
            "Place markers in pairs, starting with the fixed, and then the moving. The registered layer will automatically update as you add markers. Press 't' to toggle visiblity of the registered layer.",
        )
        self.co_ords_ready = True

    def setup_two_panel_layout(self, s: neuroglancer.ViewerState):
        all_layer_names = [layer.name for layer in s.layers]
        if len(all_layer_names) >= 2:
            half_point = len(all_layer_names) // 2
            group1_names = all_layer_names[:half_point]
            group2_names = all_layer_names[half_point:]
        else:
            group1_names = all_layer_names
            group2_names = all_layer_names
        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.LayerGroupViewer(layers=group1_names, layout="xy-3d"),
                neuroglancer.LayerGroupViewer(layers=group2_names, layout="xy-3d"),
            ]
        )
        # Unliked position solves rendering problem but makes navigation awkward
        # s.layout.children[1].position.link = "unlinked"
        # In theory we could make keep unlinked and then on state change check
        # but that could be not worth compared to trying to improve rendering
        s.layout.children[1].crossSectionOrientation.link = "unlinked"
        # s.layout.children[1].crossSectionScale.link = "unlinked"
        s.layout.children[1].projectionOrientation.link = "unlinked"
        # s.layout.children[1].projectionScale.link = "unlinked"

    def setup_second_coord_space(self):
        if not self.moving_layer_names:
            moving_layers = self.get_state().layout.children[1].layers
            self.moving_layer_names = moving_layers
            self._moving_idx = 0
        layer_name = self.moving_layer_names[self._moving_idx]
        info_future = self.viewer.volume_info(layer_name)
        info_future.add_done_callback(lambda f: self.save_coord_space_info(f))

    def combine_affine_across_dims(self, s: neuroglancer.ViewerState, affine):
        all_dims = s.dimensions.names
        moving_dims = self.output_dim_names
        # The affine matrix only applies to the moving dims
        # so we need to create a larger matrix that applies to all dims
        # by adding identity transforms for the real dims
        full_matrix = np.zeros((len(all_dims), len(all_dims) + 1))
        for i, dim in enumerate(all_dims):
            for j, dim2 in enumerate(all_dims):
                if dim in moving_dims and dim2 in moving_dims:
                    moving_i = moving_dims.index(dim)
                    moving_j = moving_dims.index(dim2)
                    full_matrix[i, j] = affine[moving_i, moving_j]
                elif dim == dim2:
                    full_matrix[i, j] = 1
            if dim in moving_dims:
                moving_i = moving_dims.index(dim)
                full_matrix[i, -1] = affine[moving_i, -1]
        return full_matrix

    def setup_registration_layers(self, s: neuroglancer.ViewerState):
        dimensions = s.dimensions
        # It is possible that the dimensions are not ready yet, return if so
        if len(dimensions.names) != self.num_dims:
            return

        # Make the annotation layer if needed
        if s.layers.index(self.annotations_name) == -1:
            if self.two_coord_spaces:
                s.layers[self.annotations_name] = neuroglancer.LocalAnnotationLayer(
                    dimensions=create_dimensions(s.dimensions)
                )
            else:
                s.layers[self.annotations_name] = neuroglancer.LocalAnnotationLayer(
                    dimensions=create_dimensions(s.dimensions),
                    annotation_properties=[
                        neuroglancer.AnnotationPropertySpec(
                            id="label",
                            type="uint32",
                            default=0,
                        ),
                        neuroglancer.AnnotationPropertySpec(
                            id="group",
                            type="uint8",
                            default=0,
                            enum_labels=["fixed", "moving"],
                            enum_values=[0, 1],
                        ),
                    ],
                    shader=MARKERS_SHADER,
                )

        # Make a copy of all the moving layers but in original coord space
        # and as part of the left hand panel
        for layer_name in self.moving_layer_names:
            copy = deepcopy(s.layers[layer_name])
            copy.name = layer_name + "_registered"
            copy.visible = False
            for source in copy.source:
                # TODO might need mapping
                source.transform = None
            s.layers[copy.name] = copy
            s.layout.children[0].layers.append(copy.name)
        s.layers[self.annotations_name].tool = "annotatePoint"
        s.selected_layer.layer = self.annotations_name
        s.selected_layer.visible = True
        s.layout.children[0].layers.append(self.annotations_name)
        s.layout.children[1].layers.append(self.annotations_name)
        self.setup_panel_coordinates(s)
        self.ready = True

    def setup_panel_coordinates(self, s: neuroglancer.ViewerState):
        dimensions = s.dimensions.names
        s.layout.children[1].displayDimensions.link = "unlinked"
        s.layout.children[1].displayDimensions.value = self.output_dim_names[:3]
        s.layout.children[0].displayDimensions.link = "unlinked"
        s.layout.children[0].displayDimensions.value = self.input_dim_names[:3]

    def save_coord_space_info(self, info_future):
        result = info_future.result()
        self.moving_name = self.moving_layer_names[self._moving_idx]
        self.stored_moving_dims[self.moving_name] = result.dimensions
        done = len(self.stored_moving_dims) == len(self.moving_layer_names)
        if not done:
            self._moving_idx += 1
            self.setup_second_coord_space()
            return
        # If we get here we have all the coord spaces ready and can update viewer
        with self.viewer.txn() as s:
            for layer_name in self.moving_layer_names:
                input_dims = self.stored_moving_dims[layer_name]
                output_dims = change_coord_names(input_dims, "2")
                self.input_dim_names = input_dims.names
                self.output_dim_names = output_dims.names
                self.num_dims = len(input_dims.names) * 2
                new_coord_space = neuroglancer.CoordinateSpaceTransform(
                    input_dimensions=input_dims,
                    output_dimensions=output_dims,
                )
                for source in s.layers[layer_name].source:
                    source.transform = new_coord_space

    def toggle_registered_visibility(self, _):
        if not self.ready:
            self.setup_viewer()
            return
        with self.viewer.txn() as s:
            for layer_name in self.moving_layer_names:
                registered_name = layer_name + "_registered"
                is_registered_visible = s.layers[registered_name].visible
                s.layers[registered_name].visible = not is_registered_visible

    def setup_viewer_actions(self):
        viewer = self.viewer
        viewer.actions.add(
            "toggleRegisteredVisibility", self.toggle_registered_visibility
        )

        with viewer.config_state.txn() as s:
            s.input_event_bindings.viewer["keyt"] = "toggleRegisteredVisibility"
            s.input_event_bindings.viewer["keyp"] = "screenshotStatistics"

    def is_fixed_image_space_last(self, dim_names):
        first_name = dim_names[0]
        return first_name not in self.input_dim_names

    def on_state_changed(self):
        self.viewer.defer_callback(self.update)

    @debounce(0.25)
    def automatically_group_markers_and_update(self):
        with self.viewer.txn() as s:
            self.automatically_group_markers(s)

    @debounce(1.5)
    def update_affine(self):
        with self.viewer.txn() as s:
            self.estimate_affine(s)

    def create_registered_image(self):
        with self.viewer.txn() as s:
            layer = deepcopy(s.layers[self.moving_name])
            layer.name = "registered"
            return layer

    def split_points_into_pairs(self, annotations, dim_names):
        if len(annotations) == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))
        if self.two_coord_spaces:
            real_dims_last = self.is_fixed_image_space_last(dim_names)
            num_points = len(annotations)
            num_dims = len(annotations[0].point) // 2
            fixed_points = np.zeros((num_points, num_dims))
            moving_points = np.zeros((num_points, num_dims))
            for i, a in enumerate(annotations):
                for j in range(num_dims):
                    fixed_index = j + num_dims if real_dims_last else j
                    moving_index = j if real_dims_last else j + num_dims
                    fixed_points[i, j] = a.point[fixed_index]
                    moving_points[i, j] = a.point[moving_index]
            return np.array(fixed_points), np.array(moving_points)
        else:
            num_points = len(annotations) // 2
            annotations = annotations[: num_points * 2]
            num_dims = len(annotations[0].point)
            fixed_points = np.zeros((num_points, num_dims))
            moving_points = np.zeros((num_points, num_dims))
            for i, a in enumerate(annotations):
                props = a.props
                if props[1] == 0:
                    fixed_points[props[0]] = a.point
                else:
                    moving_points[props[0]] = a.point

            return np.array(fixed_points), np.array(moving_points)

    def automatically_group_markers(self, s: neuroglancer.ViewerState):
        dimensions = s.dimensions.names
        if self.two_coord_spaces:
            return False
        annotations = s.layers[self.annotations_name].annotations
        if len(annotations) == self.stored_group_number:
            return False
        self.stored_group_number = len(annotations)
        if len(annotations) < 2:
            return False
        for i, a in enumerate(s.layers[self.annotations_name].annotations):
            a.props = [i // 2, i % 2]
        return True

    def update_registered_layers(self, s: neuroglancer.ViewerState):
        if self.affine is not None:
            transform = self.affine.tolist()
            # TODO handle layer being renamed
            for k, v in self.stored_moving_dims.items():
                # TODO not sure if need to handle local channels here
                # keeping code below just in case
                for source in s.layers[k].source:
                    source.transform = neuroglancer.CoordinateSpaceTransform(
                        input_dimensions=v,
                        output_dimensions=change_coord_names(v, "2"),
                        matrix=transform,
                    )
                for source in s.layers[k + "_registered"].source:
                    source.transform = neuroglancer.CoordinateSpaceTransform(
                        input_dimensions=v,
                        output_dimensions=v,
                        matrix=transform,
                    )
            print(self.combine_affine_across_dims(s, self.affine).tolist())
            s.layers[self.annotations_name].source[
                0
            ].transform = neuroglancer.CoordinateSpaceTransform(
                input_dimensions=create_dimensions(s.dimensions),
                output_dimensions=create_dimensions(s.dimensions),
                matrix=self.combine_affine_across_dims(s, self.affine).tolist(),
            )

            # print(s.layers["registered"].source[0].transform.matrix)
            # TODO this is where that mapping needs to happen of affine dims
            # overall this is a bit awkward right now, we need a lot of
            # mapping info which we just don't have
            # right now you can't input it from the command line
            # if s.layers["registered"].source[0].transform is not None:
            #     final_transform = []
            #     layer_transform = s.layers["registered"].source[0].transform
            #     local_channel_indices = [
            #         i
            #         for i, name in enumerate(layer_transform.outputDimensions.names)
            #         if name.endswith(("'", "^", "#"))
            #     ]
            #     num_local_count = 0
            #     for i, name in enumerate(layer_transform.outputDimensions.names):
            #         is_local = i in local_channel_indices
            #         if is_local:
            #             final_transform.append(layer_transform.matrix[i].tolist())
            #             num_local_count += 1
            #         else:
            #             row = transform[i - num_local_count]
            #             # At the indices corresponding to local channels, insert 0s
            #             for j in local_channel_indices:
            #                 row.insert(j, 0)
            #             final_transform.append(row)
            # else:
            #     final_transform = transform
            print("Updated affine transform:", transform)
            print(s.layers["registered"].source[0].transform)

    def estimate_affine(self, s: neuroglancer.ViewerState):
        annotations = s.layers[self.annotations_name].annotations
        if len(annotations) == 0:
            return False

        dim_names = s.dimensions.names
        fixed_points, moving_points = self.split_points_into_pairs(
            annotations, dim_names
        )
        if len(self.stored_points[0]) == len(fixed_points) and len(
            self.stored_points[1]
        ) == len(moving_points):
            if np.all(np.isclose(self.stored_points[0], fixed_points)) and np.all(
                np.isclose(self.stored_points[1], moving_points)
            ):
                return False
        self.affine = fit_model(fixed_points, moving_points)
        self.update_registered_layers(s)

        self._set_status_message(
            "info",
            f"Estimated affine transform with {len(moving_points)} point pairs",
        )
        self.stored_points = [fixed_points, moving_points]
        return True

    def get_registration_info(self):
        info = {}
        with self.viewer.txn() as s:
            annotations = s.layers[self.annotations_name].annotations
            dim_names = s.dimensions.names
            fixed_points, moving_points = self.split_points_into_pairs(
                annotations, dim_names
            )
            transformed_points = self.transform_points_with_affine(moving_points)
            info["fixedPoints"] = fixed_points.tolist()
            info["movingPoints"] = moving_points.tolist()
            if self.affine is not None and transformed_points is not None:
                info["transformedPoints"] = transformed_points.tolist()
                info["affineTransform"] = self.affine.tolist()
        return info

    def dump_info(self, path: str):
        import json

        info = self.get_registration_info()
        with open(path, "w") as f:
            json.dump(info, f, indent=4)

    def get_state(self):
        with self.viewer.txn() as s:
            return s

    def __str__(self):
        return str(self.get_state())

    def _clear_status_messages(self):
        to_pop = []
        for k, v in self.status_timers.items():
            if time() - v > MESSAGE_DURATION:
                to_pop.append(k)
        for k in to_pop:
            with self.viewer.config_state.txn() as s:
                s.status_messages.pop(k, None)
            self.status_timers.pop(k)

    def _set_status_message(self, key: str, message: str):
        with self.viewer.config_state.txn() as s:
            s.status_messages[key] = message
        self.status_timers[key] = time()

    def transform_points_with_affine(self, points: np.ndarray):
        if self.affine is not None:
            return transform_points(self.affine, points)

    def _add_demo_data_to_viewer(self):
        fixed_layer = _create_demo_fixed_image()
        moving_layer = _create_demo_moving_image()

        with self.viewer.txn() as s:
            s.layers["fixed"] = fixed_layer
            s.layers["moving"] = moving_layer


def add_mapping_args(ap: argparse.ArgumentParser):
    ap.add_argument(
        "--annotations-name",
        "-a",
        type=str,
        help="Name of the annotation layer (default is annotations)",
        default="annotation",
        required=False,
    )
    ap.add_argument(
        "--single-coord-space",
        "-s",
        action="store_true",
        help="Use a single coordinate space for both fixed and moving layers (default is two coord spaces)",
        default=False,
        required=False,
    )


def handle_args():
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_state_arguments(ap, required=False)
    neuroglancer.cli.add_server_arguments(ap)
    add_mapping_args(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    return args


if __name__ == "__main__":
    args = handle_args()

    demo = LinearRegistrationWorkflow(args)

    webbrowser.open_new(demo.viewer.get_viewer_url())
