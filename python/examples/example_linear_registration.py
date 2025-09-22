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


# TODO add other types of fits
# Inspired by https://github.com/AllenInstitute/render-python/blob/master/renderapi/transform/leaf/affine_models.py
def affine_fit(fixed_points: np.ndarray, moving_points: np.ndarray):
    # Points are NxD arrays
    assert fixed_points.shape == moving_points.shape
    N = fixed_points.shape[0]
    D = fixed_points.shape[1]
    T = fixed_points

    # Target values (B) is a D * N array
    # Input values (A) is a D * N, (D * (D + 1)) array
    # Output estimation is a (D * (D + 1)) array
    A = np.zeros(((D * N), D * (D + 1)))
    for i in range(N):
        for j in range(D):
            start_index = j * D
            end_index = (j + 1) * D
            A[D * i + j, start_index:end_index] = T[i]
            A[D * i + j, D * D + j] = 1
    B = moving_points.flatten()

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
    affine = np.round(affine, decimals=3)
    return affine


def transform_points(affine: np.ndarray, points: np.ndarray):
    # Apply the current affine transform to the points
    transformed = np.zeros_like(points)
    padded = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
    for i in range(len(points)):
        transformed[i] = affine @ padded[i]
    return transformed


def create_demo_data(size: int | tuple = 60, radius: float = 20):
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


# TODO this should be more intelligent for the copy
# in that case it should take the original layers dimensions
# and actually map the names
# the problem is that right now we can't query those names from python
# since it wraps the state and that info isn't in the state if it is still the
# default information
def create_dimensions(viewer_dims: neuroglancer.CoordinateSpace, indices=None):
    names = viewer_dims.names
    units = viewer_dims.units
    scales = viewer_dims.scales
    if indices is not None:
        names = [viewer_dims.names[i] for i in indices]
        units = [viewer_dims.units[i] for i in indices]
        scales = [viewer_dims.scales[i] for i in indices]

    return neuroglancer.CoordinateSpace(names=names, units=units, scales=scales)


class LinearRegistrationWorkflow:
    def __init__(self, args):
        starting_state = args.state
        self.moving_name = args.moving_name
        self.annotations_name = args.annotations_name
        self.status_timers = {}
        self.stored_points = [[], []]
        self.stored_group_number = -1
        self.affine = None
        self.ready = False
        self.last_updated_print = -1
        self.two_coord_spaces = False
        self.viewer = neuroglancer.Viewer()
        self.viewer.shared_state.add_changed_callback(
            lambda: self.viewer.defer_callback(self.on_state_changed)
        )

        if starting_state is None:
            self.demo_data = create_demo_data()
            self.add_fake_data_to_viewer()
        else:
            self.viewer.set_state(starting_state)

        self._set_status_message(
            "help",
            "Waiting for viewer to initialize with one layer called moving.",
        )

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

    def toggle_registered_visibility(self, _):
        with self.viewer.txn() as s:
            is_registered_visible = s.layers["registered"].visible
            s.layers["registered"].visible = not is_registered_visible

    def add_fake_data_to_viewer(self):
        viewer = self.viewer
        fixed_layer = self.create_demo_fixed_image()
        moving_layer = self.create_demo_moving_image()

        with viewer.txn() as s:
            s.layers["fixed"] = fixed_layer
            s.layers[self.moving_name] = moving_layer

    def setup_viewer_actions(self):
        viewer = self.viewer
        viewer.actions.add(
            "toggle-registered-visibility", self.toggle_registered_visibility
        )

        with viewer.config_state.txn() as s:
            s.input_event_bindings.viewer["keyt"] = "toggle-registered-visibility"

    def is_fixed_image_space_last(self, dim_names):
        first_name = dim_names[0]
        return first_name[-1] in "0123456789"

    def init_registered_transform(self, s: neuroglancer.ViewerState, force=True):
        if not force and s.layers["registered"].source[0].transform is not None:
            return
        # TODO this can fail to solve the no matrix issue if ends up as the identity
        # think I need to change something in neuroglancer python for this instead
        indices = None
        if self.check_for_two_coord_spaces(s.dimensions.names):
            self.coord_spaces = True
            num_dims = len(s.dimensions.names) // 2
            if self.is_fixed_image_space_last(s.dimensions.names):
                indices = list(
                    range(len(s.dimensions.names) - num_dims, len(s.dimensions.names))
                )
            else:
                indices = list(range(num_dims))
        existing_transform = neuroglancer.CoordinateSpaceTransform(
            output_dimensions=create_dimensions(s.dimensions, indices)
        )
        s.layers["registered"].source[0].transform = existing_transform
        print(s.layers["registered"].source[0].transform.matrix)

    @debounce(0.5)
    def check_viewer_ready_and_setup(self):
        with self.viewer.txn() as s:
            if s.dimensions.names == [] or s.layers.index(self.moving_name) == -1:
                return
            s.layers["registered"] = self.create_registered_image()
            s.layers["registered"].visible = False
            self.init_registered_transform(s)
            if s.layers.index(self.annotations_name) == -1:
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
            s.layers[self.moving_name].visible = True
            s.layers[self.annotations_name].tool = "annotatePoint"
            s.selected_layer.layer = self.annotations_name
            s.selected_layer.visible = True

            all_layer_names = [layer.name for layer in s.layers]
            group_1_names = [
                name for name in all_layer_names if name != self.moving_name
            ]
            group_2_names = [name for name in all_layer_names if name != "registered"]
            if isinstance(s.layout, neuroglancer.DataPanelLayout):
                s.layout = neuroglancer.row_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=group_1_names, layout="xy-3d"
                        ),
                        neuroglancer.LayerGroupViewer(
                            layers=group_2_names, layout="xy-3d"
                        ),
                    ]
                )
                s.layout.children[1].position.link = "unlinked"
                s.layout.children[1].crossSectionOrientation.link = "unlinked"
                s.layout.children[1].crossSectionScale.link = "unlinked"
                s.layout.children[1].projectionOrientation.link = "unlinked"
                s.layout.children[1].projectionScale.link = "unlinked"
                # TODO expand to unlink coords and make two coord spaces
            else:
                s.layout.children[0].layers.append("registered")

        self._set_status_message(
            "help",
            "Place markers in pairs, starting with the fixed, and then the moving. The registered layer will automatically update as you add markers. Press 't' to toggle visiblity of the registered layer.",
        )
        self.ready = True
        self.setup_viewer_actions()

    def on_state_changed(self):
        self.viewer.defer_callback(self.update)

    def update(self):
        current_time = time()
        if current_time - self.last_updated_print > 5:
            print(f"Viewer states are successfully syncing at {ctime()}")
            self.last_updated_print = current_time
        if not self.ready:
            self.check_viewer_ready_and_setup()
            return
        if not self.two_coord_spaces:
            self.automatically_group_markers_and_update()
        self.update_affine()
        self._clear_status_messages()

    @debounce(0.25)
    def automatically_group_markers_and_update(self):
        with self.viewer.txn() as s:
            self.automatically_group_markers(s)

    @debounce(1.5)
    def update_affine(self):
        with self.viewer.txn() as s:
            self.estimate_affine(s)

    def create_demo_fixed_image(self):
        return neuroglancer.ImageLayer(
            source=[
                neuroglancer.LayerDataSource(neuroglancer.LocalVolume(self.demo_data))
            ]
        )

    def create_demo_moving_image(self):
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
            self.demo_data,
            matrix=inverse_matrix,
        )
        print("target demo affine", inverse_matrix)
        return neuroglancer.ImageLayer(
            source=[neuroglancer.LayerDataSource(neuroglancer.LocalVolume(transformed))]
        )

    def create_registered_image(self):
        with self.viewer.txn() as s:
            layer = deepcopy(s.layers[self.moving_name])
            layer.name = "registered"
            return layer

    def check_for_two_coord_spaces(self, dim_names):
        # Dims should be exactly double the number of unique names
        if len(dim_names) == 0 or len(dim_names) % 2 != 0:
            return False
        set_of_names = set()
        for name in dim_names:
            # rstrip any number off the end
            stripped_name = name.rstrip("0123456789")
            set_of_names.add(stripped_name)
        return len(set_of_names) * 2 == len(dim_names)

    def split_points_into_pairs(self, annotations, dim_names):
        if len(annotations) == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))
        two_coord_spaces = self.check_for_two_coord_spaces(dim_names)
        if two_coord_spaces:
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
        if self.check_for_two_coord_spaces(dimensions):
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

    def update_registered_layer(self, s: neuroglancer.ViewerState):
        self.init_registered_transform(s, force=False)
        if self.affine is not None:
            print(s.layers["registered"].source[0].transform.matrix)
            transform = self.affine.tolist()
            # TODO this is where that mapping needs to happen of affine dims
            # overall this is a bit awkward right now, we need a lot of
            # mapping info which we just don't have
            # right now you can't input it from the command line
            if s.layers["registered"].source[0].transform is not None:
                final_transform = []
                layer_transform = s.layers["registered"].source[0].transform
                local_channel_indices = [
                    i
                    for i, name in enumerate(layer_transform.outputDimensions.names)
                    if name.endswith(("'", "^", "#"))
                ]
                num_local_count = 0
                for i, name in enumerate(layer_transform.outputDimensions.names):
                    is_local = i in local_channel_indices
                    if is_local:
                        final_transform.append(layer_transform.matrix[i].tolist())
                        num_local_count += 1
                    else:
                        row = transform[i - num_local_count]
                        # At the indices corresponding to local channels, insert 0s
                        for j in local_channel_indices:
                            row.insert(j, 0)
                        final_transform.append(row)
            else:
                final_transform = transform
            print("Updated affine transform:", final_transform)
            print(s.layers["registered"].source[0].transform)
            print(final_transform)
            s.layers["registered"].source[0].transform.matrix = final_transform
            print(s.layers["registered"].source[0].transform)

    def estimate_affine(self, s: neuroglancer.ViewerState):
        annotations = s.layers[self.annotations_name].annotations
        if len(annotations) < 1:
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
        self.affine = affine_fit(moving_points, fixed_points)
        self.update_registered_layer(s)

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
        "--moving-name",
        "-m",
        type=str,
        help="Name of the moving image layer (default is moving)",
        default="moving",
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
