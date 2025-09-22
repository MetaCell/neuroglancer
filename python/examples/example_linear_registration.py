import argparse
import threading
import webbrowser
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


def create_dimensions(viewer_dims: neuroglancer.CoordinateSpace):
    return neuroglancer.CoordinateSpace(
        names=viewer_dims.names, units=viewer_dims.units, scales=viewer_dims.scales
    )


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
            "Waiting for viewer to initialize with one layer called fixed and one layer called moving.",
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

    # TODO change this to replace the transform matrix directly?
    # so then only need one layer
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

    def setup_viewer(self):
        viewer = self.viewer
        viewer.actions.add(
            "toggle-registered-visibility", self.toggle_registered_visibility
        )

        with viewer.config_state.txn() as s:
            s.input_event_bindings.viewer["keyt"] = "toggle-registered-visibility"

    @debounce(0.5)
    def post_setup_viewer(self):
        # TODO why do we need the moving? In theory only at setup?
        with self.viewer.txn() as s:
            if s.dimensions.names == [] or s.layers.index(self.moving_name) == -1:
                return
            # registered_layer = self.create_registered_image()
            # TODO might be able to use deepcopy to avoid this akwardness
            # of rename
            s.layers[self.moving_name + "1"] = self.create_registered_image()
            s.layers[self.moving_name + "1"].name = "registered"
            s.layers["registered"].visible = False
            if s.layers.index("registered") == -1:
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
            # TODO in two coord space set the main one for group 2
            # TODO could load a link with layout already setup
            # in which case should just add registered to group 1
            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(layers=group_1_names, layout="xy-3d"),
                    neuroglancer.LayerGroupViewer(layers=group_2_names, layout="xy-3d"),
                ]
            )
            s.layout.children[1].position.link = "unlinked"
            s.layout.children[1].crossSectionOrientation.link = "unlinked"
            s.layout.children[1].crossSectionScale.link = "unlinked"
            s.layout.children[1].projectionOrientation.link = "unlinked"
            s.layout.children[1].projectionScale.link = "unlinked"

        self._set_status_message(
            "help",
            "Place markers in pairs, starting with the fixed, and then the moving. The registered layer will automatically update as you add markers. Press 't' to toggle between viewing the fixed and registered layers.",
        )
        self.ready = True
        self.setup_viewer()

    def on_state_changed(self):
        self.viewer.defer_callback(self.update)

    def update(self):
        current_time = time()
        if current_time - self.last_updated_print > 5:
            print(f"Viewer states are successfully syncing at {ctime()}")
            self.last_updated_print = current_time
        if not self.ready:
            self.post_setup_viewer()
            return
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
            return s.layers[self.moving_name]

    # TODO this check could maybe be more robust
    def check_for_two_coord_spaces(self, dim_names):
        set_of_names = set()
        for name in dim_names:
            # rstrip any number off the end
            stripped_name = name.rstrip("0123456789")
            set_of_names.add(stripped_name)
        return len(set_of_names) * 2 == len(dim_names)

    def split_points_into_pairs(self, annotations, dim_names):
        two_coord_spaces = self.check_for_two_coord_spaces(dim_names)
        # TODO need some way to indicate which coord space is which
        if two_coord_spaces:
            num_points = len(annotations)
            num_dims = len(annotations[0].point) // 2
            fixed_points = np.zeros((num_points, num_dims))
            moving_points = np.zeros((num_points, num_dims))
            for i, a in enumerate(annotations):
                for j in range(num_dims):
                    fixed_points[i, j] = a.point[j + num_dims]
                    moving_points[i, j] = a.point[j]
            return np.array(fixed_points), np.array(moving_points)
        else:
            num_points = len(annotations) // 2
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

    # TODO can disable this check completely if using two coord spaces
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
        # TODO might need to add some mechanism to neuroglancer to do this
        # this seems off that we can't directly set the transform matrix
        # without first creating a new transform object
        existing_transform = s.layers["registered"].source[0].transform
        if existing_transform is None:
            existing_transform = neuroglancer.CoordinateSpaceTransform(
                output_dimensions=create_dimensions(s.dimensions)
            )
            s.layers["registered"].source[0].transform = existing_transform
        if self.affine is not None:
            transform = self.affine.tolist()
            # TODO this is where that mapping needs to happen of affine dims
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
            # TODO for some reason with global dims not matching local dims
            # nothing happens at this step
            # TODO if global dims don't match local dims then the
            # tranform matrix doesn't match properly
            # the easiest to fix this would be to make the local dims of the
            # markers be the same as the local dims of the image layers
            # but we don't seem to always be able to access that info
            # we could try to check similar to the above if a transform
            # has been setup on viewer init and use that to get the
            # names of the local dims which might be easier overall
            # than trying to fiddle with these dims
            # Worst comes to worst you could ask for a copy of the layer by the user
            # with the dims if command line not specified
            # but command line could spec the markers layer dims
            # Try https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B6.500000000000001e-7%2C%22m%22%5D%2C%22y%22:%5B6.500000000000001e-7%2C%22m%22%5D%2C%22z%22:%5B0.00003%2C%22m%22%5D%7D%2C%22position%22:%5B14231.224609375%2C30510.12109375%2C0%5D%2C%22crossSectionScale%22:121.51041751873487%2C%22projectionScale%22:131072%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22s3://allen-genetic-tools/epifluorescence/1383646325/ome_zarr_conversion/1383646325.zarr/%7Czarr2:%22%2C%22localDimensions%22:%7B%22c%27%22:%5B1%2C%22%22%5D%7D%2C%22localPosition%22:%5B1%5D%2C%22tab%22:%22source%22%2C%22name%22:%221383646325.zarr%22%7D%5D%2C%22selectedLayer%22:%7B%22size%22:379%2C%22visible%22:true%2C%22layer%22:%221383646325.zarr%22%7D%2C%22layout%22:%224panel-alt%22%7D for this
            print(s.layers["registered"].source[0].transform)
            print(final_transform)
            s.layers["registered"].source[0].transform.matrix = final_transform
            print(s.layers["registered"].source[0].transform)

    def estimate_affine(self, s: neuroglancer.ViewerState):
        annotations = s.layers[self.annotations_name].annotations
        if len(annotations) < 1:
            return False

        dim_names = s.dimensions.names
        # TODO likely broken right now for non pairs non two coord spaces
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


# TODO need to add some way to handle mapping the output affine
# For e.g. the points can be in XYZ order, but the image dims
# are CZYX order
# TODO allow input arg of layer name to be the moving and the fixed
# TODO don't actually need fixed in theory?
# TODO same for markers / annotations name
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
