import argparse
import threading
import webbrowser
from time import ctime, time
from typing import Optional

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


# TODO can we avoid calling this at all? Can the layers just be created and dims inferred?
def create_dimensions():
    if NUM_DEMO_DIMS == 2:
        return neuroglancer.CoordinateSpace(names=["x", "y"], units="nm", scales=[1, 1])
    return neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
    )


class LinearRegistrationWorkflow:
    def __init__(self, fixed_url: str, moving_url: str):
        self.fixed_url = fixed_url
        self.moving_url = moving_url
        self.status_timers = {}
        self.stored_points = [[], []]
        self.stored_group_number = -1
        self.affine = None

        if fixed_url is None or moving_url is None:
            self.demo_data = create_demo_data()

        self.setup_viewer()
        self.last_updated_print = -1

    def __str__(self):
        with self.viewer.txn() as s:
            return str(s)

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
            s.layers["registered"].visible = not s.layers["registered"].visible
            s.layers["fixed"].visible = not s.layers["registered"].visible
            s.layers["markers"].visible = not s.layers["registered"].visible
            s.layers["mappedMarkers"].visible = s.layers["registered"].visible

    def setup_viewer(self):
        self.viewer = viewer = neuroglancer.Viewer()
        fixed_layer = self.create_fixed_image()
        moving_layer = self.create_moving_image()
        registered_layer = self.create_registered_image()

        with viewer.txn() as s:
            s.layers["fixed"] = fixed_layer
            s.layers["moving"] = moving_layer
            s.layers["registered"] = registered_layer
            s.layers["registered"].visible = False
            s.layers["markers"] = neuroglancer.LocalAnnotationLayer(
                dimensions=create_dimensions(),
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
            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(
                        layers=["fixed", "registered", "markers"], layout="xy-3d"
                    ),
                    neuroglancer.LayerGroupViewer(
                        layers=["moving", "markers"], layout="xy-3d"
                    ),
                ]
            )
            s.layout.children[1].position.link = "unlinked"
            s.layout.children[1].crossSectionOrientation.link = "unlinked"
            s.layout.children[1].crossSectionScale.link = "unlinked"
            s.layout.children[1].projectionOrientation.link = "unlinked"
            s.layout.children[1].projectionScale.link = "unlinked"
            s.layers["markers"].tool = "annotatePoint"
            s.selected_layer.layer = "markers"
            s.selected_layer.visible = True

        viewer.actions.add(
            "toggle-registered-visibility", self.toggle_registered_visibility
        )

        with viewer.config_state.txn() as s:
            s.input_event_bindings.viewer["keyt"] = "toggle-registered-visibility"

        self._set_status_message(
            "help",
            "Place markers in pairs, starting with the fixed, and then the moving. The registered layer will automatically update as you add markers. Press 't' to toggle between viewing the fixed and registered layers.",
        )

        self.viewer.shared_state.add_changed_callback(
            lambda: self.viewer.defer_callback(self.on_state_changed)
        )

    def on_state_changed(self):
        self.viewer.defer_callback(self.update)

    def update(self):
        current_time = time()
        if current_time - self.last_updated_print > 5:
            print(f"Viewer states are successfully syncing at {ctime()}")
            self.last_updated_print = current_time
        # with self.viewer.txn() as s:
        # self.automatically_group_markers(s)
        # self.estimate_affine(s)
        self._clear_status_messages()
        # TODO for some reason I need to keep the layer change for states
        self.update_affine()

    @debounce(1.0)
    def update_affine(self):
        with self.viewer.txn() as s:
            self.estimate_affine(s)

    def create_fixed_image(self):
        if self.fixed_url is None:
            return neuroglancer.ImageLayer(
                source=[
                    neuroglancer.LayerDataSource(
                        neuroglancer.LocalVolume(
                            self.demo_data, dimensions=create_dimensions()
                        )
                    )
                ]
            )
        else:
            return neuroglancer.ImageLayer(source=self.fixed_url)

    def create_moving_image(self, registration_matrix: list | np.ndarray | None = None):
        transform_kwargs = {}
        if registration_matrix is not None:
            transform_kwargs["matrix"] = registration_matrix
        if self.moving_url is None:
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
                source=[
                    neuroglancer.LayerDataSource(
                        neuroglancer.LocalVolume(
                            transformed, dimensions=create_dimensions()
                        ),
                        transform=neuroglancer.CoordinateSpaceTransform(
                            output_dimensions=create_dimensions(),
                            **transform_kwargs,
                        ),
                    )
                ]
            )
        else:
            return neuroglancer.ImageLayer(
                source=self.moving_url,
                transform=neuroglancer.CoordinateSpaceTransform(**transform_kwargs),
            )

    def create_registered_image(self):
        return self.create_moving_image(registration_matrix=self.affine)

    def split_points_into_pairs(self, annotations):
        fixed_points = []
        moving_points = []
        for i, a in enumerate(annotations):
            props = a.props
            if props[1] == 0:
                fixed_points.append(a.point)
            else:
                moving_points.append(a.point)
        # If the moving points is not evenly split, instead split differently
        if len(moving_points) != len(fixed_points):
            fixed_points = []
            moving_points = []
            for i, a in enumerate(annotations):
                if i % 2 == 0:
                    fixed_points.append(a.point)
                else:
                    moving_points.append(a.point)
        return np.array(fixed_points), np.array(moving_points)

    def automatically_group_markers(self, s: neuroglancer.ViewerState):
        annotations = s.layers["markers"].annotations
        if len(annotations) < 2:
            return False
        if len(annotations) == self.stored_group_number:
            return False
        print("Updating marker groups")
        for i, a in enumerate(s.layers["markers"].annotations):
            a.props = [i // 2, i % 2]
            print(a.props, i // 2, i % 2)
        self.stored_group_number = len(annotations)
        return True

    def estimate_affine(self, s: neuroglancer.ViewerState):
        annotations = s.layers["markers"].annotations
        if len(annotations) < 2:
            return False

        # Ignore annotations not part of a pair
        annotations = annotations[: (len(annotations) // 2) * 2]
        fixed_points, moving_points = self.split_points_into_pairs(annotations)
        if len(self.stored_points[0]) == len(fixed_points) and len(
            self.stored_points[1]
        ) == len(moving_points):
            if np.all(np.isclose(self.stored_points[0], fixed_points)) and np.all(
                np.isclose(self.stored_points[1], moving_points)
            ):
                return False
        else:
            # TODO instead of directly doing the update, debounce it and only do
            # it after a short delay if no other updates
            for i, a in enumerate(s.layers["markers"].annotations):
                a.props = [i // 2, i % 2]
        self.affine = affine_fit(moving_points, fixed_points)

        # Set the transformation on the layer that is being registered
        # Something seems to go wrong with the state updates once this happens
        # s.layers["registered"].source[0].transform.matrix = A.T[:3]
        # Because of this, trying to replace the whole layer to see if that helps
        # TODO in theory should be able to just update the matrix,
        # but if cannot, can replace whole layer and restore settings
        old_visible = s.layers["registered"].visible
        s.layers["registered"] = self.create_registered_image()
        s.layers["registered"].visible = old_visible

        self._set_status_message(
            "info",
            f"Estimated affine transform with {len(annotations) // 2} point pairs",
        )
        self.stored_points = [fixed_points, moving_points]
        return True

    def get_registration_info(self):
        info = {}
        with self.viewer.txn() as s:
            annotations = s.layers["markers"].annotations
            fixed_points, moving_points = self.split_points_into_pairs(annotations)
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


def handle_args():
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    ap.add_argument(
        "--fixed",
        "-f",
        type=str,
        help="Source URL for the fixed image",
    )
    ap.add_argument(
        "--moving",
        "-m",
        type=str,
        help="Source URL for the image to be registered",
    )
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    return args


if __name__ == "__main__":
    args = handle_args()

    demo = LinearRegistrationWorkflow(
        fixed_url=args.fixed,
        moving_url=args.moving,
    )

    webbrowser.open_new(demo.viewer.get_viewer_url())
