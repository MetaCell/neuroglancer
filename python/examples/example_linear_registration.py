import argparse
import webbrowser
from time import time

import neuroglancer
import neuroglancer.cli
import numpy as np
import scipy.ndimage

MESSAGE_DURATION = 5  # seconds
NUM_DEMO_DIMS = 2  # Currently can be 2D or 3D

MARKERS_SHADER = """
#uicontrol vec3 templatePointColor color(default="#00FF00")
#uicontrol vec3 sourcePointColor color(default="#0000FF")
#uicontrol float pointSize slider(min=1, max=16, default=6)
void main() {
    if (int(prop_index()) % 2 == 0) {
        setColor(templatePointColor);
    } else {
        setColor(sourcePointColor);
    }
    setPointMarkerSize(pointSize);
}
"""


def affine_fit(template_points: np.ndarray, target_points: np.ndarray):
    # Source points and target points are NxD arrays
    assert template_points.shape == target_points.shape
    N = template_points.shape[0]
    D = template_points.shape[1]
    T = template_points

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
    B = target_points.T.flatten()

    print(A.shape, B.shape)
    print(A, B)
    # The estimated affine transform params will be flattened
    # and there will be D * (D + 1) of them
    # Format is x1, x2, ..., b1, b2, ...
    tvec, res, rank, sd = np.linalg.lstsq(A, B)
    print(A, target_points, tvec)
    # Put the flattened version back into the matrix
    affine = np.zeros((D, D + 1))
    for i in range(D):
        start_index = i * D
        end_index = start_index + D
        affine[i, :D] = tvec[start_index:end_index]
        affine[i, -1] = tvec[D * D + i]

    # Round to close decimal
    affine = np.round(affine, decimals=2)
    print(affine)
    return affine


def create_demo_data(size: int | tuple = 60, radius: float = 20):
    import numpy as np

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


def create_dimensions():
    if NUM_DEMO_DIMS == 2:
        return neuroglancer.CoordinateSpace(names=["x", "y"], units="nm", scales=[1, 1])
    return neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
    )


def create_identity_matrix(num_dims: int):
    id_list = [[int(i == j) for j in range(num_dims + 1)] for i in range(num_dims)]
    return np.array(id_list)


class LinearRegistrationWorkflow:
    def __init__(self, template_url: str, source_url: str):
        self.template_url = template_url
        self.source_url = source_url
        self.status_timers = {}
        self.stored_points = [[], []]
        # Will be an Nx(N+1) matrix for N input dimensions
        self.affine = None

        if template_url is None or source_url is None:
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

    def transform_points(self, points: np.ndarray):
        # Apply the current affine transform to the points
        transformed = np.zeros_like(points)
        padded = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
        for i in range(len(points)):
            transformed[i] = self.affine @ padded[i]
        return transformed

    def toggle_registered_visibility(self, _):
        with self.viewer.txn() as s:
            s.layers["registered"].visible = not s.layers["registered"].visible
            s.layers["template"].visible = not s.layers["registered"].visible

    def setup_viewer(self):
        self.viewer = viewer = neuroglancer.Viewer()
        source_layer = self.create_source_image()
        template_layer = self.create_template_image()
        registered_layer = self.create_registered_image()

        with viewer.txn() as s:
            s.layers["template"] = template_layer
            s.layers["source"] = source_layer
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
                        enum_labels=["template", "source"],
                        enum_values=[0, 1],
                    ),
                    neuroglancer.AnnotationPropertySpec(
                        id="index",
                        type="uint32",
                        default=0,
                    ),
                ],
                shader=MARKERS_SHADER,
            )
            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(
                        layers=["template", "registered", "markers"], layout="xy-3d"
                    ),
                    neuroglancer.LayerGroupViewer(
                        layers=["source", "markers"], layout="xy-3d"
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
            "Place markers in pairs, starting with the template, and then the source. The registered layer will automatically update as you add markers. Press 't' to toggle between viewing the template and registered layers.",
        )

        self.viewer.shared_state.add_changed_callback(
            lambda: self.viewer.defer_callback(self.on_state_changed)
        )

    def on_state_changed(self):
        self.viewer.defer_callback(self.update)

    def update(self):
        current_time = time()
        if current_time - self.last_updated_print > 1:
            # TODO format the time nicely in the print
            print(f"Viewer states are successfully syncing at {current_time}")
            self.last_updated_print = current_time
        with self.viewer.txn() as s:
            self.estimate_affine(s)
        self._clear_status_messages()

    def create_template_image(self):
        if self.template_url is None:
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
            return neuroglancer.ImageLayer(source=self.template_url)

    def create_source_image(self, registration_matrix: list | np.ndarray | None = None):
        transform_kwargs = {}
        if registration_matrix is not None:
            transform_kwargs["matrix"] = registration_matrix
        if self.source_url is None:
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
            print(inverse_matrix)
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
                source=self.source_url,
                transform=neuroglancer.CoordinateSpaceTransform(**transform_kwargs),
            )

    def create_registered_image(self):
        return self.create_source_image(registration_matrix=self.affine)

    def split_points_into_pairs(self, annotations):
        # TODO allow a different way to group. Right now the order informs
        # the properties
        # But ideally we'd like to allow the other way around as well
        # This needs to be reworked just a bit to allow that
        # As a first step for that, this should inspect the properties
        # and group based on that instead of this grouping
        template_points = []
        source_points = []
        for i, a in enumerate(annotations):
            if i % 2 == 0:
                template_points.append(a.point)
            else:
                source_points.append(a.point)
        return np.array(template_points), np.array(source_points)

    def estimate_affine(self, s: neuroglancer.ViewerState):
        # TODO do we need to throttle this update or make it manually triggered?
        # While updating by moving a point, this can break right now
        annotations = s.layers["markers"].annotations
        if len(annotations) < 2:
            return False
        # TODO expand this to estimate different types of transforms
        # Depending on the number of points

        # Ignore annotations not part of a pair
        annotations = annotations[: (len(annotations) // 2) * 2]
        template_points, source_points = self.split_points_into_pairs(annotations)
        if len(self.stored_points[0]) == len(template_points) and len(
            self.stored_points[1]
        ) == len(source_points):
            if np.all(np.isclose(self.stored_points[0], template_points)) and np.all(
                np.isclose(self.stored_points[1], source_points)
            ):
                return False
        else:
            # TODO instead of directly doing the update, debounce it and only do
            # it after a short delay if no other updates
            for i, a in enumerate(s.layers["markers"].annotations):
                a.props = [i // 2, i % 2, i]

        template_points, source_points = self.split_points_into_pairs(annotations)

        # Estimate transform
        self.affine = affine_fit(template_points, source_points)

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
        print("Estimated points are", self.transform_points(source_points))
        print("Original points are", template_points)
        self.stored_points = [template_points, source_points]
        return True

    def get_registration_info(self):
        info = {}
        with self.viewer.txn() as s:
            annotations = s.layers["markers"].annotations
            template_points, source_points = self.split_points_into_pairs(annotations)
            # transformed_points = self.transform_template_points(template_points)
            info["template"] = template_points.tolist()
            info["source"] = source_points.tolist()
            # info["transformed_template"] = transformed_points.tolist()
            info["transform"] = self.affine.tolist()
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
        "--template",
        type=str,
        help="Source URL for the template image",
    )
    ap.add_argument(
        "--source",
        type=str,
        help="Source URL for the image to be registered",
    )
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    return args


if __name__ == "__main__":
    args = handle_args()

    demo = LinearRegistrationWorkflow(
        template_url=args.template,
        source_url=args.source,
    )

    webbrowser.open_new(demo.viewer.get_viewer_url())
