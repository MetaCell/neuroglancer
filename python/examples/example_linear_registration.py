import argparse
import webbrowser
from time import time

import scipy.ndimage
import numpy as np

import neuroglancer
import neuroglancer.cli


def create_demo_data(size=(64, 64, 64), radius=20):
    import numpy as np

    data = np.zeros(size, dtype=np.uint8)
    zz, yy, xx = np.indices(data.shape)
    center = np.array(data.shape) / 2
    sphere_mask = (xx - center[2]) ** 2 + (yy - center[1]) ** 2 + (
        zz - center[0]
    ) ** 2 < radius**2
    data[sphere_mask] = 255
    return data


def create_dimensions():
    return neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
    )


class LinearRegistrationWorkflow:
    def __init__(self, template_url, source_url):
        self.template_url = template_url
        self.source_url = source_url

        if template_url is None or source_url is None:
            self.demo_data = create_demo_data()

        self.status_timers = {}
        self.setup_viewer()
        self.last_run_points = 0

    def _clear_status_messages(self):
        to_pop = []
        for k, v in self.status_timers.items():
            if time() - v > 5:
                to_pop.append(k)
        for k in to_pop:
            with self.viewer.config_state.txn() as s:
                s.status_messages.pop(k, None)
            self.status_timers.pop(k)

    def _set_status_message(self, key, message):
        with self.viewer.config_state.txn() as s:
            s.status_messages[key] = message
        self.status_timers[key] = time()

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
                annotation_color="#00FF00",
            )
            # TODO set these to be in 3D layout
            # TODO unlink any controls that should be unlinked
            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(
                        layers=["template", "registered", "markers"]
                    ),
                    neuroglancer.LayerGroupViewer(
                        layers=["source", "registered", "markers"]
                    ),
                ]
            )
            s.layers["markers"].tool = "annotatePoint"
            s.selected_layer.layer = "markers"
            s.selected_layer.visible = True

        self._set_status_message(
            "help",
            "Place markers in pairs, starting with the template, and then the source. The registered layer will automatically update as you add markers.",
        )

        self.viewer.shared_state.add_changed_callback(
            lambda: self.viewer.defer_callback(self.on_state_changed)
        )

    def on_state_changed(self):
        self.viewer.defer_callback(self.update)

    def update(self):
        print("Updating")
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

    # TODO probably need to be a little more careful about the size of the T matrix
    # based on the number of input dims
    def create_source_image(self, registration_matrix=None):
        if registration_matrix is None:
            registration_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        if self.source_url is None:
            # TODO might be helpful to randomize this and check how close after registration
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
            return neuroglancer.ImageLayer(
                source=[
                    neuroglancer.LayerDataSource(
                        neuroglancer.LocalVolume(
                            transformed, dimensions=create_dimensions()
                        ),
                        transform=neuroglancer.CoordinateSpaceTransform(
                            output_dimensions=create_dimensions(),
                            matrix=registration_matrix,
                        ),
                    )
                ]
            )
        else:
            return neuroglancer.ImageLayer(
                source=self.source_url,
                transform=neuroglancer.CoordinateSpaceTransform(
                    matrix=registration_matrix
                ),
            )

    def create_registered_image(self, registration_matrix=None):
        return self.create_source_image(registration_matrix=registration_matrix)

    def estimate_affine(self, s):
        annotations = s.layers["markers"].annotations
        # TODO expand this to estimate different types of transforms
        # Depending on the number of points
        # For now, just ignore non pairs
        annotations = annotations[: (len(annotations) // 2) * 2]
        if len(annotations) < 2:
            return False

        # TODO allow a different way to group, such as by description
        # TODO color points differently based on whether template or source
        # in the shader
        print(len(annotations) // 2, self.last_run_points)
        if len(annotations) // 2 == self.last_run_points:
            return False

        template_points = []
        source_points = []
        for i, a in enumerate(annotations):
            if i % 2 == 0:
                template_points.append(a.point)
            else:
                source_points.append(a.point)

        import numpy as np

        template_points = np.array(template_points)
        source_points = np.array(source_points)

        # Estimate affine transform using least squares, for now using
        # https://stackoverflow.com/questions/20546182/how-to-perform-coordinates-affine-transformation-using-python-part-2
        # but can replace later

        n = template_points.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        X = pad(template_points)
        Y = pad(source_points)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        A, res, rank, sd = np.linalg.lstsq(X, Y)
        # Zero out really small values on A
        A[np.abs(A) < 1e-4] = 0
        # Round all other values to 4 decimal places
        A = np.round(A, 4)

        transform = lambda x: unpad(np.dot(pad(x), A))

        transformed = transform(source_points)
        print(f"Transformed points: {transformed}")

        # Set the transformation on the layer that is being registered
        # Something seems to go wrong with the state updates once this happens
        # s.layers["registered"].source[0].transform.matrix = A.T[:3]
        # Because of this, trying to replace the whole layer to see if that helps
        # TODO in theory should be able to just update the matrix,
        # but if cannot, can replace whole layer and restore settings
        old_visible = s.layers["registered"].visible
        s.layers["registered"] = self.create_registered_image(A[:3])
        s.layers["registered"].visible = old_visible

        self._set_status_message(
            ("info"), f"Estimated affine transform with {n} point pairs"
        )
        # TODO actually want to check if the number of points or the points
        # themselves have changed
        self.last_run_points = n
        return True


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


def main():
    args = handle_args()

    demo = LinearRegistrationWorkflow(
        template_url=args.template,
        source_url=args.source,
    )

    webbrowser.open_new(demo.viewer.get_viewer_url())


if __name__ == "__main__":
    main()
