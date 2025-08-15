import argparse

import neuroglancer
import neuroglancer.cli
import numpy as np

def add_example_layers(state):
    state.dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[10, 10, 10]
    )
    state.layers.append(
        name="example_layer",
        layer=neuroglancer.LocalVolume(
            data=np.random.rand(10, 10, 10).astype(np.float32),
            dimensions=state.dimensions,
        ),
    )
    return state.layers[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        add_example_layers(s)
        s.layers[0].annotations_accordion["annotationsExpanded"] = False
        s.layers[0].annotations_accordion["relatedSegmentExpanded"] = True
        s.layers[0].rendering_accordion["sliceExpanded"] = True       
        s.layers[0].rendering_accordion["shaderExpanded"] = False       
        s.layers[0].source_accordion["sourceExpanded"] = False      

    print(viewer)