"""Serves a Neuroglancer image layer as OME-Zarr 0.4, read through the viewer.

Any OME-Zarr reader, such as vizarr, can then open a format that only
Neuroglancer reads, such as JPEG-encoded precomputed:

    python zarr_gateway.py --port 9000

Open the printed Neuroglancer link and keep its tab open, then open the
printed vizarr link. Chrome asks vizarr for permission to reach localhost;
allow it.
"""

import argparse
import collections.abc
import http.server
import json
import select
import socket
import urllib.parse

import neuroglancer
import neuroglancer.cli
import numpy as np
import tensorstore as ts

ZARR_CHUNK_SHAPE = (1, 256, 256)


class ZarrGateway:
    def __init__(self, viewer: neuroglancer.Viewer, layer: str):
        info = viewer.volume_info(layer).result()
        names = list(reversed(info.dimensions.names))
        if names != ["z", "y", "x"] or set(info.dimensions.units) != {"m"}:
            raise ValueError(
                f"Expected spatial dimensions x, y, z in meters, "
                f"not {info.dimensions.names} in {info.dimensions.units}"
            )
        self.layer = layer
        cache = ts.Context({"cache_pool": {"total_bytes_limit": 2**30}})
        self.levels: list[ts.TensorStore] = [
            viewer.volume(
                layer,
                dimensions=neuroglancer.CoordinateSpace(
                    names=info.dimensions.names,
                    units=info.dimensions.units,
                    scales=scales,
                ),
                context=cache,
            )
            .result()
            .T
            for scales in info.available_scales
        ]

    def read(
        self, path: str, client_disconnected: collections.abc.Callable[[], bool]
    ) -> bytes | None:
        parts = path.strip("/").split("/")
        if parts == [".zgroup"]:
            return json.dumps({"zarr_format": 2}).encode()
        if parts == [".zattrs"]:
            return json.dumps(self._multiscales()).encode()
        if not parts[0].isdigit() or int(parts[0]) >= len(self.levels):
            return None
        level = self.levels[int(parts[0])]
        if parts[1:] == [".zarray"]:
            return json.dumps(self._array_metadata(level)).encode()
        if len(parts) == 4 and all(part.isdigit() for part in parts[1:]):
            return self._chunk(
                level, [int(part) for part in parts[1:]], client_disconnected
            )
        return None

    def serve(self, port: int) -> None:
        gateway = self

        class GatewayRequestHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = gateway.read(
                    urllib.parse.urlsplit(self.path).path, self._client_disconnected
                )
                if self._client_disconnected():
                    return
                self.send_response(404 if body is None else 200)
                self._send_cors_headers()
                if body is not None:
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body is not None:
                    self.wfile.write(body)

            def do_OPTIONS(self):
                self.send_response(204)
                self._send_cors_headers()
                self.end_headers()

            def _client_disconnected(self) -> bool:
                readable, _, _ = select.select([self.connection], [], [], 0)
                try:
                    return bool(readable) and not self.connection.recv(
                        1, socket.MSG_PEEK
                    )
                except ConnectionResetError:
                    return True

            def _send_cors_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "*")

        http.server.ThreadingHTTPServer(
            ("", port), GatewayRequestHandler
        ).serve_forever()

    def _multiscales(self):
        base_scale = _scale_in_nanometers(self.levels[0])
        datasets = []
        for level_index, level in enumerate(self.levels):
            scale = _scale_in_nanometers(level)
            # OME-Zarr places the voxel centre at its index, not the corner.
            translation = [
                start * size + (size - base_size) / 2
                for start, size, base_size in zip(
                    level.domain.inclusive_min, scale, base_scale
                )
            ]
            datasets.append(
                {
                    "path": str(level_index),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale},
                        {"type": "translation", "translation": translation},
                    ],
                }
            )
        return {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": self.layer,
                    "axes": [
                        {"name": name, "type": "space", "unit": "nanometer"}
                        for name in ["z", "y", "x"]
                    ],
                    "datasets": datasets,
                }
            ]
        }

    def _array_metadata(self, level: ts.TensorStore):
        return {
            "zarr_format": 2,
            "shape": list(level.shape),
            "chunks": list(ZARR_CHUNK_SHAPE),
            "dtype": level.dtype.numpy_dtype.str,
            "compressor": None,
            "filters": None,
            "fill_value": 0,
            "order": "C",
            "dimension_separator": "/",
        }

    def _chunk(
        self,
        level: ts.TensorStore,
        chunk_position: list[int],
        client_disconnected: collections.abc.Callable[[], bool],
    ) -> bytes | None:
        level = level.translate_to[0]
        origin = [
            position * size for position, size in zip(chunk_position, ZARR_CHUNK_SHAPE)
        ]
        if any(start >= extent for start, extent in zip(origin, level.shape)):
            return None
        box = tuple(
            slice(start, min(start + size, extent))
            for start, size, extent in zip(origin, ZARR_CHUNK_SHAPE, level.shape)
        )
        read = level[box].read()
        while True:
            try:
                voxels = read.result(timeout=0.2)
                break
            except TimeoutError:
                if client_disconnected():
                    read.cancel()
                    return None
        chunk = np.zeros(ZARR_CHUNK_SHAPE, dtype=level.dtype.numpy_dtype)
        chunk[tuple(slice(0, extent) for extent in voxels.shape)] = voxels
        return chunk.tobytes()


def _scale_in_nanometers(level: ts.TensorStore) -> list[float]:
    return [unit.multiplier * 1e9 for unit in level.dimension_units]


def main():
    parser = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(parser)
    parser.add_argument(
        "--source",
        default="precomputed://gs://neuroglancer-public-data/flyem_fib-25/image",
    )
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    neuroglancer.cli.handle_server_arguments(args)

    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.layers.append(
            name="image",
            layer=neuroglancer.ImageLayer(source=args.source),
            visible=False,
        )
    print(f"Open Neuroglancer and keep its tab open: {viewer}")
    gateway = ZarrGateway(viewer, "image")
    print(
        "Open vizarr: "
        f"https://hms-dbmi.github.io/vizarr/?source=http://localhost:{args.port}/"
    )
    gateway.serve(args.port)


if __name__ == "__main__":
    main()
