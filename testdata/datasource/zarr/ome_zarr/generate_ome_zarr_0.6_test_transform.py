#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ngff-zarr[tensorstore]",
#     "numpy",
#     "tensorstore",
#     "scipy",
# ]
# ///
"""Generate test data for verifying OME-ZARR 0.6 transformations."""

import numpy as np
import json
import ngff_zarr as nz
import tensorstore as ts
from pathlib import Path
import shutil

THIS_DIR = Path(__file__).parent

def create_test_pattern(shape=(16, 16, 16)):
    """Create an asymmetrical test pattern - an L-shaped figure visible in X-Y top view."""
    data = np.zeros(shape, dtype=np.uint16)
    # Create an L-shape that's clearly asymmetrical when viewed from top (X-Y plane)
    # The L extends through multiple Z layers so it's visible in any Z slice
    
    # Vertical bar of the L: extends in +Y direction
    # Position: x=[6:8], y=[4:10], z=[5:11]
    data[5:11, 4:10, 6:8] = 1000
    
    # Horizontal bar of the L: extends in +X direction from bottom of vertical bar
    # Position: x=[8:14], y=[4:6], z=[5:11]
    data[5:11, 4:6, 8:14] = 1000
    
    return data

def create_ome_zarr_0_5(path, data, name="test"):
    """Create an OME-ZARR 0.5 file using ngff_zarr."""
    # Convert numpy array to ngff_zarr image
    image = nz.to_ngff_image(
        data,
        dims=["z", "y", "x"],
        scale={"z": 1, "y": 1, "x": 1},
        axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
        name=name,
    )
    
    # Create multiscales (just single scale for simplicity)
    multiscales = nz.to_multiscales(image)
    
    # Remove existing directory
    shutil.rmtree(path, ignore_errors=True)
    
    # Write to disk using ngff_zarr
    nz.to_ngff_zarr(
        str(path),
        multiscales,
        use_tensorstore=True,
        version="0.5",
    )

def apply_transform_to_data(data, transform_matrix):
    """Apply transformation to data by resampling.
    
    For each output position, we compute the inverse transform to find
    the input position to sample from.
    """
    from scipy.ndimage import map_coordinates
    
    shape = data.shape
    output = np.zeros_like(data)
    
    # Create coordinate grids for output
    coords_out = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
    
    # Flatten for easier processing
    coords_flat = coords_out.reshape(3, -1)
    
    # Apply inverse transform to find input coordinates
    # First, invert the affine matrix
    # Format: each row is [a, b, c, tx] representing: out_i = a*in_0 + b*in_1 + c*in_2 + tx
    M = np.array(transform_matrix)  # 3x4 matrix
    
    # Convert to homogeneous 4x4
    M_hom = np.eye(4)
    M_hom[:3, :] = M
    
    # Invert
    M_inv = np.linalg.inv(M_hom)
    
    # Add homogeneous coordinate
    coords_hom = np.vstack([coords_flat, np.ones(coords_flat.shape[1])])
    
    # Apply inverse transform
    coords_in_hom = M_inv @ coords_hom
    coords_in = coords_in_hom[:3, :]
    
    # Reshape back
    coords_in_grid = coords_in.reshape(3, *shape)
    
    # Sample from input using the computed coordinates
    output = map_coordinates(data, coords_in_grid, order=1, mode='constant', cval=0)
    
    return output

# Create base data
print("Creating base data...")
base_data = create_test_pattern()
print(f"  Data shape: {base_data.shape}")
print(f"  Nonzero voxels: {np.count_nonzero(base_data)}")
print(f"  L-shaped figure (visible in X-Y top view):")
print(f"    Vertical bar: x=[6:8], y=[4:10], z=[5:11]")
print(f"    Horizontal bar: x=[8:14], y=[4:6], z=[5:11]")

# Define a simple transformation: swap y and x, with offset
forward_transform = [
    [1, 0, 0, 0],      # z_out = z_in
    [0, 0, 1, 0],      # y_out = x_in 
    [0, 1, 0, 0],      # x_out = y_in
]

# Compute inverse
inverse_transform = [
    [1, 0, 0, 0],      # z_in = z_out
    [0, 0, 1, 0],      # y_in = x_out
    [0, 1, 0, 0],      # x_in = y_out
]

print(f"\nForward transform (applied to data):")
for row in forward_transform:
    print(f"  {row}")

print(f"\nInverse transform (in metadata):")
for row in inverse_transform:
    print(f"  {row}")

# Apply forward transform to data
print("\nApplying forward transform to data...")
transformed_data = apply_transform_to_data(base_data, forward_transform)
print(f"  Nonzero voxels after transform: {np.count_nonzero(transformed_data)}")

# Create output directories
base_dir = THIS_DIR / "test_base_0.5.zarr"
intermediate_dir = THIS_DIR / "test_intermediate_0.5.zarr"
transformed_dir = THIS_DIR / "test_transformed_0.6.zarr"

print(f"\nCreating {base_dir}...")
create_ome_zarr_0_5(base_dir, base_data, name="base")

print(f"Creating {intermediate_dir}...")
create_ome_zarr_0_5(intermediate_dir, transformed_data, name="intermediate")

print(f"Creating {transformed_dir}...")
shutil.rmtree(transformed_dir, ignore_errors=True)

# For the transformed file, we need to create a Zarr v3 array manually
# because we need to add a custom affine transformation in the metadata
array_path = transformed_dir / 'array'
array_path.mkdir(parents=True, exist_ok=True)

spec = {
    'driver': 'zarr3',
    'kvstore': {
        'driver': 'file',
        'path': str(array_path)
    },
    'metadata': {
        'shape': list(transformed_data.shape),
        'chunk_grid': {
            'name': 'regular',
            'configuration': {'chunk_shape': list(transformed_data.shape)}
        },
        'chunk_key_encoding': {
            'name': 'default',
            'configuration': {'separator': '/'}
        },
        'codecs': [
            {'name': 'bytes', 'configuration': {'endian': 'little'}}
        ],
        'data_type': str(transformed_data.dtype),
        'fill_value': 0
    }
}

array = ts.open(spec, create=True, dtype=ts.uint16).result()
array[:] = transformed_data

# Create metadata for transformed (inverse transform in metadata)
transformed_metadata = {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
        "ome": {
            "version": "0.6-dev2",
            "multiscales": [{
                "name": "test",
                "coordinateSystems": [{
                    "name": "physical",
                    "axes": [
                        {"name": "z", "type": "space", "unit": "micrometer", "discrete": False},
                        {"name": "y", "type": "space", "unit": "micrometer", "discrete": False},
                        {"name": "x", "type": "space", "unit": "micrometer", "discrete": False},
                    ]
                }],
                "datasets": [{
                    "path": "array",
                    "coordinateTransformations": [{
                        "type": "affine",
                        "affine": inverse_transform,
                        "input": "array",
                        "output": "physical"
                    }]
                }]
            }]
        }
    }
}

with open(transformed_dir / "zarr.json", "w") as f:
    json.dump(transformed_metadata, f, indent=2)

print("\n" + "="*70)
print("Test files created successfully!")
print("="*70)
print(f"\nBase file: {base_dir}")
print(f"  - OME-ZARR 0.5 format")
print(f"  - Identity transform in metadata")
print(f"  - L-shaped figure in original orientation (visible in X-Y top view)")
print(f"    Vertical bar: x=[6:8], y=[4:10], z=[5:11]")
print(f"    Horizontal bar: x=[8:14], y=[4:6], z=[5:11]")
print(f"\nIntermediate file: {intermediate_dir}")
print(f"  - OME-ZARR 0.5 format")
print(f"  - Identity transform in metadata")
print(f"  - Data has forward transform applied (y and x swapped)")
print(f"  - Shows transformed orientation directly - L rotated 90° in X-Y plane")
print(f"\nTransformed file: {transformed_dir}")
print(f"  - OME-ZARR 0.6 format (Zarr v3)")
print(f"  - Inverse transform in metadata")
print(f"  - Data has forward transform applied")
print(f"  - When rendered, should match base file!")
print(f"  - Expected L-shape at original position in physical space")

