#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "icechunk",
#     "zarr>=3",
# ]
# ///

# Writes the test repositories of one Icechunk format. Each format needs its
# own Icechunk release:
#
#   uv run --with 'icechunk==1.1.21' generate_icechunk.py 1
#   uv run --with 'icechunk==2.4.0' generate_icechunk.py 2
#
# Format 2 also migrates the format 1 repository, so write format 1 first.
#
# The format 0.1 repositories in this directory were written by Icechunk
# 0.1.x, whose releases no longer install. Their ids are pinned by the tests,
# so this script does not rewrite them.

import asyncio
import json
import os
import pathlib
import shutil
import sys

import icechunk
import numpy as np
import zarr
import zarr.core.buffer

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

FORMAT_DIRECTORIES = {"1": "v1", "2": "v2"}


# With `repo_info_copy_path`, also saves the Icechunk 2 repo info as it was
# before the branches and tags other than main existed.
def write_zarr_array(path, repo_info_copy_path=None):
    shutil.rmtree(path, ignore_errors=True)
    storage = icechunk.local_filesystem_storage(path)

    repo = icechunk.Repository.create(storage)

    session = repo.writable_session("main")

    a = zarr.create_array(
        shape=[6, 7],
        chunks=[4, 5],
        dtype="uint8",
        store=session.store,
        compressors=None,
    )
    a[...] = np.arange(np.prod(a.shape), dtype=a.dtype).reshape(a.shape)

    session.commit("first commit")
    if repo_info_copy_path is not None:
        shutil.copyfile(os.path.join(path, "repo"), repo_info_copy_path)

    main_snapshot = repo.lookup_branch("main")

    repo.create_branch("other_branch", main_snapshot)
    repo.create_tag("tag1", main_snapshot)
    repo.create_tag("tag2", main_snapshot)
    repo.create_tag("tag3", main_snapshot)
    repo.delete_tag("tag3")


def write_zarr_hierarchy(path):
    shutil.rmtree(path, ignore_errors=True)
    storage = icechunk.local_filesystem_storage(path)

    repo = icechunk.Repository.create(storage)

    session = repo.writable_session("main")

    root = zarr.group(session.store, attributes={"a": 10})
    a = root.create_array(
        "a", shape=[6, 7], chunks=[4, 5], dtype="uint8", compressors=None
    )
    a[...] = np.arange(np.prod(a.shape), dtype=a.dtype).reshape(a.shape)
    root.create_array("bar", shape=[6, 7], chunks=[4, 5], dtype="int32")
    c = root.create_group("cde", attributes={"b": 11})
    c.create_array("abc", shape=[6, 7], chunks=[4, 5], dtype="int32")
    c.create_array("def", shape=[6, 7], chunks=[4, 5], dtype="int32")
    root.create_array("def", shape=[6, 7], chunks=[4, 5], dtype="int32")
    root.create_group("e", attributes={"b": 11})
    c.create_array("xyz", shape=[6, 7], chunks=[4, 5], dtype="int32")

    session.commit("first commit")


def copy_to_dir(path, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    storage = icechunk.local_filesystem_storage(path)
    repo = icechunk.Repository.open(storage)
    read_session = repo.readonly_session(branch="main")
    store = read_session.store

    async def do_copy():
        buffer_prototype = zarr.core.buffer.default_buffer_prototype()
        async for key in store.list():
            value = (await store.get(key, prototype=buffer_prototype)).to_bytes()
            if key.endswith("zarr.json"):
                # icechunk incorrectly includes `"dimension_names": null` fields.
                # https://github.com/earth-mover/icechunk/issues/706
                j = json.loads(value)
                if "dimension_names" in j and j["dimension_names"] is None:
                    del j["dimension_names"]
                value = json.dumps(j).encode("utf-8")
            output_path = os.path.join(output_dir, key)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pathlib.Path(output_path).write_bytes(value)

    asyncio.run(do_copy())


def create_virtual_repository(path, manifest_config=None):
    shutil.rmtree(path, ignore_errors=True)
    storage = icechunk.local_filesystem_storage(path)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "s3://mybucket/", icechunk.s3_store(region="us-east-1")
        )
    )
    if manifest_config is not None:
        config.manifest = manifest_config
    return icechunk.Repository.create(storage, config=config)


def write_virtual_ref_test(path):
    repo = create_virtual_repository(path)
    session = repo.writable_session("main")

    zarr.create_array(
        shape=[4, 5],
        chunks=[4, 5],
        dtype="uint8",
        store=session.store,
        compressors=None,
    )
    session.store.set_virtual_ref(
        "c/0/0", "s3://mybucket/myobject", offset=5, length=20
    )

    session.commit("first commit")


# Icechunk 2 compresses virtual chunk locations with a zstd dictionary
# once a manifest holds `min_num_chunks` chunks.
def write_compressed_virtual_ref_test(path):
    repo = create_virtual_repository(
        path,
        icechunk.ManifestConfig(
            virtual_chunk_location_compression=icechunk.ManifestVirtualChunkLocationCompressionConfig(
                min_num_chunks=1
            )
        ),
    )
    session = repo.writable_session("main")

    zarr.create_array(
        shape=[4, 5],
        chunks=[1, 1],
        dtype="uint8",
        store=session.store,
        compressors=None,
    )
    for i in range(4):
        for j in range(5):
            session.store.set_virtual_ref(
                f"c/{i}/{j}",
                f"s3://mybucket/object_{i}_{j}",
                offset=i * 5 + j,
                length=1,
            )

    session.commit("first commit")


# After the migration, the stale `refs/` files still name the first commit,
# while `repo` names a second commit with an "after_migration" attribute.
def migrate_with_v1_files(v1_path, path):
    shutil.rmtree(path, ignore_errors=True)
    shutil.copytree(v1_path, path)
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    repo = icechunk.upgrade_icechunk_repository(
        repo, dry_run=False, delete_unused_v1_files=False
    )
    session = repo.writable_session("main")
    zarr.open_array(session.store, mode="r+").attrs["after_migration"] = True
    session.commit("after migration")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in FORMAT_DIRECTORIES:
        sys.exit(f"usage: {sys.argv[0]} {{{','.join(FORMAT_DIRECTORIES)}}}")
    format = sys.argv[1]
    major_version = icechunk.__version__.split(".")[0]
    if major_version != format:
        sys.exit(f"format {format} needs Icechunk {format}.x, not {icechunk.__version__}")
    output_dir = os.path.join(SCRIPT_DIR, FORMAT_DIRECTORIES[format])

    def output_path(name):
        return os.path.join(output_dir, name)

    write_zarr_array(
        output_path("single_array.icechunk"),
        output_path("single_array_repo_before_refs") if format == "2" else None,
    )
    write_zarr_hierarchy(output_path("hierarchy.icechunk"))
    for name in ["single_array", "hierarchy"]:
        copy_to_dir(output_path(f"{name}.icechunk"), output_path(name))
    write_virtual_ref_test(output_path("virtual_ref.icechunk"))
    if format == "2":
        write_compressed_virtual_ref_test(
            output_path("compressed_virtual_ref.icechunk")
        )
        migrate_with_v1_files(
            os.path.join(SCRIPT_DIR, FORMAT_DIRECTORIES["1"], "single_array.icechunk"),
            output_path("migrated_single_array.icechunk"),
        )


main()
