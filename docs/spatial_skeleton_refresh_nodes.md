# Why `refreshNodes()` Is Called So Often

This note documents every current `refreshNodes()` call site in `src/ui/spatial_skeleton_edit_tab.ts` and whether it is strictly required by the current implementation.

## What `refreshNodes()` actually recomputes

`refreshNodes()` is not just a repaint helper. It rebuilds the tab's backing data by:

- Recomputing `activeSegmentIds` from `getVisibleSegments(...)`.
- Resolving the current spatial skeleton render layer and CATMAID client.
- Clearing the full-skeleton cache when `layer.spatialSkeletonNodeDataVersion` changes.
- Fetching full CATMAID skeletons for every active segment.
- Fetching CATMAID "true end" labels and applying them to the node list.
- Rebuilding `nodesBySegment`, `allNodes`, the summary text, and the selected node fallback.

Because of that, a refresh is only fundamentally needed when one of those inputs changes.

## Call sites

### 1. `deleteNode(...)` success path

Why it is called:

- Deleting a node changes the local skeleton layer immediately.
- A delete can also split one skeleton into new segment ids, so the active visible segment set may change.
- The tab wants to re-fetch the authoritative CATMAID skeleton after the mutation instead of trusting only local chunk edits.

Assessment:

- Reasonable as an eager refresh.
- Probably redundant in practice, because the same code path also updates `visibleSegments` and bumps `layer.spatialSkeletonNodeDataVersion`, and both already have refresh listeners below.

### 2. `layer.displayState.segmentSelectionState.changed`

Why it is called today:

- Most likely defensive: segment selection changes often happen while the user is interacting with the Seg tab.

Assessment:

- This does not look required by the current code.
- `refreshNodes()` does not read `segmentSelectionState`.
- If only the selected segment changes while the visible segment set stays the same, the node list inputs have not changed.

### 3. `segmentationGroupState.visibleSegments.changed`

Why it is called:

- This is a direct dependency.
- `refreshNodes()` derives `activeSegmentIds` from the visible segment set, so adding or removing visible skeletons must rebuild the node list.

Assessment:

- Required.

### 4. `segmentationGroupState.temporaryVisibleSegments.changed`

Why it is called:

- Merge/split previews can populate the temporary visible segment set.
- When the tab is currently using the temporary set, those preview edits change which skeletons should be shown.

Assessment:

- Required for preview mode correctness.

### 5. `segmentationGroupState.useTemporaryVisibleSegments.changed`

Why it is called:

- `getVisibleSegments(...)` switches between the normal and temporary visible sets based on this flag.
- Flipping the flag changes the source of truth for `activeSegmentIds` even if neither set changed contents.

Assessment:

- Required.

### 6. `layer.layersChanged`

Why it is called:

- `refreshNodes()` resolves the active spatial skeleton render layer on every run.
- If render layers are rebuilt, added, removed, or swapped, the tab may need a different skeleton layer or CATMAID client.

Assessment:

- Required.

### 7. `layer.manager.chunkManager.layerChunkStatisticsUpdated`

Why it is called today:

- Probably to keep the tab in sync with spatial skeleton loading progress.

Assessment:

- `updateGateStatus()` is definitely needed here because action availability depends on chunk loading.
- The `refreshNodes()` part does not look required by the current implementation.
- `refreshNodes()` does not consume chunk statistics directly; it fetches full skeletons from CATMAID based on visible segments.
- This looks like defensive overlap or a holdover from an older "loaded nodes" model.

### 8. `layer.displayState.spatialSkeletonGridLevel2d.changed`

Why it is called today:

- Likely meant to react when the user changes the skeleton grid resolution used by the render layers.

Assessment:

- Not a direct dependency of `refreshNodes()`.
- Grid level changes do affect edit eligibility and chunk loading, but those are already covered by `spatialSkeletonActionsAllowed`, `layersChanged`, and chunk-stat updates.
- For the current CATMAID-backed full-skeleton list, this refresh looks likely redundant.

### 9. `layer.displayState.spatialSkeletonGridLevel3d.changed`

Why it is called today:

- Same rationale as the 2D grid-level watcher.

Assessment:

- Same conclusion as above: likely redundant for the current implementation.

### 10. `layer.spatialSkeletonNodeDataVersion.changed`

Why it is called:

- This is the main explicit invalidation channel for real skeleton edits.
- When the version changes, `refreshNodes()` clears the full-skeleton cache and re-fetches from CATMAID.
- Other tools rely on this after add, move, merge, and split operations.

Assessment:

- Required.
- This is the cleanest "the node topology changed" signal in the file.

### 11. Constructor tail: initial `refreshNodes()`

Why it is called:

- The tab needs an initial population pass after wiring observers.
- Without it, the list would stay empty until some later state change happens.

Assessment:

- Required.

## Practical takeaway

The clearly justified refresh triggers are:

- `visibleSegments.changed`
- `temporaryVisibleSegments.changed`
- `useTemporaryVisibleSegments.changed`
- `layersChanged`
- `spatialSkeletonNodeDataVersion.changed`
- the initial constructor call

The calls that currently look redundant or at least weakly justified are:

- `segmentSelectionState.changed`
- the `refreshNodes()` inside `layerChunkStatisticsUpdated`
- `spatialSkeletonGridLevel2d.changed`
- `spatialSkeletonGridLevel3d.changed`
- the direct refresh after successful delete, because that path already triggers other refresh signals

If we want to reduce refresh churn safely, those are the first places to verify with tests or manual UI checks.
