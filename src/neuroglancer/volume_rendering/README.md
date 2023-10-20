# Volume Rendering

This folder contains code for volume rendering, which is a technique used in computer graphics to display a 3D image.
The code in this folder includes functions and classes for processing volumetric data and rendering the data via ray casting.

## Code Architecture

The main modules in this folder are:

1. `base.ts` - This primarily sets up the near and far bounds of the volume rendering view frustum, establishes how each visible chunk should be processed, and establishes the conversion between physical spacing and view spacing. The conversion between physical spacing and view spacing is used to establish the current resolution of the data chosen in multi-resolution datasets in tandem with the user selected number of samples along a single ray.
2. `backend.ts` - This extends the original chunk manager with a volume-rendering-specific chunk manager. Chunk priority is established here. On view change, these get updated.
3. `volume_render_layer.ts` - This links up to UI parameters from the `ImageUserLayer`, binds together callbacks and chunk management, etc. The drawing operation also happens here. For each chunk that is visible, all of the shader parameters get passed to the shader (e.g. the model view projection matrix), and then each chunk that is in GPU memory is processed separately and drawn. The state is considered ready if no chunks that are in GPU memory have not yet been drawn. The vertex shader and the fragment shader are defined in this file. Additionally, the user defined fragment shader is injected into the fragment shader here.

   * The vertex shader essentially passes normalised screen coordinates along with the inverse matrix of the model view projection matrix to get back from screen space to model space (see Model View Projection).
   * The fragment shader uses this information to determine the start and end point of each ray based on the screen position given by the vertex shader. The fragment shader then establishes how color is accumulated along the rays. The ray start and end points are set up such that the rays all lie within the view-clipping bounds and volume bounds. Finally, the rays are marched through that small clipping box, providing the `curChunkPosition` at each step and also allowing access to the scalar voxel value via `getDataValue()` for the nearest voxel, or `getInterpolatedDataValue()` for a weighted contribution from the nearest eight voxels. The value return will be typed, so use `toRaw` or `toNormalized` to convert to a float (high precision).
