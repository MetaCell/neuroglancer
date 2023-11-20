# Volume Rendering

This folder contains code for volume rendering, which is a technique used in computer graphics to display a 3D image.
The code in this folder includes functions and classes for processing volumetric data and rendering the data via ray marching.

## Ray marching

The volume rendering is performed via ray marching. This comes in three different modes:

1. `OFF`: Volume rendering is disabled. Only slices are shown. This is the default. In the shader, the `VOLUME_RENDERING` flag is set to `false` when this mode is enabled, and `true` otherwise.
2. `ON`: Direct volume rendering via ray marching is enabled.
3. `MAX`: Maximum intensity projection is enabled. This is a special case of direct volume rendering where the maximum value along each ray is used. In the shader, the `MAX_PROJECTION` flag is set to `true` when this mode is enabled, and `false` otherwise.

## Shader code

To map the voxel values to a color, two built-in transfer functions are provided:

1. `linear`: The voxel values are mapped linearly to the color space. This can be selected by using the `#uicontrol normalized NAME(PARAMS)` syntax in the shader code.
2. `rgba`: The voxel values are mapped to the color space using a lookup table. This can be selected by using the `#uicontrol transferFunction NAME(PARAMS)` syntax in the shader code.

Currently, the data histogram is only visualised in the `linear` mode for slice views.

To write custom transfer functions, you can use the `getDataValue()` and `getInterpolatedDataValue` functions to build up a transfer function. The `getDataValue()` function returns the value of the nearest voxel, and the `getInterpolatedDataValue()` function returns a weighted contribution from the nearest eight voxels. The value return will be typed, so use `toRaw` or `toNormalized` to convert to a float (high precision). Finally, contribute to the `outputColor` `vec4` variable to set the color of the voxel.

## Front-to-back composition along rays

Color is accumulated along each ray using the `outputColor` variable. The `outputColor` variable is initialised to `(0, 0, 0, 0)` at the start of each ray. The `outputColor` variable is then updated at each step along the ray via: 

```glsl
// rgba is the color of the current voxel
// uSamplingRatio indicates whether the volume is being oversampled or undersampled

// Perform opacity correction for undersampling/oversampling
float alpha = 1.0 - (pow(clamp(1.0 - rgba.a, 0.0, 1.0), uSamplingRatio));

// Accumulate color (premultiplied alpha)
outputColor.rgb += (1.0 - outputColor.a) * alpha * rgba.rgb;

// Accumulate opacity
outputColor.a += (1.0 - outputColor.a) * alpha;
```

## Transfer function control

To add points to the transfer function, select the `transferFunction` control in the UI and click on the transfer function to add a point. The point will be added at the current mouse position, with color matching the selected color in the UI. The point can be dragged around to change its position. The point can be deleted by double clicking on it. To change the color of the point, select the desired color in the UI and then `shift/alt/ctrl/cmd` click on the point. The point will be updated to the selected color.

From the shader, a number of parameters are available to control the transfer function. These are passed as `#uicontrol transferFunction NAME(PARAMS)` in the shader code. The following parameters are available:

1. `channel`: the data channel to use for the transfer function. This data type corresponds to the number of channel dimensions in the data. For a single channel dimension, this is an int/float.
2. `color`: the default color of the transfer function for new points. This is a `vec4`, and supports hex colors (e.g. `#ff0000ff`).
3. `range`: the range of the transfer function. This is a `vec2`, and values should match the data type of the channel.
4. `points`: the points of the transfer function. This is a array of vectors with 5 dimensions, and each point should have the following values:
   * `x`: the position of the point along the transfer function. This is a float, and values should be between 0 and 1.
   * `r`: the red value of the point. This is a uint8, and values should be between 0 and 255.
   * `g`: the green value of the point. This is a uint8, and values should be between 0 and 255.
   * `b`: the blue value of the point. This is a uint8, and values should be between 0 and 255.
   * `a`: the alpha value of the point. This is a uint8, and values should be between 0 and 255.

## Code Architecture

The main modules in this folder are:

1. `base.ts` - establishes how each visible chunk should be processed, and establishes the conversion between physical spacing and view spacing. The conversion between physical spacing and view spacing is used to determine the optimal number of depth samples along each ray for each resolution of the dataset.
2. `backend.ts` - extends the original chunk manager with a volume-rendering-specific chunk manager to establish chunk priority.
3. `volume_render_layer.ts` - links up to UI parameters from the `ImageUserLayer`, binds together callbacks and chunk management, etc. The drawing operation happens here. For each chunk that is visible, all of the shader parameters get passed to the shader (e.g. the model view projection matrix), and then each chunk that is in GPU memory is processed separately and drawn. The state is considered ready if no chunks that are in GPU memory have not yet been drawn. The vertex shader and the fragment shader are defined in this file. Additionally, the user defined fragment shader is injected into the fragment shader here.

   * The vertex shader essentially passes normalised screen coordinates along with the inverse matrix of the model view projection matrix to get back from screen space to model space (see Model View Projection).
   * The fragment shader uses this information to determine the start and end point of each ray based on the screen position given by the vertex shader. The fragment shader then establishes how color is accumulated along the rays. The ray start and end points are set up such that the rays all lie within the view-clipping bounds and volume bounds. Finally, the rays are marched through that small clipping box, providing the `curChunkPosition` at each step and also allowing access to the scalar voxel value via `getDataValue()` for the nearest voxel, or `getInterpolatedDataValue()` for a weighted contribution from the nearest eight voxels. The value return will be typed, so use `toRaw` or `toNormalized` to convert to a float (high precision).
