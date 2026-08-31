/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Shared GLSL for raycast primitives: a camera facing screen space quad
 * whose fragment shader ray casts to find the 3D surface.
 * Emits the depth and a lighting factor.
 *
 * Positions, radii and normals are in raycast space, the space that `uProjection`
 * maps to clip space. That space must reach clip space through a rotation, a
 * uniform scale and the projection alone. Any anisotropic scale left in
 * `uProjection` draws a sphere as an ellipsoid, because the intersection solves a
 * true sphere in raycast space. `uLightDirection` is read in the same space, so
 * the surface normal needs no further transform.
 *
 * `emitRaycastAabbQuad` and `emitRaycastAxialObbQuad` bound a primitive for
 * rasterisation by bounding the object in raycast space - then projecting to
 * screen space and emit the screen space quad which covers the
 * projected bounding box.
 * Use the AABB (axis aligned bounding box) for objects like spheres, cubes
 * and other fairly uniform geometries.
 * Use the axial OBB (oriented bounding box) for objects with one defined long
 * axis, like cylinders, capsules, cones, etc.
 */

import { mat4 } from "#src/util/geom.js";
import { glsl_getQuadVertexPosition } from "#src/webgl/quad.js";
import {
  glsl_intersectRaycastCircle,
  glsl_splitAlongDirection,
} from "#src/webgl/raycast_shader_lib.js";
import type { ShaderBuilder, ShaderProgram } from "#src/webgl/shader.js";
import { glsl_clipLineToDepthRange } from "#src/webgl/shader_lib.js";

export function projectionMatrixShaderModule(builder: ShaderBuilder) {
  builder.addUniform("highp mat4", "uProjection");
}

const glsl_raycastPrimitiveFragmentUtil = `
struct RaycastRay {
  highp vec3 origin;
  highp vec3 direction;
};
struct RaycastHit {
  bool hit;
  highp float windowDepth;
  highp float lightingFactor;
};

highp float raycastSurfaceDepth = 0.0;
highp float raycastLightingFactor = 1.0;

RaycastRay getRaycastRayThroughFragment() {
  highp vec2 ndc = (gl_FragCoord.xy / uViewportSize) * 2.0 - 1.0;
  highp vec4 nearClip = uInvProjection * vec4(ndc, -1.0, 1.0);
  highp vec4 farClip = uInvProjection * vec4(ndc, 1.0, 1.0);
  highp vec3 nearPoint = nearClip.xyz / nearClip.w;
  highp vec3 farPoint = farClip.xyz / farClip.w;
  RaycastRay ray;
  ray.origin = nearPoint;
  ray.direction = normalize(farPoint - nearPoint);
  return ray;
}
highp float getRaycastWindowDepth(highp vec3 point) {
  highp vec4 clip = uProjection * vec4(point, 1.0);
  return 0.5 * (clip.z / clip.w) + 0.5;
}
highp float getRaycastSurfaceLightingFactor(highp vec3 normal) {
  return abs(dot(normalize(normal), uLightDirection.xyz)) + uLightDirection.w;
}
RaycastHit raycastMiss() {
  RaycastHit hit;
  hit.hit = false;
  return hit;
}
RaycastHit makeRaycastHit(highp vec3 surfacePoint, highp vec3 normal) {
  RaycastHit hit;
  hit.hit = true;
  hit.windowDepth = getRaycastWindowDepth(surfacePoint);
  hit.lightingFactor = getRaycastSurfaceLightingFactor(normal);
  return hit;
}
`;

// The emitters below over-cover so that a primitive straddling the near plane is
// never lost. That also drags an out-of-range primitive back on screen, past a
// fixed-function clipper that can no longer see where it really is.
const glsl_raycastDepthRangeCull = `
highp vec2 raycastDepthPlaneDistances(highp vec4 clip) {
  return vec2(clip.z + clip.w, clip.w - clip.z);
}
// Both distances are linear, so callers pass the maximum over the box corners. That
// is the larger base value plus the magnitude of each half-extent term. Negative
// form, so a non-finite value fails open and leaves the box drawn.
bool raycastBoxOutsideDepthRange(highp vec2 maxDepthDistances) {
  return maxDepthDistances.x < 0.0 || maxDepthDistances.y < 0.0;
}
`;

const glsl_raycastQuadConstants = `
// Must exceed 1.0. Pinned exactly at the viewport edge, the margin an emitter adds
// would drag a fully off-screen primitive back on screen as a sliver.
const highp float RAYCAST_OFFSCREEN_NDC = 2.0;
// Smallest clip w a projected point may be treated as having, as a fraction of the
// local w scale. Relative, so it holds whatever units the projection works in.
const highp float RAYCAST_MIN_RELATIVE_W = 1e-4;
// Nearest clip w an axis may keep, as a multiple of the depth that the radial
// half-extents span. The margin over 1.0 is what a corner keeps in front of the
// eye, and so what caps how far outside the viewport a corner can project.
const highp float RAYCAST_MIN_AXIS_W_MARGIN = 1.25;
`;

// Emits the screen-axis-aligned quad covering the raycast-space box
// `center +/- halfExtent`.
//
// Dropping a corner on or behind the near plane would leave a primitive that
// straddles the near plane undrawn. Its w is floored positive instead, which throws
// it far off-screen, and the NDC is clamped to keep the box finite. A clamped corner
// no longer bounds the silhouette, which is what the relative margin covers.
const glsl_raycastAabbQuad = `
void emitRaycastAabbQuad(highp vec3 center, highp vec3 halfExtent) {
  highp vec4 clipCenter = uProjection * vec4(center, 1.0);
  highp vec4 clipX = uProjection[0] * halfExtent.x;
  highp vec4 clipY = uProjection[1] * halfExtent.y;
  highp vec4 clipZ = uProjection[2] * halfExtent.z;

  if (raycastBoxOutsideDepthRange(
          raycastDepthPlaneDistances(clipCenter)
          + abs(raycastDepthPlaneDistances(clipX))
          + abs(raycastDepthPlaneDistances(clipY))
          + abs(raycastDepthPlaneDistances(clipZ)))) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  // The largest |w| any corner can reach, and so the box's own w scale. Zero only
  // for a zero-extent box on the eye plane, which draws nothing either way.
  highp float maxAbsW = abs(clipCenter.w)
      + abs(clipX.w) + abs(clipY.w) + abs(clipZ.w);
  if (!(maxAbsW > 0.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  highp float minClipW = RAYCAST_MIN_RELATIVE_W * maxAbsW;

  highp vec2 ndcMin = vec2(RAYCAST_OFFSCREEN_NDC);
  highp vec2 ndcMax = vec2(-RAYCAST_OFFSCREEN_NDC);
  highp float ndcNearZ = 1.0;

  for (int corner = 0; corner < 8; ++corner) {
    highp vec4 clip = clipCenter
        + ((corner & 1) == 0 ? -clipX : clipX)
        + ((corner & 2) == 0 ? -clipY : clipY)
        + ((corner & 4) == 0 ? -clipZ : clipZ);
    highp float clipW = max(clip.w, minClipW);
    highp vec2 ndcXY =
        clamp(clip.xy / clipW, vec2(-RAYCAST_OFFSCREEN_NDC), vec2(RAYCAST_OFFSCREEN_NDC));
    ndcMin = min(ndcMin, ndcXY);
    ndcMax = max(ndcMax, ndcXY);
    ndcNearZ = min(ndcNearZ, clamp(clip.z / clipW, -1.0, 1.0));
  }

  highp vec2 margin = (ndcMax - ndcMin) * 0.02 + 2.0 / uViewportSize;
  highp vec2 quadCorner = getQuadVertexPosition(ndcMin - margin, ndcMax + margin);
  gl_Position = vec4(quadCorner, ndcNearZ, 1.0);
}
`;

// A quad oriented along the projected axis, covering the OBB about the segment
// endpointA..endpointB with radial half-extents radiusVectorA/B.
//
// Depth-clipping the segment first is what makes an oriented quad possible. A
// primitive crossing the eye plane has an unbounded footprint, and clipping leaves
// every corner in front of the eye where the projected-corner hull is a valid bound.
// The radial half-extents reach nearer than the axis does, so the near end is
// trimmed again by their own depth. The part dropped there wraps the eye, and no
// quad of bounded size covers it.
const glsl_raycastAxialObbQuad = `
highp vec2 raycastClipToPixels(highp vec4 clip) {
  return clip.xy / clip.w * uViewportSize * 0.5;
}
void emitRaycastAxialObbQuad(highp vec3 endpointA, highp vec3 endpointB,
                             highp vec3 radiusVectorA, highp vec3 radiusVectorB) {
  highp vec4 clipA = uProjection * vec4(endpointA, 1.0);
  highp vec4 clipB = uProjection * vec4(endpointB, 1.0);
  highp vec4 clipVectorA = uProjection * vec4(radiusVectorA, 0.0);
  highp vec4 clipVectorB = uProjection * vec4(radiusVectorB, 0.0);
  highp vec2 quadCoefficient = getQuadVertexPosition(vec2(-1.0), vec2(1.0));

  if (raycastBoxOutsideDepthRange(
          max(raycastDepthPlaneDistances(clipA), raycastDepthPlaneDistances(clipB))
          + abs(raycastDepthPlaneDistances(clipVectorA))
          + abs(raycastDepthPlaneDistances(clipVectorB)))) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  // Clips clipA and clipB in place, so everything below uses the clipped segment.
  bool clipped = clipLineToDepthRange(clipA, clipB);

  // w runs linearly along the axis, so one crossing bounds the near end.
  highp float radialW = abs(clipVectorA.w) + abs(clipVectorB.w);
  highp float minAxisW = max(radialW * RAYCAST_MIN_AXIS_W_MARGIN,
                             RAYCAST_MIN_RELATIVE_W * max(clipA.w, clipB.w));
  highp float axisDeltaW = clipB.w - clipA.w;
  highp float startT = 0.0;
  highp float endT = 1.0;
  if (axisDeltaW > 0.0) {
    startT = max(startT, (minAxisW - clipA.w) / axisDeltaW);
  } else if (axisDeltaW < 0.0) {
    endT = min(endT, (minAxisW - clipA.w) / axisDeltaW);
  }
  highp vec4 axisA = mix(clipA, clipB, startT);
  highp vec4 axisB = mix(clipA, clipB, endT);

  // Positive form, so a non-finite result culls rather than proceeding. The equal
  // depth case trims nothing, so the corner test still has to reject it.
  if (!(clipped && startT < endT && min(axisA.w, axisB.w) >= radialW)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  highp vec2 pixelsA = raycastClipToPixels(axisA);
  highp vec2 pixelsB = raycastClipToPixels(axisB);
  highp vec2 axisPixels = pixelsB - pixelsA;
  highp float axisLengthPixels = length(axisPixels);
  highp vec2 alongDirection =
      axisLengthPixels > 1e-3 ? axisPixels / axisLengthPixels : vec2(1.0, 0.0);
  highp vec2 perpDirection = vec2(-alongDirection.y, alongDirection.x);
  highp vec2 pixelCenter = (pixelsA + pixelsB) * 0.5;
  highp float halfAlongPixels = 0.0;
  highp float halfPerpPixels = 0.0;
  highp float ndcNearZ = 1.0;

  for (int corner = 0; corner < 8; ++corner) {
    highp vec4 clip = ((corner & 1) == 0 ? axisA : axisB)
        + ((corner & 2) == 0 ? -clipVectorA : clipVectorA)
        + ((corner & 4) == 0 ? -clipVectorB : clipVectorB);
    highp vec2 offset = raycastClipToPixels(clip) - pixelCenter;
    halfAlongPixels = max(halfAlongPixels, abs(dot(offset, alongDirection)));
    halfPerpPixels = max(halfPerpPixels, abs(dot(offset, perpDirection)));
    ndcNearZ = min(ndcNearZ, clamp(clip.z / clip.w, -1.0, 1.0));
  }

  // The corner bound is exact. One pixel covers numerical error.
  highp vec2 pixels = pixelCenter
      + alongDirection * (quadCoefficient.x * (halfAlongPixels + 1.0))
      + perpDirection * (quadCoefficient.y * (halfPerpPixels + 1.0));
  gl_Position = vec4(pixels * 2.0 / uViewportSize, ndcNearZ, 1.0);
}
`;

// Raycast-space radius projecting to `radiusInPixels` device px, measured on the
// vertical viewport extent. A primitive sized this way holds a constant on-screen
// size as the camera moves.
const glsl_raycastPrimitivePixelRadius = `
highp float raycastRadiusFromClipW(highp float clipW, highp float radiusInPixels) {
  // At or behind the eye there is no on-screen size to match.
  if (!(clipW > 0.0)) return 0.0;
  // uInvProjection column 1 is one NDC unit of y in raycast space. The positive
  // scalar factors straight out of the length. Raycast space reaches the eye
  // through a rotation and a uniform scale, so the length does not turn with the
  // camera.
  return length(uInvProjection[1].xyz) * (2.0 / uViewportSize.y) * clipW * radiusInPixels;
}
highp float getRaycastRadiusForPixels(highp vec3 point, highp float radiusInPixels) {
  return raycastRadiusFromClipW((uProjection * vec4(point, 1.0)).w, radiusInPixels);
}
// A radius for each end of a segment, so that the segment holds one on-screen
// width along its whole length. A single radius cannot: the far end of a receding
// segment would draw thinner than the near end, and thinner than a node drawn
// there at the same pixel radius.
//
// x is the radius at endpointA and y the radius at endpointB. An endpoint at or
// behind the eye has no on-screen size, so it borrows the other end's radius and
// the segment draws without taper. Both behind the eye leaves both zero, which
// the emitter culls.
highp vec2 getRaycastSegmentRadiiForPixels(
    highp vec3 endpointA, highp vec3 endpointB, highp float radiusInPixels) {
  highp vec2 radii = vec2(
      getRaycastRadiusForPixels(endpointA, radiusInPixels),
      getRaycastRadiusForPixels(endpointB, radiusInPixels));
  if (!(radii.x > 0.0)) radii.x = radii.y;
  if (!(radii.y > 0.0)) radii.y = radii.x;
  return radii;
}
`;

export const glsl_raycastFragmentSetup = `
RaycastHit raycastHit = intersectRaycastPrimitive();
if (!raycastHit.hit) discard;
// Positive form, so a non-finite depth fails closed rather than poisoning the OIT
// weight.
if (!(raycastHit.windowDepth >= 0.0 && raycastHit.windowDepth <= 1.0)) discard;
gl_FragDepth = raycastHit.windowDepth;
raycastSurfaceDepth = raycastHit.windowDepth;
raycastLightingFactor = raycastHit.lightingFactor;
`;

export function defineRaycastPrimitiveCommon(builder: ShaderBuilder) {
  builder.require(projectionMatrixShaderModule);
  builder.addUniform("highp mat4", "uInvProjection");
  builder.addUniform("highp vec4", "uLightDirection");
  builder.addUniform("highp vec2", "uViewportSize");
  builder.addVertexCode(glsl_getQuadVertexPosition);
  builder.addVertexCode(glsl_clipLineToDepthRange);
  builder.addVertexCode(glsl_raycastDepthRangeCull);
  builder.addVertexCode(glsl_raycastQuadConstants);
  builder.addVertexCode(glsl_raycastAabbQuad);
  builder.addVertexCode(glsl_raycastAxialObbQuad);
  builder.addVertexCode(glsl_raycastPrimitivePixelRadius);
  builder.addFragmentCode(glsl_raycastPrimitiveFragmentUtil);
  builder.addFragmentCode(glsl_splitAlongDirection);
  builder.addFragmentCode(glsl_intersectRaycastCircle);
}

const tempInvProjection = mat4.create();

// `raycastClip` maps raycast space to clip space. See the constraint on that
// space at the top of this file.
export function initializeRaycastPrimitiveShader(
  shader: ShaderProgram,
  raycastClip: mat4,
  projectionParameters: { width: number; height: number },
) {
  const { gl } = shader;
  gl.uniformMatrix4fv(shader.uniform("uProjection"), false, raycastClip);
  mat4.invert(tempInvProjection, raycastClip);
  gl.uniformMatrix4fv(
    shader.uniform("uInvProjection"),
    false,
    tempInvProjection,
  );
  gl.uniform2f(
    shader.uniform("uViewportSize"),
    projectionParameters.width,
    projectionParameters.height,
  );
}
