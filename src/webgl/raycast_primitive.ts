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
 * @file Shared GLSL for raycast primitives: a camera-facing quad whose fragment
 * shader ray casts to find the 3D surface, writes `gl_FragDepth` and shades a
 * normal. Intersection is in model space and must be transformed into
 * display-space, similar to `src/annotation/ellipsoid.ts`.
 *
 * `emitRaycastBoundingQuad` is an AABB that any primitive can use. One whose shape
 * admits a tighter bound can define its own instead, as `raycast_cylinder.ts` does.
 */

import { mat4 } from "#src/util/geom.js";
import { glsl_getQuadVertexPosition } from "#src/webgl/quad.js";
import type { ShaderBuilder, ShaderProgram } from "#src/webgl/shader.js";

export const raycastProjectionUniform = (builder: ShaderBuilder) => {
  builder.addUniform("highp mat4", "uProjection");
};

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

RaycastRay getRaycastEyeRay() {
  highp vec2 ndc = (gl_FragCoord.xy / uViewportSize) * 2.0 - 1.0;
  highp vec4 nearClip = uInvProjection * vec4(ndc, -1.0, 1.0);
  highp vec4 farClip = uInvProjection * vec4(ndc, 1.0, 1.0);
  highp vec3 nearModel = nearClip.xyz / nearClip.w;
  highp vec3 farModel = farClip.xyz / farClip.w;
  RaycastRay ray;
  ray.origin = nearModel;
  ray.direction = normalize(farModel - nearModel);
  return ray;
}
highp float getRaycastWindowDepth(highp vec3 modelPoint) {
  // Assumes the default depth range [0, 1] and NDC z in [-1, 1].
  highp vec4 clip = uProjection * vec4(modelPoint, 1.0);
  return 0.5 * (clip.z / clip.w) + 0.5;
}
// modelNormal can be non-normalized.
highp float getRaycastSurfaceLightingFactor(highp vec3 modelNormal) {
  highp vec3 displayNormal = normalize(uNormalTransform * modelNormal);
  return abs(dot(displayNormal, uLightDirection.xyz)) + uLightDirection.w;
}
RaycastHit raycastMiss() {
  RaycastHit hit;
  hit.hit = false;
  return hit;
}
RaycastHit makeRaycastHit(highp vec3 surfacePoint, highp vec3 modelNormal) {
  RaycastHit hit;
  hit.hit = true;
  hit.windowDepth = getRaycastWindowDepth(surfacePoint);
  hit.lightingFactor = getRaycastSurfaceLightingFactor(modelNormal);
  return hit;
}
`;

// Emits the screen-axis-aligned quad covering the model-space box
// `center +/- halfExtent`.
//
// A corner on or behind the near plane must not be dropped -- that would
// under-cover a primitive straddling the near plane and leave it undrawn -- so its
// w is floored positive, which projects it far off-screen; NDC is clamped so that
// expansion stays finite.  Once a corner is clamped the projected-corner hull no
// longer bounds the silhouette, which is what the relative margin covers.
const glsl_raycastPrimitiveVertexUtil = `
const highp float RAYCAST_NDC_BOUND = 2.0;
void emitRaycastBoundingQuad(highp vec3 center, highp vec3 halfExtent) {
  // Projection is linear before the divide, so each axis is one scaled column.
  highp vec4 clipCenter = uProjection * vec4(center, 1.0);
  highp vec4 clipX = uProjection[0] * halfExtent.x;
  highp vec4 clipY = uProjection[1] * halfExtent.y;
  highp vec4 clipZ = uProjection[2] * halfExtent.z;
  highp vec2 ndcMin = vec2(RAYCAST_NDC_BOUND);
  highp vec2 ndcMax = vec2(-RAYCAST_NDC_BOUND);
  highp float ndcNearZ = 1.0;
  for (int corner = 0; corner < 8; ++corner) {
    highp vec4 clip = clipCenter
        + ((corner & 1) == 0 ? -clipX : clipX)
        + ((corner & 2) == 0 ? -clipY : clipY)
        + ((corner & 4) == 0 ? -clipZ : clipZ);
    highp float clipW = max(clip.w, 1e-4);
    highp vec2 ndcXY =
        clamp(clip.xy / clipW, vec2(-RAYCAST_NDC_BOUND), vec2(RAYCAST_NDC_BOUND));
    ndcMin = min(ndcMin, ndcXY);
    ndcMax = max(ndcMax, ndcXY);
    ndcNearZ = min(ndcNearZ, clamp(clip.z / clipW, -1.0, 1.0));
  }
  highp vec2 margin = (ndcMax - ndcMin) * 0.02 + 2.0 / uViewportSize;
  highp vec2 quadCorner = getQuadVertexPosition(ndcMin - margin, ndcMax + margin);
  gl_Position = vec4(quadCorner, ndcNearZ, 1.0);
}
`;

// Model-space radius projecting to `radiusInPixels` device px at `modelPoint`,
// measured on the vertical viewport extent, so raycasts hold a constant on-screen
// size like the billboards they replace.
const glsl_raycastPrimitivePixelRadius = `
highp float getRaycastModelRadiusForPixels(highp vec3 modelPoint, highp float radiusInPixels) {
  highp float clipW = max((uProjection * vec4(modelPoint, 1.0)).w, 1e-6);
  // uInvProjection column 1 is one NDC unit of y in model space; the positive
  // scalar factors straight out of the length.
  return length(uInvProjection[1].xyz) * (2.0 / uViewportSize.y) * clipW * radiusInPixels;
}
`;

export const glsl_raycastFragmentSetup = `
RaycastHit raycastHit = intersectRaycastPrimitive();
if (!raycastHit.hit) discard;
// Positive-form range test, so a NaN depth from a degenerate projection is
// rejected rather than poisoning the OIT weight.
if (!(raycastHit.windowDepth >= 0.0 && raycastHit.windowDepth <= 1.0)) discard;
gl_FragDepth = raycastHit.windowDepth;
raycastSurfaceDepth = raycastHit.windowDepth;
raycastLightingFactor = raycastHit.lightingFactor;
`;

export function defineRaycastPrimitiveCommon(builder: ShaderBuilder) {
  builder.require(raycastProjectionUniform);
  builder.addUniform("highp mat4", "uInvProjection");
  builder.addUniform("highp mat3", "uNormalTransform");
  builder.addUniform("highp vec4", "uLightDirection");
  builder.addUniform("highp vec2", "uViewportSize");
  builder.addVertexCode(glsl_getQuadVertexPosition);
  builder.addVertexCode(glsl_raycastPrimitiveVertexUtil);
  builder.addVertexCode(glsl_raycastPrimitivePixelRadius);
  builder.addFragmentCode(glsl_raycastPrimitiveFragmentUtil);
}

const tempInvProjection = mat4.create();

export function initializeRaycastPrimitiveShader(
  shader: ShaderProgram,
  modelClip: mat4,
  projectionParameters: { width: number; height: number },
) {
  const { gl } = shader;
  gl.uniformMatrix4fv(shader.uniform("uProjection"), false, modelClip);
  mat4.invert(tempInvProjection, modelClip);
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
