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
 * @file Raycast sphere drawn on a camera-facing quad. The vertex stage bounds the
 * sphere with a quad and the fragment stage returns depth and a lighting factor.
 *
 * Both halves follow Inigo Quilez's sphere functions
 * (https://iquilezles.org/articles/intersectors/ and
 * https://iquilezles.org/articles/spherefunctions/), MIT licensed:
 *
 *   The MIT License. Copyright (c) 2016 Inigo Quilez.
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to
 *   deal in the Software without restriction, including without limitation the
 *   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 *   sell copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions: the above copyright
 *   notice and this permission notice shall be included in all copies or
 *   substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS".
 *
 * The bound is the sphere's own silhouette, which is exact and needs no margin.
 * See `nearQuadraticRoot` in `raycast_shader_lib.ts` for how the intersection
 * differs from the original.
 */

import { raycastPrimitiveCoreModule } from "#src/webgl/raycast_primitive.js";
import type { ShaderBuilder } from "#src/webgl/shader.js";

export function defineRaycastSphereShader(builder: ShaderBuilder) {
  builder.require(raycastPrimitiveCoreModule);
  // xyz: center, w: radius.
  builder.addVarying("highp vec4", "vSphere", "flat");
  builder.addVertexCode(`
// The quad covering the sphere's screen-space silhouette.
//
// That silhouette is a conic, and its clip-space form is the dual quadric
// M * Q * transpose(M), for M the x, y and w rows of uProjection and Q the dual of
// the sphere. Only five entries are needed, and each reduces to a product of two
// clip-space center components minus radiusSq times a dot product of two rows of
// uProjection. The extent along an axis is then the pair of roots of
// conicWW * t^2 - 2 * conicCross * t + conicDiagonal.
void emitRaycastSphereQuad(highp vec3 center, highp float radius) {
  highp vec4 clipCenter = uProjection * vec4(center, 1.0);
  highp float radiusSq = radius * radius;
  highp vec3 rowX = vec3(uProjection[0].x, uProjection[1].x, uProjection[2].x);
  highp vec3 rowY = vec3(uProjection[0].y, uProjection[1].y, uProjection[2].y);
  highp vec3 rowZ = vec3(uProjection[0].z, uProjection[1].z, uProjection[2].z);
  highp vec3 rowW = vec3(uProjection[0].w, uProjection[1].w, uProjection[2].w);

  // Each plane distance is largest at the center plus the radius along that
  // distance's own gradient.
  if (raycastOutsideDepthRange(
          raycastDepthPlaneDistances(clipCenter)
          + radius * vec2(length(rowZ + rowW), length(rowW - rowZ)))) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  // Positive exactly when the sphere clears the eye plane, which is when the conic
  // is an ellipse. Otherwise part of the sphere projects arbitrarily far and no
  // quad short of the whole viewport covers it. Positive form, so a non-finite
  // value takes the whole viewport rather than emitting a garbage quad.
  highp float conicWW = clipCenter.w * clipCenter.w - radiusSq * dot(rowW, rowW);
  if (!(conicWW > 0.0)) {
    gl_Position = vec4(getQuadVertexPosition(vec2(-1.0), vec2(1.0)), 0.0, 1.0);
    return;
  }

  highp vec2 conicDiagonal = vec2(
      clipCenter.x * clipCenter.x - radiusSq * dot(rowX, rowX),
      clipCenter.y * clipCenter.y - radiusSq * dot(rowY, rowY));
  highp vec2 conicCross = vec2(
      clipCenter.x * clipCenter.w - radiusSq * dot(rowX, rowW),
      clipCenter.y * clipCenter.w - radiusSq * dot(rowY, rowW));
  // The max guards rounding only. A real ellipse cannot give a negative value.
  highp vec2 halfExtent =
      sqrt(max(conicCross * conicCross - conicDiagonal * conicWW, vec2(0.0)))
      / conicWW;
  highp vec2 centerNdc = conicCross / conicWW;
  highp vec2 ndcMin = centerNdc - halfExtent;
  highp vec2 ndcMax = centerNdc + halfExtent;
  if (any(greaterThan(ndcMin, vec2(RAYCAST_OFFSCREEN_NDC))) ||
      any(lessThan(ndcMax, vec2(-RAYCAST_OFFSCREEN_NDC)))) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  // The fragment shader writes gl_FragDepth and discards a depth outside the
  // range, so the quad's own depth only has to survive clipping. Zero always does.
  gl_Position = vec4(
      clamp(getQuadVertexPosition(ndcMin, ndcMax),
            vec2(-RAYCAST_OFFSCREEN_NDC), vec2(RAYCAST_OFFSCREEN_NDC)),
      0.0, 1.0);
}

void emitRaycastSphere(highp vec3 center, highp float radius) {
  // No radius, no surface to hit. A node behind the eye reaches this every frame.
  // Positive form, so a non-finite radius culls too.
  if (!(radius > 0.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vSphere = vec4(center, radius);
  emitRaycastSphereQuad(center, radius);
}
`);
  builder.addFragmentCode(`
RaycastHit intersectRaycastPrimitive() {
  RaycastRay ray = getRaycastRayThroughFragment();
  highp float radius = vSphere.w;

  // Split along the ray direction, so the parameter below is measured from the
  // closest approach to the center. The direction is unit, so the leading
  // coefficient is one and the linear term vanishes.
  VectorSplit originSplit =
      splitAlongDirection(ray.origin - vSphere.xyz, ray.direction);
  highp float perpendicularDistSq =
      dot(originSplit.perpendicular, originSplit.perpendicular);
  QuadraticNearRoot root =
      nearQuadraticRoot(1.0, 0.0, perpendicularDistSq - radius * radius);
  if (!root.exists) return raycastMiss();

  // The near crossing. Taking the far one would fill the view when the camera
  // clips inside the geometry.
  highp float hitDist = -originSplit.parallelDist + root.value;
  if (!(hitDist >= 0.0)) return raycastMiss();

  // Hit point minus center, formed from two small terms rather than by subtracting
  // the center from a hit point that can be far from the origin.
  return makeRaycastHit(
      ray.origin + hitDist * ray.direction,
      originSplit.perpendicular + root.value * ray.direction);
}
`);
}
