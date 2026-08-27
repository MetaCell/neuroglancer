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
 * @file Raycast sphere drawn on a camera-facing quad; see `raycast_primitive.ts`
 * for the shared conventions.
 *
 * Adapted from Inigo Quilez's sphere intersector
 * (https://iquilezles.org/articles/intersectors/) and related shadertoy code.
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
 * Modifications:
 *   - The bounding-quad vertex stage has no counterpart in the original, which
 *     ray-marches a full-screen quad.
 *   - The discriminant uses the perpendicular-distance rearrangement instead of
 *     `c = dot(oc, oc) - r * r`, and moves into the shared
 *     `intersectRaycastCircle` in `raycast_intersect.ts`, which `raycast_cylinder.ts`
 *     also calls. This is for better scaling as neuroglancer can have large depth
 *     range.
 *   - Returns depth and a lighting factor rather than a ray distance.
 */

import { defineRaycastPrimitiveCommon } from "#src/webgl/raycast_primitive.js";
import type { ShaderBuilder } from "#src/webgl/shader.js";

/**
 * Adds `emitRaycastSphere(center, radius)` (vertex) and
 * `intersectRaycastPrimitive()` (fragment); `center`/`radius` are in model space.
 */
export function defineRaycastSphereShader(builder: ShaderBuilder) {
  defineRaycastPrimitiveCommon(builder);
  builder.addVarying("highp vec3", "vSphereCenter", "flat");
  builder.addVarying("highp float", "vSphereRadius", "flat");
  builder.addVertexCode(`
void emitRaycastSphere(highp vec3 center, highp float radius) {
  vSphereCenter = center;
  vSphereRadius = radius;
  emitRaycastAabbQuad(center, vec3(radius));
}
`);
  builder.addFragmentCode(`
RaycastHit intersectRaycastPrimitive() {
  RaycastRay ray = getRaycastEyeRay();

  // A sphere is one ray/circle test about its centre C, and nothing else. The
  // normal at the hit point H is the standard sphere normal H - C, which
  // intersectRaycastCircle returns as offsetFromCenter.
  RaycastCircleHit circleHit = intersectRaycastCircle(
      ray.origin - vSphereCenter, ray.direction, vSphereRadius);
  if (!circleHit.hit) return raycastMiss();
  return makeRaycastHit(ray.origin + circleHit.distanceAlongRay * ray.direction,
                        circleHit.offsetFromCenter);
}
`);
}
