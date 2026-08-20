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
 *     `c = dot(oc, oc) - r * r`; see `intersectRaycastPrimitive` below.
 *     This is for better scaling as neuroglancer can have large depth range.
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
  emitRaycastBoundingQuad(center, vec3(radius));
}
`);
  builder.addFragmentCode(`
RaycastHit intersectRaycastPrimitive() {
  RaycastRay ray = getRaycastEyeRay();

  // ray.direction is a unit vector, so this projection locates where the ray passes
  // closest to the centre: at t = -projectedDistance, offset by perpendicular.
  // The half-chord is the third side of a right triangle with hypotenuse radius and
  // leg perpendicular, so it exists only while halfChordSquared is non-negative;
  // the two surface crossings are then at -projectedDistance -/+ halfChord.
  highp vec3 centerToOrigin = ray.origin - vSphereCenter;
  highp float projectedDistance = dot(centerToOrigin, ray.direction);
  highp vec3 perpendicular = centerToOrigin - projectedDistance * ray.direction;
  highp float halfChordSquared =
      vSphereRadius * vSphereRadius - dot(perpendicular, perpendicular);
  // Positive-form guards, so a NaN ray from a degenerate projection misses
  // rather than slipping through (NaN < 0.0 is false).
  if (!(halfChordSquared >= 0.0)) return raycastMiss();  // triangle cannot close
  highp float halfChord = sqrt(halfChordSquared);
  // Only the near crossing is drawn. A negative one means the sphere is behind us or
  // the origin is inside it, and drawing the far surface then fills the view when the
  // camera clips inside the geometry.
  highp float hitDistance = -projectedDistance - halfChord;
  if (!(hitDistance >= 0.0)) return raycastMiss();
  highp vec3 offsetFromCenter = centerToOrigin + hitDistance * ray.direction;
  return makeRaycastHit(ray.origin + hitDistance * ray.direction, offsetFromCenter);
}
`);
}
