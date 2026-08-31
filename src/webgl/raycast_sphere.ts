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
 */

import { defineRaycastPrimitiveCommon } from "#src/webgl/raycast_primitive.js";
import type { ShaderBuilder } from "#src/webgl/shader.js";

export function defineRaycastSphereShader(builder: ShaderBuilder) {
  defineRaycastPrimitiveCommon(builder);
  // xyz: center, w: radius.
  builder.addVarying("highp vec4", "vSphere", "flat");
  builder.addVertexCode(`
void emitRaycastSphere(highp vec3 center, highp float radius) {
  // No radius, no surface to hit. A node behind the eye reaches this every frame.
  // Positive form, so a non-finite radius culls too.
  if (!(radius > 0.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vSphere = vec4(center, radius);
  emitRaycastAabbQuad(center, vec3(radius));
}
`);
  builder.addFragmentCode(`
RaycastHit intersectRaycastPrimitive() {
  RaycastRay ray = getRaycastRayThroughFragment();
  RaycastCircleHit circleHit = intersectRaycastCircle(
      ray.origin - vSphere.xyz, ray.direction, vSphere.w);
  if (!circleHit.hit) return raycastMiss();
  return makeRaycastHit(ray.origin + circleHit.distAlongRay * ray.direction,
                        circleHit.normal);
}
`);
}
