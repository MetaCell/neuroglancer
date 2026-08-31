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
 * @file Raycast cylinder drawn on a camera-facing quad. The vertex stage bounds the
 * cylinder with a quad and the fragment stage returns depth and a lighting factor.
 *
 * The ends are open, because skeleton joints are drawn as spheres. Each end also
 * takes a clip radius, which removes the part of the surface that the joint covers.
 */

import { defineRaycastPrimitiveCommon } from "#src/webgl/raycast_primitive.js";
import type { ShaderBuilder } from "#src/webgl/shader.js";

export function defineRaycastCylinderShader(builder: ShaderBuilder) {
  defineRaycastPrimitiveCommon(builder);
  // The cylinder is a base circle swept along an axis. xyz: center of that
  // circle, which is endpoint A. w: its radius.
  builder.addVarying("highp vec4", "vCylinderBaseCircle", "flat");
  // xyz: unit axis direction, w: axis length.
  builder.addVarying("highp vec4", "vCylinderAxis", "flat");
  // x: clip radius at endpoint A, y: clip radius at endpoint B.
  builder.addVarying("highp vec2", "vCylinderClipRadii", "flat");
  builder.addVertexCode(`
void emitRaycastCylinder(highp vec3 endpointA, highp vec3 endpointB,
                         highp float radius,
                         highp float clipRadiusA, highp float clipRadiusB) {
  // No radius, no surface to hit. A segment with both endpoints behind the eye
  // reaches this every frame. Positive form, so a non-finite radius culls too.
  if (!(radius > 0.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vCylinderBaseCircle = vec4(endpointA, radius);
  vCylinderClipRadii = vec2(clipRadiusA, clipRadiusB);
  highp vec3 axisVector = endpointB - endpointA;
  highp float axisLength = length(axisVector);
  highp vec3 axisDirection = axisLength > 1e-6 ? axisVector / axisLength : vec3(0.0, 1.0, 0.0);
  vCylinderAxis = vec4(axisDirection, axisLength);

  // Two perpendicular radius vectors spanning the circular cross-section. The
  // scale by radius comes last: a zero radius would otherwise leave the second
  // cross product normalising the zero vector, which GLSL ES leaves undefined.
  highp vec3 offAxisVector =
      abs(axisDirection.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
  highp vec3 unitRadiusA = normalize(cross(offAxisVector, axisDirection));
  // Already unit length, being the cross product of two perpendicular unit vectors.
  highp vec3 unitRadiusB = cross(axisDirection, unitRadiusA);
  emitRaycastAxialObbQuad(endpointA, endpointB,
                          unitRadiusA * radius, unitRadiusB * radius);
}
`);
  builder.addFragmentCode(`
// Where the surface point falls between the endpoints, 0.0 at A and 1.0 at B.
// Only meaningful once intersectRaycastPrimitive has returned a hit.
highp float raycastCylinderAxialFraction = 0.0;

// A surface point sits exactly one radius from the axis, so its distance to an
// endpoint follows from the axial distance alone.
bool cylinderEndClipped(highp float axialDist, highp float radius) {
  highp float axialDistFromB = axialDist - vCylinderAxis.w;
  highp float radiusSq = radius * radius;
  return axialDist * axialDist + radiusSq
             < vCylinderClipRadii.x * vCylinderClipRadii.x ||
         axialDistFromB * axialDistFromB + radiusSq
             < vCylinderClipRadii.y * vCylinderClipRadii.y;
}

RaycastHit intersectRaycastPrimitive() {
  RaycastRay ray = getRaycastRayThroughFragment();
  highp vec3 endpointA = vCylinderBaseCircle.xyz;
  highp float radius = vCylinderBaseCircle.w;
  highp vec3 axisDirection = vCylinderAxis.xyz;
  highp float axisLength = vCylinderAxis.w;

  VectorSplit originSplit =
      splitAlongDirection(ray.origin - endpointA, axisDirection);
  VectorSplit directionSplit = splitAlongDirection(ray.direction, axisDirection);

  // Zero when the ray runs parallel to the axis, which never meets the lateral
  // surface. GLSL ES leaves 0.0 / 0.0 undefined, so reject it before the divide.
  highp float sinAngleToAxis = length(directionSplit.perpendicular);
  if (!(sinAngleToAxis > 0.0)) return raycastMiss();

  // Step 1. Across the axis the cylinder is a circle. The distance comes back in
  // that plane, so scale it onto the ray.
  RaycastCircleHit circleHit = intersectRaycastCircle(
      originSplit.perpendicular, directionSplit.perpendicular / sinAngleToAxis,
      radius);
  if (!circleHit.hit) return raycastMiss();
  highp float hitDist = circleHit.distAlongRay / sinAngleToAxis;

  // Step 2. Along the axis it is an interval.
  highp float axialDist =
      originSplit.parallelDist + hitDist * directionSplit.parallelDist;
  if (!(axialDist >= 0.0 && axialDist <= axisLength)) return raycastMiss();
  if (cylinderEndClipped(axialDist, radius)) return raycastMiss();
  highp vec3 surfacePoint = ray.origin + hitDist * ray.direction;
  // A zero length axis passes the test above only at axialDist 0.0, which is A.
  raycastCylinderAxialFraction = axisLength > 0.0 ? axialDist / axisLength : 0.0;

  // Open ends, so the circle normal holds everywhere. Caps would not.
  return makeRaycastHit(surfacePoint, circleHit.normal);
}
`);
}
