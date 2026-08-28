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
  builder.addVarying("highp vec3", "vCylinderEndpointA", "flat");
  builder.addVarying("highp vec3", "vCylinderEndpointB", "flat");
  // xyz: unit axis direction, w: axis length.
  builder.addVarying("highp vec4", "vCylinderAxis", "flat");
  builder.addVarying("highp float", "vCylinderRadius", "flat");
  builder.addVarying("highp float", "vCylinderClipRadiusA", "flat");
  builder.addVarying("highp float", "vCylinderClipRadiusB", "flat");
  builder.addVertexCode(`
void emitRaycastCylinder(highp vec3 endpointA, highp vec3 endpointB,
                         highp float radius,
                         highp float clipRadiusA, highp float clipRadiusB) {
  vCylinderEndpointA = endpointA;
  vCylinderEndpointB = endpointB;
  vCylinderRadius = radius;
  vCylinderClipRadiusA = clipRadiusA;
  vCylinderClipRadiusB = clipRadiusB;
  highp vec3 axisVector = endpointB - endpointA;
  highp float axisLength = length(axisVector);
  highp vec3 axisDirection = axisLength > 1e-6 ? axisVector / axisLength : vec3(0.0, 1.0, 0.0);
  vCylinderAxis = vec4(axisDirection, axisLength);

  // Two perpendicular radius vectors spanning the circular cross-section.
  highp vec3 offAxisVector =
      abs(axisDirection.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
  highp vec3 radiusVectorA = normalize(cross(offAxisVector, axisDirection)) * radius;
  highp vec3 radiusVectorB = normalize(cross(axisDirection, radiusVectorA)) * radius;
  emitRaycastAxialObbQuad(endpointA, endpointB, radiusVectorA, radiusVectorB);
}
`);
  builder.addFragmentCode(`
bool cylinderPointClipped(highp vec3 surfacePoint) {
  highp vec3 offsetA = surfacePoint - vCylinderEndpointA;
  highp vec3 offsetB = surfacePoint - vCylinderEndpointB;
  return dot(offsetA, offsetA) < vCylinderClipRadiusA * vCylinderClipRadiusA ||
         dot(offsetB, offsetB) < vCylinderClipRadiusB * vCylinderClipRadiusB;
}

RaycastHit intersectRaycastPrimitive() {
  RaycastRay ray = getModelRayThroughFragment();
  highp vec3 axisDirection = vCylinderAxis.xyz;
  highp float axisLength = vCylinderAxis.w;

  VectorSplit originSplit =
      splitAlongDirection(ray.origin - vCylinderEndpointA, axisDirection);
  VectorSplit directionSplit = splitAlongDirection(ray.direction, axisDirection);

  // Zero when the ray runs parallel to the axis, which never meets the lateral
  // surface. GLSL ES leaves 0.0 / 0.0 undefined, so reject it before the divide.
  highp float sinAngleToAxis = length(directionSplit.perpendicular);
  if (!(sinAngleToAxis > 0.0)) return raycastMiss();

  // Step 1. Across the axis the cylinder is a circle. The distance comes back in
  // that plane, so scale it onto the ray.
  RaycastCircleHit circleHit = intersectRaycastCircle(
      originSplit.perpendicular, directionSplit.perpendicular / sinAngleToAxis,
      vCylinderRadius);
  if (!circleHit.hit) return raycastMiss();
  highp float hitDist = circleHit.distAlongRay / sinAngleToAxis;

  // Step 2. Along the axis it is an interval.
  highp float axialDist =
      originSplit.parallelDist + hitDist * directionSplit.parallelDist;
  if (!(axialDist >= 0.0 && axialDist <= axisLength)) return raycastMiss();
  highp vec3 surfacePoint = ray.origin + hitDist * ray.direction;
  if (cylinderPointClipped(surfacePoint)) return raycastMiss();

  // Open ends, so the circle normal holds everywhere. Caps would not.
  return makeRaycastHit(surfacePoint, circleHit.normal);
}
`);
}
