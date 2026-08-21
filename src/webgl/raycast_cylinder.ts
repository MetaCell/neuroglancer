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
 * @file Raycast open-ended cylinder drawn on a camera-facing quad; see
 * `raycast_primitive.ts` for the shared conventions.
 *
 * Adapted from Inigo Quilez's cylinder intersector
 * (https://iquilezles.org/articles/intersectors/,
 * https://www.shadertoy.com/view/4lcSRn), MIT licensed:
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
 *   - The vertex stage has no counterpart in the original, which ray-marches a
 *     full-screen quad. It bounds the cylinder with
 *     `emitRaycastAxialObbQuad` and hands the axis frame to the fragment stage in
 *     `vCylinderAxis`.
 *   - The quadratic becomes a ray/circle test in the plane perpendicular to a unit
 *     axis, in the same perpendicular-distance form as `raycast_sphere.ts`; see
 *     `intersectRaycastPrimitive` below.
 *   - The end caps are dropped (skeleton joints are covered by spheres) and
 *     endpoint clipping is added.
 *   - Returns depth and a lighting factor rather than a ray distance and normal.
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

  // Find two perpendicular radius vectors spanning the circular cross-section.
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
  RaycastRay ray = getRaycastEyeRay();
  highp vec3 axisDirection = vCylinderAxis.xyz;
  highp float axisLength = vCylinderAxis.w;

  // Split the ray about the unit axis. In the plane perpendicular to the axis the
  // cylinder is only a circle of vCylinderRadius centred on the axis, so the same
  // right-triangle test as raycast_sphere.ts finds the crossing; the axial parts
  // then say whether that crossing lies between the two endpoints.
  highp vec3 baseToOrigin = ray.origin - vCylinderEndpointA;
  highp float directionAlongAxis = dot(axisDirection, ray.direction);
  highp float originAlongAxis = dot(axisDirection, baseToOrigin);
  highp vec3 directionInPlane = ray.direction - directionAlongAxis * axisDirection;
  highp vec3 originInPlane = baseToOrigin - originAlongAxis * axisDirection;

  // ray.direction is a unit vector, so this length is the sine of the angle between
  // the ray and the axis. It scales ray distance into in-plane distance, and is zero
  // exactly when the ray runs parallel to the axis.
  highp float sinAngleToAxis = length(directionInPlane);
  highp vec3 inPlaneDirection = directionInPlane / sinAngleToAxis;
  highp float projectedDistance = dot(originInPlane, inPlaneDirection);
  highp vec3 perpendicular = originInPlane - projectedDistance * inPlaneDirection;
  highp float halfChordSquared =
      vCylinderRadius * vCylinderRadius - dot(perpendicular, perpendicular);

  // Positive-form guards throughout, so a ray parallel to the axis (a zero
  // sinAngleToAxis) and any NaN it produces miss rather than slipping through.
  if (!(halfChordSquared >= 0.0)) return raycastMiss();
  highp float halfChord = sqrt(halfChordSquared);
  // Only the near crossing is drawn, as in raycast_sphere.ts: a negative one means
  // the cylinder is behind us or we are inside it, and both are a miss.
  highp float hitDistance = (-projectedDistance - halfChord) / sinAngleToAxis;
  if (!(hitDistance >= 0.0)) return raycastMiss();

  highp float axialDistance = originAlongAxis + hitDistance * directionAlongAxis;
  if (!(axialDistance >= 0.0 && axialDistance <= axisLength)) return raycastMiss();
  highp vec3 surfacePoint = ray.origin + hitDistance * ray.direction;
  if (cylinderPointClipped(surfacePoint)) return raycastMiss();

  return makeRaycastHit(surfacePoint, originInPlane + hitDistance * directionInPlane);
}
`);
}
