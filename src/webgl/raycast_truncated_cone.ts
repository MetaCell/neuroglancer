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
 * @file Raycast truncated cone drawn on a camera-facing quad. The vertex stage
 * bounds the cone with a quad and the fragment stage returns depth and a lighting
 * factor. Symbols below say cone for brevity; the surface is always the truncated
 * one, and its ends are open.
 *
 * The radius is given at each end and runs linearly between them. Equal radii give
 * an exact cylinder, which is the common case. A cone sized for a constant
 * on-screen width needs the taper, because the far end of a receding cone sits at a
 * larger radius than the near end.
 *
 * Each end also takes a clip radius, which removes the part of the surface that a
 * primitive drawn at that end covers.
 */

import { defineRaycastAxialObbQuad } from "#src/webgl/raycast_primitive.js";
import type { ShaderBuilder } from "#src/webgl/shader.js";

export function defineRaycastConeShader(builder: ShaderBuilder) {
  defineRaycastAxialObbQuad(builder);
  builder.addVarying("highp vec3", "vConeEndpointA", "flat");
  // xyz: unit axis direction, w: axis length.
  builder.addVarying("highp vec4", "vConeAxis", "flat");
  // xy: surface radius at endpoint A and at endpoint B.
  // zw: clip radius at endpoint A and at endpoint B.
  builder.addVarying("highp vec4", "vConeEndRadii", "flat");
  builder.addVertexCode(`
void emitRaycastCone(highp vec3 endpointA, highp vec3 endpointB,
                         highp float radiusA, highp float radiusB,
                         highp float clipRadiusA, highp float clipRadiusB) {
  highp float widestRadius = max(radiusA, radiusB);
  // No radius, no surface to hit. A segment with both endpoints behind the eye
  // reaches this every frame. Positive form, so a non-finite radius culls too.
  if (!(widestRadius > 0.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vConeEndpointA = endpointA;
  vConeEndRadii = vec4(radiusA, radiusB, clipRadiusA, clipRadiusB);
  highp vec3 axisVector = endpointB - endpointA;
  highp float axisLength = length(axisVector);
  highp vec3 axisDirection = axisLength > 1e-6 ? axisVector / axisLength : vec3(0.0, 1.0, 0.0);
  vConeAxis = vec4(axisDirection, axisLength);

  // Two perpendicular radius vectors spanning the widest cross-section. The
  // scale comes last: a zero radius would otherwise leave the second cross
  // product normalising the zero vector, which GLSL ES leaves undefined.
  highp vec3 offAxisVector =
      abs(axisDirection.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
  highp vec3 unitRadiusA = normalize(cross(offAxisVector, axisDirection));
  // Already unit length, being the cross product of two perpendicular unit vectors.
  highp vec3 unitRadiusB = cross(axisDirection, unitRadiusA);
  emitRaycastAxialObbQuad(endpointA, endpointB,
                          unitRadiusA * widestRadius,
                          unitRadiusB * widestRadius);
}
`);
  builder.addFragmentCode(`
// Where the surface point falls between the endpoints, 0.0 at A and 1.0 at B.
// Only meaningful once intersectRaycastPrimitive has returned a hit.
highp float raycastConeAxialFraction = 0.0;

// A surface point sits one local radius from the axis, so its distance to an
// endpoint follows from the axial distance alone.
bool coneEndClipped(highp float axialDist, highp float radiusAtHit) {
  highp float axialDistFromB = axialDist - vConeAxis.w;
  highp float radiusSq = radiusAtHit * radiusAtHit;
  return axialDist * axialDist + radiusSq
             < vConeEndRadii.z * vConeEndRadii.z ||
         axialDistFromB * axialDistFromB + radiusSq
             < vConeEndRadii.w * vConeEndRadii.w;
}

// Across the axis the cone is a circle whose radius grows along the axis, so the
// in-plane test is a quadratic rather than the fixed-radius circle the sphere uses.
// Equal end radii leave the taper rate at zero, and this reduces to that circle.
RaycastHit intersectRaycastPrimitive() {
  RaycastRay ray = getRaycastRayThroughFragment();
  highp vec3 axisDirection = vConeAxis.xyz;
  highp float axisLength = vConeAxis.w;
  highp float radiusA = vConeEndRadii.x;
  highp float inverseAxisLength = axisLength > 0.0 ? 1.0 / axisLength : 0.0;
  // Radius added per unit along the axis. Zero for an exact cylinder.
  highp float taperRate = (vConeEndRadii.y - radiusA) * inverseAxisLength;

  VectorSplit originSplit =
      splitAlongDirection(ray.origin - vConeEndpointA, axisDirection);
  VectorSplit directionSplit = splitAlongDirection(ray.direction, axisDirection);
  highp float perpendicularSpeedSq =
      dot(directionSplit.perpendicular, directionSplit.perpendicular);
  // Radius added per unit along the ray.
  highp float radiusRate = taperRate * directionSplit.parallelDist;

  // Zero for a ray along the axis, which never meets the surface. Negative for a
  // ray running inside the taper angle, where the near crossing lies past the
  // apex. Positive form, so a non-finite value misses. This also guards the
  // divides below, since a positive value puts perpendicularSpeedSq above zero.
  highp float quadraticA = perpendicularSpeedSq - radiusRate * radiusRate;
  if (!(quadraticA > 0.0)) return raycastMiss();

  // Measured from the closest approach to the axis, so that the constant term is a
  // difference of two small numbers. Neuroglancer models can sit far from the
  // origin, and the unshifted form subtracts two large ones.
  highp float closestDist =
      -dot(originSplit.perpendicular, directionSplit.perpendicular)
      / perpendicularSpeedSq;
  highp vec3 perpendicularAtClosest =
      originSplit.perpendicular + closestDist * directionSplit.perpendicular;
  highp float radiusAtClosest = radiusA + taperRate *
      (originSplit.parallelDist + closestDist * directionSplit.parallelDist);

  QuadraticNearRoot root = nearQuadraticRoot(
      quadraticA,
      -radiusAtClosest * radiusRate,
      dot(perpendicularAtClosest, perpendicularAtClosest)
          - radiusAtClosest * radiusAtClosest);
  if (!root.exists) return raycastMiss();

  // The near crossing. Taking the far one would fill the view from inside.
  highp float hitDist = closestDist + root.value;
  if (!(hitDist >= 0.0)) return raycastMiss();

  // Along the axis the cone is an interval. That also holds the radius between the
  // two end radii, so a surface past a cone apex never draws.
  highp float axialDist =
      originSplit.parallelDist + hitDist * directionSplit.parallelDist;
  if (!(axialDist >= 0.0 && axialDist <= axisLength)) return raycastMiss();
  highp float radiusAtHit = radiusA + taperRate * axialDist;
  if (coneEndClipped(axialDist, radiusAtHit)) return raycastMiss();
  raycastConeAxialFraction = axialDist * inverseAxisLength;

  // The gradient of the surface equation. The axial term is what the taper adds,
  // and it vanishes for an exact cylinder, leaving the radial direction.
  highp vec3 perpendicularAtHit =
      originSplit.perpendicular + hitDist * directionSplit.perpendicular;
  return makeRaycastHit(
      ray.origin + hitDist * ray.direction,
      perpendicularAtHit - radiusAtHit * taperRate * axisDirection);
}
`);
}
