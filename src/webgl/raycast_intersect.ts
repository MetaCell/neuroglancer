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
 * @file Ray intersection GLSL shared by the raycast primitives. Pure geometry:
 * nothing here refers to a uniform, to `RaycastRay`, or to `RaycastHit`.
 *
 * Naming follows the conventional letters for a ray/sphere test:
 *
 *   C    centre of the circle or sphere the ray is tested against
 *   R    its radius
 *   D    perpendicular distance from C to the ray
 *   H    near hit point, where the ray first meets the surface
 *   N    outward normal at H, equal to H - C
 *
 * `raycast_cylinder.ts` adds an axis and reuses the same test in the plane
 * perpendicular to it; see `glsl_splitAboutAxis`.
 */

// Both `raycast_sphere.ts` and `raycast_cylinder.ts` reduce to this one test.
// The sphere applies it in three dimensions; the cylinder applies it to the
// circular cross-section, with every vector confined to one plane.
export const glsl_intersectRaycastCircle = `
struct RaycastCircleHit {
  bool hit;
  // Distance along unitDirection from the ray origin to the near hit point H.
  highp float distanceAlongRay;
  // N = H - C, the offset from the centre to the hit point. For a sphere, and for
  // the lateral surface of a cylinder, this is the outward normal at H. It is not
  // a unit vector: the caller normalises after the normal transform, which a
  // non-uniform transform changes anyway, so normalising here would be discarded.
  highp vec3 offsetFromCenter;
};

RaycastCircleHit raycastCircleMiss() {
  RaycastCircleHit hit;
  hit.hit = false;
  return hit;
}

// Nearest crossing of the ray \`centerToOrigin + t * unitDirection\` with the sphere
// of radius R about the centre C, where centerToOrigin is the ray origin measured
// from C. unitDirection must have unit length. When every vector lies in one plane
// this is a ray/circle test instead; the algebra does not change.
//
// Only the near crossing at t >= 0 is reported. A negative one means the surface is
// behind us or the origin is inside it, and drawing the far surface then fills the
// view when the camera clips inside the geometry.
RaycastCircleHit intersectRaycastCircle(highp vec3 centerToOrigin,
                                        highp vec3 unitDirection,
                                        highp float radius) {
  // unitDirection has unit length, so this projection locates where the ray passes
  // closest to C: at t = -projectedDistance, offset from C by the perpendicular D.
  highp float projectedDistance = dot(centerToOrigin, unitDirection);
  highp vec3 perpendicular = centerToOrigin - projectedDistance * unitDirection;
  highp float radiusSquared = radius * radius;
  highp float perpendicularDistanceSquared = dot(perpendicular, perpendicular);

  // D <= R, the hit test; a tangent at D == R counts as a hit. Comparisons are in
  // positive form so that a non-finite value misses rather than slipping through.
  // Defence only: GLSL ES guarantees nothing about NaN.
  if (!(perpendicularDistanceSquared <= radiusSquared)) return raycastCircleMiss();

  // The half-chord is the third side of a right triangle with hypotenuse R and leg
  // D, so the two crossings are at -projectedDistance -/+ halfChord.
  highp float halfChord = sqrt(radiusSquared - perpendicularDistanceSquared);
  highp float distanceAlongRay = -projectedDistance - halfChord;
  if (!(distanceAlongRay >= 0.0)) return raycastCircleMiss();

  RaycastCircleHit hit;
  hit.hit = true;
  hit.distanceAlongRay = distanceAlongRay;
  // N = H - C: the perpendicular, walked back along the ray by the half-chord.
  // Built from the ray so that two large model coordinates never subtract.
  hit.offsetFromCenter = perpendicular - halfChord * unitDirection;
  return hit;
}
`;

// Separates a vector into the part along an axis and the part across it, which is
// what turns a cylinder into an independent circle problem and interval problem.
export const glsl_splitAboutAxis = `
struct AxialSplit {
  // Signed component along the unit axis.
  highp float alongAxis;
  // The remainder, which lies in the plane perpendicular to the axis.
  highp vec3 inPlane;
};

// unitAxis must have unit length.
AxialSplit splitAboutAxis(highp vec3 vectorToSplit, highp vec3 unitAxis) {
  AxialSplit split;
  split.alongAxis = dot(unitAxis, vectorToSplit);
  split.inPlane = vectorToSplit - split.alongAxis * unitAxis;
  return split;
}
`;
