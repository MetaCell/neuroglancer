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
 * @file Small GLSL geometry helpers for ray casting.
 *
 * `intersectRaycastCircle` is adapted from Inigo Quilez's sphere intersector
 * (https://iquilezles.org/articles/intersectors/), MIT licensed:
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
 * The hit test subtracts the perpendicular distance from the radius rather than
 * forming the original `c = dot(oc, oc) - r * r`. Neuroglancer models can sit far
 * from the origin, and the rearranged form never subtracts two large numbers.
 */

export const glsl_splitAlongDirection = `
struct VectorSplit {
  highp float parallelDist;
  highp vec3 perpendicular;
};

VectorSplit splitAlongDirection(highp vec3 vectorToSplit, highp vec3 unitDirection) {
  VectorSplit split;
  split.parallelDist = dot(unitDirection, vectorToSplit);
  split.perpendicular = vectorToSplit - split.parallelDist * unitDirection;
  return split;
}
`;

export const glsl_intersectRaycastCircle = `
struct RaycastCircleHit {
  bool hit;
  // To the near crossing.
  highp float distAlongRay;
  // Hit point minus centre, not normalised.
  highp vec3 normal;
};

RaycastCircleHit raycastCircleMiss() {
  RaycastCircleHit hit;
  hit.hit = false;
  return hit;
}

// Also a sphere test, when the vectors are not confined to one plane.
RaycastCircleHit intersectRaycastCircle(highp vec3 centerToOrigin,
                                        highp vec3 unitDirection,
                                        highp float radius) {
  VectorSplit originSplit = splitAlongDirection(centerToOrigin, unitDirection);
  highp float radiusSq = radius * radius;
  highp float perpendicularDistSq =
      dot(originSplit.perpendicular, originSplit.perpendicular);

  // Positive form so that a NaN falls through to the miss. IEEE floats guarantee
  // that, GLSL ES does not, so this is defence and not a promise.
  if (!(perpendicularDistSq <= radiusSq)) return raycastCircleMiss();

  highp float halfChord = sqrt(radiusSq - perpendicularDistSq);
  // Taking the far crossing instead would fill the view when the camera clips
  // inside the geometry.
  highp float distAlongRay = -originSplit.parallelDist - halfChord;
  if (!(distAlongRay >= 0.0)) return raycastCircleMiss();

  RaycastCircleHit hit;
  hit.hit = true;
  hit.distAlongRay = distAlongRay;
  hit.normal = originSplit.perpendicular - halfChord * unitDirection;
  return hit;
}
`;
