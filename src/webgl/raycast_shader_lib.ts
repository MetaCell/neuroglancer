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
 * @file General GLSL algebra for ray casting against a quadric surface.
 *
 * `nearQuadraticRoot`, and the way callers form the coefficients they pass it, are
 * adapted from Inigo Quilez's sphere intersector
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
 * One change. A caller measures its parameter from the ray's closest approach to
 * the surface's axis and subtracts the perpendicular distance from the radius,
 * rather than forming the original `c = dot(oc, oc) - r * r`. Neuroglancer models
 * can sit far from the origin, and the rearranged form never subtracts two large
 * numbers.
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

export const glsl_nearQuadraticRoot = `
struct QuadraticNearRoot {
  bool exists;
  highp float value;
};

// Smaller root of quadraticA * t^2 + 2 * quadraticB * t + quadraticC, for a
// quadraticA above zero. quadraticB is half the linear coefficient, which is the
// form a ray against a quadric produces and which keeps the discriminant free of a
// factor of four.
QuadraticNearRoot nearQuadraticRoot(highp float quadraticA, highp float quadraticB,
                                    highp float quadraticC) {
  highp float discriminant = quadraticB * quadraticB - quadraticA * quadraticC;
  QuadraticNearRoot root;
  // Positive form so that a NaN falls through to no root, and so that sqrt is
  // never reached with a negative argument. IEEE floats guarantee the NaN half,
  // GLSL ES does not, so that part is defence and not a promise.
  if (!(discriminant >= 0.0)) {
    root.exists = false;
    root.value = 0.0;
    return root;
  }
  root.exists = true;
  root.value = (-quadraticB - sqrt(discriminant)) / quadraticA;
  return root;
}
`;
