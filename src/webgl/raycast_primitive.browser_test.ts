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

import { describe, it } from "vitest";
import { defineRaycastCylinderShader } from "#src/webgl/raycast_cylinder.js";
import { glsl_raycastFragmentSetup } from "#src/webgl/raycast_primitive.js";
import { defineRaycastSphereShader } from "#src/webgl/raycast_sphere.js";
import { ShaderBuilder } from "#src/webgl/shader.js";
import { webglTest } from "#src/webgl/testing.js";

function buildShader(
  definePrimitive: (builder: ShaderBuilder) => void,
  emitPrimitive: string,
) {
  webglTest((gl) => {
    const builder = new ShaderBuilder(gl);
    builder.addOutputBuffer("vec4", "out_color", 0);
    definePrimitive(builder);
    builder.setVertexMain(emitPrimitive);
    // Mirrors how a consumer emits: from a helper function, which can only see
    // the published globals and not main's locals.
    builder.addFragmentCode(`
void emitShaded() {
  out_color = vec4(vec3(raycastLightingFactor), raycastSurfaceDepth);
}
`);
    builder.setFragmentMain(glsl_raycastFragmentSetup + "emitShaded();\n");
    builder.build().dispose();
  });
}

describe("raycast primitives", () => {
  it("compiles the sphere shader", () => {
    buildShader(
      defineRaycastSphereShader,
      `emitRaycastSphere(vec3(0.0), getRaycastModelRadiusForPixels(vec3(0.0), 5.0));`,
    );
  });

  it("compiles the cylinder shader", () => {
    buildShader(
      defineRaycastCylinderShader,
      `emitRaycastCylinder(vec3(0.0), vec3(0.0, 1.0, 0.0),
                    getRaycastModelRadiusForPixels(vec3(0.0), 2.0), 1.0, 1.0);`,
    );
  });
});
