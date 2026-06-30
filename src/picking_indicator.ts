/**
 * @license
 * Copyright 2024 Google Inc.
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

import { RefCounted } from "#src/util/disposable.js";
import type { GL } from "#src/webgl/context.js";
import { ShaderBuilder, type ShaderProgram } from "#src/webgl/shader.js";

const RADIUS_PIXELS = 14;

export class PickingIndicatorHelper extends RefCounted {
  private readonly shader: ShaderProgram;

  constructor(private readonly gl: GL) {
    super();
    const builder = new ShaderBuilder(gl);
    builder.addUniform("highp vec2", "uNDCPosition");
    builder.addUniform("highp vec2", "uSizeNDC");
    builder.addVarying("highp vec2", "vUV");
    builder.setVertexMain(`
vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0,
                   float((gl_VertexID >> 1) & 1) * 2.0 - 1.0);
vUV = corner;
gl_Position = vec4(uNDCPosition + corner * uSizeNDC, 0.0, 1.0);
`);
    builder.addOutputBuffer("vec4", "out_color", 0);
    builder.setFragmentMain(`
float dist = length(vUV);
if (dist > 1.0 || dist < 0.5) discard;
// White ring bordered by black on both sides for contrast on any background.
float isWhite = step(0.62, dist) * step(dist, 0.87);
out_color = vec4(vec3(isWhite), 0.92);
`);
    this.shader = this.registerDisposer(builder.build());
  }

  draw(
    ndcX: number,
    ndcY: number,
    logicalWidth: number,
    logicalHeight: number,
  ) {
    const { gl, shader } = this;
    shader.bind();
    gl.uniform2f(shader.uniform("uNDCPosition"), ndcX, ndcY);
    gl.uniform2f(
      shader.uniform("uSizeNDC"),
      RADIUS_PIXELS / logicalWidth,
      RADIUS_PIXELS / logicalHeight,
    );
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  static get(gl: GL) {
    return gl.memoize.get(
      "picking_indicator/PickingIndicatorHelper",
      () => new PickingIndicatorHelper(gl),
    );
  }
}
