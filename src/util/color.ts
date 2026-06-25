/**
 * @license
 * Copyright 2018 Google Inc.
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
 * @file Facilities for converting between strings and RGB/RGBA colors.
 */

import { WatchableValue } from "#src/trackable_value.js";
import { float32ToString } from "#src/util/float32_to_string.js";
import { vec3, vec4 } from "#src/util/geom.js";
import { hexEncodeByte } from "#src/util/hex.js";

/**
 * Parse the serialization of a color.
 *
 * This is based on the definition here:
 * https://html.spec.whatwg.org/multipage/canvas.html#serialisation-of-a-color
 */
export function parseColorSerialization(x: string) {
  const rgbaPattern =
    /^rgba\(([0-9]+), ([0-9]+), ([0-9]+), (0(?:\.[0-9]+)?)\)$/;
  {
    const m = x.match(rgbaPattern);
    if (m !== null) {
      return [
        parseInt(m[1], 10),
        parseInt(m[2], 10),
        parseInt(m[3], 10),
        parseFloat(m[4]),
      ];
    }
  }
  const hexPattern = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/;
  {
    const m = x.match(hexPattern);
    if (m !== null) {
      return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16), 1.0];
    }
  }
  throw new Error(`Invalid serialized color: ${JSON.stringify(x)}.`);
}

export function parseRGBAColorSpecification(x: any) {
  try {
    if (typeof x !== "string") {
      throw new Error(`Expected string, but received ${JSON.stringify(x)}.`);
    }
    const context = document.createElement("canvas").getContext("2d")!;
    context.fillStyle = x;
    const result = parseColorSerialization(context.fillStyle);
    return vec4.fromValues(
      result[0] / 255,
      result[1] / 255,
      result[2] / 255,
      result[3],
    );
  } catch (parseError) {
    throw new Error(
      `Failed to parse color specification: ${parseError.message}`,
    );
  }
}

export function parseRGBColorSpecification(x: any) {
  const result = parseRGBAColorSpecification(x);
  return <vec3>result.subarray(0, 3);
}

/**
 * Returns an integer formed by concatenating the channels of the input color vector.
 * Each channel is clamped to the range [0.0, 1.0] before being converted to 8 bits.
 * An RGB color is packed into 24 bits, and a RGBA into 32 bits.
 * Packed colors are stored in (AA)GGBBRR order.
 */
export function packColor(x: vec3 | vec4): number {
  const size = x[3] === undefined ? 3 : 4;
  let result = 0;
  for (let i = 0; i < size; i++) {
    // The ">>> 0" ensures an unsigned value.
    result =
      ((result << 8) >>> 0) +
      Math.min(255, Math.max(0, Math.round(x[size - 1 - i] * 255)));
  }
  return result;
}

/**
 * Extend a packed RGB color to a packed RGBA color with full opacity.
 */
export function extendPackedRGB(value: number): number {
  return (0xff000000 | value) >>> 0;
}

export function unpackRGB(value: number) {
  return vec3.fromValues(
    ((value >>> 0) & 0xff) / 255,
    ((value >>> 8) & 0xff) / 255,
    ((value >>> 16) & 0xff) / 255,
  );
}

export function unpackRGBA(value: number) {
  return vec4.fromValues(
    ((value >>> 0) & 0xff) / 255,
    ((value >>> 8) & 0xff) / 255,
    ((value >>> 16) & 0xff) / 255,
    ((value >>> 24) & 0xff) / 255,
  );
}

export function serializeColor(x: vec3 | vec4) {
  if (x[3] === undefined || x[3] === 1) {
    let result = "#";
    for (let i = 0; i < 3; ++i) {
      result += hexEncodeByte(
        Math.min(255, Math.max(0, Math.round(x[i] * 255))),
      );
    }
    return result;
  }
  let result = "rgba(";
  for (let i = 0; i < 3; ++i) {
    if (i !== 0) {
      result += ", ";
    }
    result += Math.min(255, Math.max(0, Math.round(x[i] * 255)));
  }
  result += `, ${float32ToString(x[3])})`;
  return result;
}

// Converts an sRGB color component to the gamma-expanded ("linear") value.
export function srgbGammaExpand(value: number) {
  return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

// Computes the relative luminance according to Web Content Accessibility Guidelines (WCAG) 2.0
//
// https://www.w3.org/TR/WCAG20/#relativeluminancedef
//
// @param color sRGB color
export function getRelativeLuminance(color: ArrayLike<number>) {
  return (
    0.2126 * srgbGammaExpand(color[0]) +
    0.7152 * srgbGammaExpand(color[1]) +
    0.0722 * srgbGammaExpand(color[2])
  );
}

export function getContrastRatio(
  colorA: ArrayLike<number>,
  colorB: ArrayLike<number>,
) {
  const luminanceA = getRelativeLuminance(colorA);
  const luminanceB = getRelativeLuminance(colorB);
  const darker = Math.min(luminanceA, luminanceB);
  const lighter = Math.max(luminanceA, luminanceB);
  return (lighter + 0.05) / (darker + 0.05);
}

// Determines whether a white background would provide higher contrast than a black background for
// the given foreground color.
//
// This is determined according to the Web Content Accessibility Guidelines (WCAG) 2.0:
// https://www.w3.org/TR/WCAG20/#contrast-ratiodef
//
// https://stackoverflow.com/a/3943023
export function useWhiteBackground(foregroundColor: vec3 | vec4) {
  return getRelativeLuminance(foregroundColor) <= 0.179;
}

// Two disjoint palettes drive the node outline highlights. For a given segment
// fill color, an outline is the palette entry with the highest contrast against
// that segment. The hovered and selected colors are each computed independently
// from their own palette, so each is fully determined by the segment color alone
// (the same segment always yields the same hovered color and the same selected
// color), and the two never collide since the palettes are disjoint.

// Vivid, saturated colors for the hovered node -- the actively pointed-at node,
// drawn to stand out. Spans hue and luminance (white through blue) so a
// high-contrast option exists for any segment color.
const HOVERED_NODE_HIGHLIGHT_COLORS: readonly vec3[] = [
  vec3.fromValues(1.0, 1.0, 1.0), // white
  vec3.fromValues(1.0, 0.95, 0.0), // yellow
  // vec3.fromValues(0.0, 0.95, 1.0), // cyan
  // vec3.fromValues(0.1, 1.0, 0.25), // green
  // vec3.fromValues(1.0, 0.55, 0.0), // orange
  // vec3.fromValues(1.0, 0.1, 0.65), // pink
  vec3.fromValues(1.0, 0.12, 0.12), // red
  // vec3.fromValues(0.2, 0.45, 1.0), // blue
];

// Muted, lower-chroma colors for the selected (pinned) node -- a calmer,
// persistent highlight. Spans hue and luminance like the hovered set so a
// reasonable-contrast option exists for any segment color.
const SELECTED_NODE_HIGHLIGHT_COLORS: readonly vec3[] = [
  vec3.fromValues(0.1, 0.1, 0.1), // near-black
  vec3.fromValues(0.7, 0.67, 0.6), // stone (light warm gray)
  vec3.fromValues(0.5, 0.45, 0.15), // olive
  // vec3.fromValues(0.15, 0.42, 0.42), // teal
  // vec3.fromValues(0.5, 0.18, 0.18), // maroon
  // vec3.fromValues(0.25, 0.3, 0.5), // slate blue
  // vec3.fromValues(0.42, 0.22, 0.45), // plum
  // vec3.fromValues(0.25, 0.42, 0.22), // moss
  // vec3.fromValues(0.5, 0.38, 0.2), // tan
];

// Returns the palette color with the highest contrast against `sourceColor`.
function pickHighestContrastColor(
  palette: readonly vec3[],
  sourceColor: ArrayLike<number>,
): vec3 {
  let bestColor = palette[0];
  let bestContrast = -1;
  for (const candidate of palette) {
    const contrast = getContrastRatio(candidate, sourceColor);
    if (contrast > bestContrast) {
      bestContrast = contrast;
      bestColor = candidate;
    }
  }
  return bestColor;
}

// Writes into `out` the vivid hovered-node outline color with the highest
// contrast against `sourceColor`.
export function computeHoveredNodeHighlightColor<T extends Float32Array>(
  out: T,
  sourceColor: ArrayLike<number>,
): T {
  out.set(pickHighestContrastColor(HOVERED_NODE_HIGHLIGHT_COLORS, sourceColor));
  return out;
}

// Writes into `out` the muted selected-node outline color with the highest
// contrast against `sourceColor`.
export function computeSelectedNodeHighlightColor<T extends Float32Array>(
  out: T,
  sourceColor: ArrayLike<number>,
): T {
  out.set(
    pickHighestContrastColor(SELECTED_NODE_HIGHLIGHT_COLORS, sourceColor),
  );
  return out;
}

export class TrackableRGB extends WatchableValue<vec3> {
  constructor(public defaultValue: vec3) {
    super(vec3.clone(defaultValue));
  }
  toString() {
    return serializeColor(this.value);
  }
  toJSON() {
    if (vec3.equals(this.value, this.defaultValue)) {
      return undefined;
    }
    return serializeColor(this.value);
  }
  reset() {
    this.value = vec3.clone(this.defaultValue);
  }
  restoreState(x: any) {
    if (x === undefined) {
      this.reset();
      return;
    }
    const { value } = this;
    const newValue = parseRGBColorSpecification(x);
    if (!vec3.equals(value, newValue)) {
      this.value = newValue;
    }
  }
}

export class TrackableOptionalRGB extends WatchableValue<vec3 | undefined> {
  constructor() {
    super(undefined);
  }
  toJSON() {
    const { value } = this;
    if (value === undefined) return undefined;
    return serializeColor(value);
  }
  reset() {
    this.value = undefined;
  }
  restoreState(x: any) {
    if (x === undefined) {
      this.reset();
      return;
    }
    const { value } = this;
    const newValue = parseRGBColorSpecification(x);
    if (value === undefined || !vec3.equals(value, newValue)) {
      this.value = newValue;
    }
  }
}

export function createSteppedCssGradient(colors: string[], conic = false) {
  const numSteps = colors.length;
  if (conic) {
    const stepSize = 360.0 / numSteps;
    return `conic-gradient(${colors
      .map((color, i) => {
        const start = Math.round(i * stepSize);
        const end = Math.round((i + 1) * stepSize);
        return `${color} ${start}deg, ${color} ${end}deg`;
      })
      .join(", ")})`;
  }
  const stepSize = 100.0 / numSteps;
  return `linear-gradient(to right, ${colors
    .map((color, i) => {
      const start = Math.round(i * stepSize);
      const end = Math.round((i + 1) * stepSize);
      return `${color} ${start}%, ${color} ${end}%`;
    })
    .join(", ")})`;
}
