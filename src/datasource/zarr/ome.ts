/**
 * @license
 * Copyright 2022 Google Inc.
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

import type { CoordinateSpace } from "#src/coordinate_transform.js";
import { makeCoordinateSpace } from "#src/coordinate_transform.js";
import type {
  SingleChannelMetadata,
  ChannelMetadata,
} from "#src/datasource/index.js";
import {
  joinBaseUrlAndPath,
  kvstoreEnsureDirectoryPipelineUrl,
} from "#src/kvstore/url.js";
import { parseRGBColorSpecification } from "#src/util/color.js";
import {
  parseArray,
  parseFixedLengthArray,
  verifyBoolean,
  verifyFiniteFloat,
  verifyFinitePositiveFloat,
  verifyObject,
  verifyObjectProperty,
  verifyOptionalObjectProperty,
  verifyString,
} from "#src/util/json.js";
import { clampToInterval } from "#src/util/lerp.js";
import * as matrix from "#src/util/matrix.js";
import { allSiPrefixes } from "#src/util/si_units.js";

export interface OmeMultiscaleScale {
  url: string;
  transform: Float64Array;
}

export interface OmeMultiscaleMetadata {
  scales: OmeMultiscaleScale[];
  coordinateSpace: CoordinateSpace;
}

export interface OmeMetadata {
  multiscale: OmeMultiscaleMetadata;
  channels: ChannelMetadata | undefined;
}

const SUPPORTED_OME_MULTISCALE_VERSIONS = new Set(["0.4", "0.5-dev", "0.5", "0.6.dev1", "0.6-dev2"]);

const OME_UNITS = new Map<string, { unit: string; scale: number }>([
  ["angstrom", { unit: "m", scale: 1e-10 }],
  ["foot", { unit: "m", scale: 0.3048 }],
  ["inch", { unit: "m", scale: 0.0254 }],
  ["mile", { unit: "m", scale: 1609.34 }],
  // eslint-disable-next-line no-loss-of-precision
  ["parsec", { unit: "m", scale: 3.0856775814913673e16 }],
  ["yard", { unit: "m", scale: 0.9144 }],
  ["minute", { unit: "s", scale: 60 }],
  ["hour", { unit: "s", scale: 60 * 60 }],
  ["day", { unit: "s", scale: 60 * 60 * 24 }],
]);

for (const unit of ["meter", "second"]) {
  for (const siPrefix of allSiPrefixes) {
    const { longPrefix } = siPrefix;
    if (longPrefix === undefined) continue;
    OME_UNITS.set(`${longPrefix}${unit}`, {
      unit: unit[0],
      scale: 10 ** siPrefix.exponent,
    });
  }
}

interface Axis {
  name: string;
  unit: string;
  scale: number;
  type: string | undefined;
  discrete?: boolean;
  longName?: string;
}

function parseOmeroChannel(omeroChannel: unknown): SingleChannelMetadata {
  verifyObject(omeroChannel);

  const getProp = <T>(
    key: string,
    verifier: (value: unknown) => T,
  ): T | undefined => verifyOptionalObjectProperty(omeroChannel, key, verifier);
  const inputWindow = getProp("window", verifyObject);
  const getWindowProp = <T>(
    key: string,
    verifier: (value: unknown) => T,
  ): T | undefined =>
    inputWindow
      ? verifyOptionalObjectProperty(inputWindow, key, verifier)
      : undefined;

  const active = getProp("active", verifyBoolean);
  const coefficient = getProp("coefficient", verifyFiniteFloat);
  let colorString = getProp("color", verifyString);
  // If six hex digits, needs the # in front of the hex color
  if (colorString && /^[0-9a-f]{6}$/i.test(colorString)) {
    colorString = `#${colorString}`;
  }
  const color = parseRGBColorSpecification(colorString);
  const inverted = getProp("inverted", verifyBoolean);
  const label = getProp("label", verifyString);

  const windowMin = getWindowProp("min", verifyFiniteFloat);
  const windowMax = getWindowProp("max", verifyFiniteFloat);
  const windowStart = getWindowProp("start", verifyFiniteFloat);
  const windowEnd = getWindowProp("end", verifyFiniteFloat);

  const window =
    windowMin !== undefined && windowMax !== undefined
      ? ([windowMin, windowMax] as [number, number])
      : undefined;

  const range =
    windowStart !== undefined && windowEnd !== undefined
      ? inverted
        ? ([windowEnd, windowStart] as [number, number])
        : ([windowStart, windowEnd] as [number, number])
      : undefined;
  // If there is a window, then clamp the range to the window.
  if (window !== undefined && range !== undefined) {
    range[0] = clampToInterval(window, range[0]) as number;
    range[1] = clampToInterval(window, range[1]) as number;
  }

  return {
    active,
    label,
    color,
    coefficient,
    range,
    window,
  };
}

function parseOmeroMetadata(omero: unknown): ChannelMetadata {
  verifyObject(omero);
  const name = verifyOptionalObjectProperty(omero, "name", verifyString);
  const channels = verifyObjectProperty(omero, "channels", (x) =>
    parseArray(x, parseOmeroChannel),
  );

  return { name, channels };
}

function parseOmeAxis(axis: unknown): Axis {
  verifyObject(axis);
  const name = verifyObjectProperty(axis, "name", verifyString);
  const type = verifyOptionalObjectProperty(axis, "type", verifyString);
  const discrete = verifyOptionalObjectProperty(axis, "discrete", verifyBoolean);
  const longName = verifyOptionalObjectProperty(axis, "longName", verifyString);
  const parsedUnit = verifyOptionalObjectProperty(
    axis,
    "unit",
    (unit) => {
      const x = OME_UNITS.get(unit);
      if (x === undefined) {
        throw new Error(`Unsupported unit: ${JSON.stringify(unit)}`);
      }
      return x;
    },
    { unit: "", scale: 1 },
  );
  return { name, unit: parsedUnit.unit, scale: parsedUnit.scale, type, discrete, longName };
}

function parseOmeAxes(axes: unknown): CoordinateSpace {
  const parsedAxes = parseArray(axes, parseOmeAxis);
  return makeCoordinateSpace({
    names: parsedAxes.map((axis) => {
      const { name, type } = axis;
      if (type === "channel") {
        return `${name}'`;
      }
      return name;
    }),
    scales: Float64Array.from(parsedAxes, (axis) => axis.scale),
    units: parsedAxes.map((axis) => axis.unit),
  });
}

// Parse a single coordinate system (0.6 format)
function parseCoordinateSystem(coordinateSystem: unknown): { name: string; coordinateSpace: CoordinateSpace } {
  verifyObject(coordinateSystem);
  const name = verifyObjectProperty(coordinateSystem, "name", verifyString);
  const axes = verifyObjectProperty(coordinateSystem, "axes", parseOmeAxes);
  return { name, coordinateSpace: axes };
}

// Parse coordinateSystems array (0.6 format)
function parseCoordinateSystems(coordinateSystems: unknown): Map<string, CoordinateSpace> {
  const systems = parseArray(coordinateSystems, parseCoordinateSystem);
  const map = new Map<string, CoordinateSpace>();
  for (const system of systems) {
    if (map.has(system.name)) {
      throw new Error(`Duplicate coordinate system name: ${JSON.stringify(system.name)}`);
    }
    map.set(system.name, system.coordinateSpace);
  }
  return map;
}

function parseScaleTransform(rank: number, obj: unknown) {
  const scales = verifyObjectProperty(obj, "scale", (values) =>
    parseFixedLengthArray(
      new Float64Array(rank),
      values,
      verifyFinitePositiveFloat,
    ),
  );
  return matrix.createHomogeneousScaleMatrix(Float64Array, scales);
}

function parseIdentityTransform(rank: number, obj: unknown) {
  obj;
  return matrix.createIdentity(Float64Array, rank + 1);
}

function parseTranslationTransform(rank: number, obj: unknown) {
  const translation = verifyObjectProperty(obj, "translation", (values) =>
    parseFixedLengthArray(new Float64Array(rank), values, verifyFiniteFloat),
  );
  return matrix.createHomogeneousTranslationMatrix(Float64Array, translation);
}

function parseMapAxisTransform(rank: number, obj: unknown): Float64Array<ArrayBuffer> {
  const mapAxis = verifyObjectProperty(obj, "mapAxis", (values) =>
    parseFixedLengthArray(new Int32Array(rank), values, (x) => {
      if (typeof x !== "number" || !Number.isInteger(x) || x < 0 || x >= rank) {
        throw new Error(`Invalid axis index: ${x}`);
      }
      return x;
    }),
  );
  
  // Verify each index appears exactly once
  const seen = new Set<number>();
  for (const idx of mapAxis) {
    if (seen.has(idx)) {
      throw new Error(`Duplicate axis index in mapAxis: ${idx}`);
    }
    seen.add(idx);
  }
  
  // Create permutation matrix
  const mat = new Float64Array((rank + 1) ** 2) as Float64Array<ArrayBuffer>;
  mat[(rank + 1) ** 2 - 1] = 1; // Bottom-right corner = 1
  for (let outAxis = 0; outAxis < rank; ++outAxis) {
    const inAxis = mapAxis[outAxis];
    mat[outAxis * (rank + 1) + inAxis] = 1;
  }
  return mat;
}

function parseAffineTransform(rank: number, obj: unknown): Float64Array<ArrayBuffer> {
  verifyObject(obj);
  
  // Check if affine is provided inline
  const affineMatrix = verifyOptionalObjectProperty(obj, "affine", (values) => {
    if (!Array.isArray(values)) {
      throw new Error("affine field must be a 2D array");
    }
    // Values should be MxN+1 where M can vary, but we need rank dimensions input
    const result = new Float64Array((rank + 1) ** 2) as Float64Array<ArrayBuffer>;
    result[(rank + 1) ** 2 - 1] = 1; // Homogeneous coordinate
    
    if (values.length === 0 || values.length > rank) {
      throw new Error(`affine matrix must have between 1 and ${rank} rows`);
    }
    
    for (let row = 0; row < values.length; ++row) {
      if (!Array.isArray(values[row]) || values[row].length !== rank + 1) {
        throw new Error(`affine matrix row ${row} must have ${rank + 1} columns`);
      }
      for (let col = 0; col < rank + 1; ++col) {
        const val = verifyFiniteFloat(values[row][col]);
        result[row * (rank + 1) + col] = val;
      }
    }
    
    return result;
  });
  
  if (affineMatrix !== undefined) {
    return affineMatrix;
  }
  
  // TODO: Handle path-based affine loading if needed
  const path = verifyOptionalObjectProperty(obj, "path", verifyString);
  if (path !== undefined) {
    throw new Error("path-based affine transforms are not yet supported");
  }
  
  throw new Error("affine transform must specify either 'affine' or 'path'");
}

function parseRotationTransform(rank: number, obj: unknown): Float64Array<ArrayBuffer> {
  verifyObject(obj);
  
  // Check if rotation is provided inline
  const rotationMatrix = verifyOptionalObjectProperty(obj, "rotation", (values) => {
    if (!Array.isArray(values)) {
      throw new Error("rotation field must be a 2D array");
    }
    
    if (values.length !== rank) {
      throw new Error(`rotation matrix must be ${rank}x${rank}`);
    }
    
    const linearPart = new Float64Array(rank * rank);
    for (let row = 0; row < rank; ++row) {
      if (!Array.isArray(values[row]) || values[row].length !== rank) {
        throw new Error(`rotation matrix row ${row} must have ${rank} columns`);
      }
      for (let col = 0; col < rank; ++col) {
        linearPart[row * rank + col] = verifyFiniteFloat(values[row][col]);
      }
    }
    
    // Convert to homogeneous coordinates
    const result = new Float64Array((rank + 1) ** 2) as Float64Array<ArrayBuffer>;
    for (let row = 0; row < rank; ++row) {
      for (let col = 0; col < rank; ++col) {
        result[row * (rank + 1) + col] = linearPart[row * rank + col];
      }
    }
    result[(rank + 1) ** 2 - 1] = 1; // Homogeneous coordinate
    
    return result;
  });
  
  if (rotationMatrix !== undefined) {
    return rotationMatrix;
  }
  
  // TODO: Handle path-based rotation loading if needed
  const path = verifyOptionalObjectProperty(obj, "path", verifyString);
  if (path !== undefined) {
    throw new Error("path-based rotation transforms are not yet supported");
  }
  
  throw new Error("rotation transform must specify either 'rotation' or 'path'");
}

function parseSequenceTransform(rank: number, obj: unknown): Float64Array<ArrayBuffer> {
  const transformations = verifyObjectProperty(obj, "transformations", (values) => {
    if (!Array.isArray(values) || values.length === 0) {
      throw new Error("sequence transformations must be a non-empty array");
    }
    return values;
  });
  
  let result = matrix.createIdentity(Float64Array, rank + 1);
  
  for (const transformJson of transformations) {
    const transform = parseOmeCoordinateTransform(rank, transformJson);
    result = matrix.multiply(
      new Float64Array((rank + 1) ** 2) as Float64Array<ArrayBuffer>,
      rank + 1,
      transform,
      rank + 1,
      result,
      rank + 1,
      rank + 1,
      rank + 1,
      rank + 1,
    );
  }
  
  return result;
}

const coordinateTransformParsers = new Map([
  ["scale", parseScaleTransform],
  ["identity", parseIdentityTransform],
  ["translation", parseTranslationTransform],
  ["mapAxis", parseMapAxisTransform],
  ["affine", parseAffineTransform],
  ["rotation", parseRotationTransform],
  ["sequence", parseSequenceTransform],
]);

function parseOmeCoordinateTransform(
  rank: number,
  transformJson: unknown,
): Float64Array<ArrayBuffer> {
  verifyObject(transformJson);
  
  // Note: In OME-Zarr 0.6, transforms may have "input" and "output" fields
  // that reference coordinate system names. These are currently ignored
  // since we only support a single coordinate system per multiscale.
  // Future enhancements could validate these fields against coordinateSystems.
  
  const transformType = verifyObjectProperty(
    transformJson,
    "type",
    verifyString,
  );
  const parser = coordinateTransformParsers.get(transformType);
  if (parser === undefined) {
    throw new Error(
      `Unsupported coordinate transform type: ${JSON.stringify(transformType)}`,
    );
  }
  return parser(rank, transformJson);
}

function parseOmeCoordinateTransforms(
  rank: number,
  transforms: unknown,
): Float64Array {
  let transform = matrix.createIdentity(Float64Array, rank + 1);
  if (transforms === undefined) return transform;
  parseArray(transforms, (transformJson) => {
    const newTransform = parseOmeCoordinateTransform(rank, transformJson);
    transform = matrix.multiply(
      new Float64Array(transform.length) as Float64Array<ArrayBuffer>,
      rank + 1,
      newTransform,
      rank + 1,
      transform,
      rank + 1,
      rank + 1,
      rank + 1,
      rank + 1,
    );
  });
  return transform;
}

function parseMultiscaleScale(
  rank: number,
  url: string,
  obj: unknown,
): OmeMultiscaleScale {
  const path = verifyObjectProperty(obj, "path", verifyString);
  const transform = verifyObjectProperty(
    obj,
    "coordinateTransformations",
    (x) => parseOmeCoordinateTransforms(rank, x),
  );
  const scaleUrl = kvstoreEnsureDirectoryPipelineUrl(
    joinBaseUrlAndPath(url, path),
  );
  return { url: scaleUrl, transform };
}

function parseOmeMultiscale(
  url: string,
  multiscale: unknown,
): OmeMultiscaleMetadata {
  verifyObject(multiscale);
  
  // Try to get coordinate space - support both 0.6 (coordinateSystems) and pre-0.6 (axes) formats
  let coordinateSpace: CoordinateSpace;
  
  // Check for coordinateSystems (0.6 format)
  const coordinateSystems = verifyOptionalObjectProperty(
    multiscale,
    "coordinateSystems",
    parseCoordinateSystems,
  );
  
  if (coordinateSystems !== undefined) {
    // 0.6 format: use the first coordinate system
    // In the future, we might want to support multiple coordinate systems
    const firstSystem = coordinateSystems.values().next().value;
    if (firstSystem === undefined) {
      throw new Error("coordinateSystems array must contain at least one coordinate system");
    }
    coordinateSpace = firstSystem;
  } else {
    // Pre-0.6 format: use axes directly
    coordinateSpace = verifyObjectProperty(
      multiscale,
      "axes",
      parseOmeAxes,
    );
  }
  
  const rank = coordinateSpace.rank;
  const transform = verifyObjectProperty(
    multiscale,
    "coordinateTransformations",
    (x) => parseOmeCoordinateTransforms(rank, x),
  );
  const scales = verifyObjectProperty(multiscale, "datasets", (obj) =>
    parseArray(obj, (x) => {
      const scale = parseMultiscaleScale(rank, url, x);
      scale.transform = matrix.multiply(
        new Float64Array((rank + 1) ** 2) as Float64Array<ArrayBuffer>,
        rank + 1,
        transform,
        rank + 1,
        scale.transform,
        rank + 1,
        rank + 1,
        rank + 1,
        rank + 1,
      );
      return scale;
    }),
  );
  if (scales.length === 0) {
    throw new Error("At least one scale must be specified");
  }

  const baseTransform = scales[0].transform;
  
  // Compute the scale factors from the base transform as the length of each basis vector
  // (column norm for each dimension)
  const baseScales = new Float64Array(rank);
  for (let col = 0; col < rank; ++col) {
    let normSq = 0;
    for (let row = 0; row < rank; ++row) {
      const val = baseTransform[row * (rank + 1) + col];
      normSq += val * val;
    }
    baseScales[col] = Math.sqrt(normSq);
    coordinateSpace.scales[col] *= baseScales[col];
  }

  // Compute inverse of base transform for normalization
  const baseTransformInv = new Float64Array((rank + 1) ** 2);
  matrix.inverse(
    baseTransformInv,
    rank + 1,
    baseTransform,
    rank + 1,
    rank + 1,
  );

  for (const scale of scales) {
    const t = scale.transform;
    
    // In OME's coordinate space, the origin of a voxel is its center, while in Neuroglancer it is
    // the "lower" (in coordinates) corner.
    // For arbitrary linear transforms, this is: translation -= Linear * (0.5, 0.5, ..., 0.5)
    for (let row = 0; row < rank; ++row) {
      let offset = 0;
      for (let col = 0; col < rank; ++col) {
        offset += t[row * (rank + 1) + col] * 0.5;
      }
      t[rank * (rank + 1) + row] -= offset;
    }

    // Make the scale relative to the base scale by multiplying with inverse of base transform
    const normalized = matrix.multiply(
      new Float64Array((rank + 1) ** 2) as Float64Array<ArrayBuffer>,
      rank + 1,
      baseTransformInv,
      rank + 1,
      t,
      rank + 1,
      rank + 1,
      rank + 1,
      rank + 1,
    );
    
    // Copy normalized transform back
    for (let i = 0; i < (rank + 1) ** 2; ++i) {
      t[i] = normalized[i];
    }
  }
  return { coordinateSpace, scales };
}

export function parseOmeMetadata(
  url: string,
  attrs: any,
  zarrVersion: number,
): OmeMetadata | undefined {
  const ome = attrs.ome;
  const multiscales = ome == undefined ? attrs.multiscales : ome.multiscales; // >0.4
  const omero = attrs.omero;

  if (!Array.isArray(multiscales)) return undefined;
  const errors: string[] = [];
  for (const multiscale of multiscales) {
    if (
      typeof multiscale !== "object" ||
      multiscale == null ||
      Array.isArray(multiscale)
    ) {
      // Not valid OME multiscale spec.
      return undefined;
    }

    const version = ome == undefined ? multiscale.version : ome.version; // >0.4

    if (version === undefined) return undefined;
    if (!SUPPORTED_OME_MULTISCALE_VERSIONS.has(version)) {
      errors.push(
        `OME multiscale metadata version ${JSON.stringify(
          version,
        )} is not supported`,
      );
      continue;
    }
    if (version === "0.5" && zarrVersion !== 3) {
      errors.push(
        `OME multiscale metadata version ${JSON.stringify(
          version,
        )} is not supported for zarr v${zarrVersion}`,
      );
      continue;
    }
    const multiScaleInfo = parseOmeMultiscale(url, multiscale);
    const channelMetadata = omero ? parseOmeroMetadata(omero) : undefined;
    return { multiscale: multiScaleInfo, channels: channelMetadata };
  }
  if (errors.length !== 0) {
    throw new Error(errors[0]);
  }
  return undefined;
}
