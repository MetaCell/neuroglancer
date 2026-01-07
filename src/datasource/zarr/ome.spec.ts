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

import { describe, it, expect } from "vitest";
import { parseOmeMetadata } from "#src/datasource/zarr/ome.js";

const regularCoordinateSystem = {
  name: "physical",
  axes: [
    { type: "space", name: "z", unit: "micrometer" },
    { type: "space", name: "y", unit: "micrometer" },
    { type: "space", name: "x", unit: "micrometer" },
  ],
};

function makeOmeAttrsWithTransform(transform: any) {
  return {
    ome: {
      version: "0.6",
      multiscales: [
        {
          name: "multiscales",
          coordinateSystems: [regularCoordinateSystem],
          datasets: [
            {
              path: "array",
              coordinateTransformations: [transform],
            },
          ],
        },
      ],
    },
  };
}

describe("OME-Zarr 0.6 coordinate transformations", () => {
  it("should parse an identity transformation", () => {
    const attrs = makeOmeAttrsWithTransform({
      type: "identity",
      output: "physical",
    });
    const metadata = parseOmeMetadata("test://", attrs, 3);
    expect(metadata!.multiscale.baseInfo.baseTransform).toStrictEqual(
      new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    );
  });

  it("should parse a scale transformation", () => {
    const attrs = makeOmeAttrsWithTransform({
      type: "scale",
      scale: [10, 0.3, 2],
      output: "physical",
    });
    const metadata = parseOmeMetadata("test://", attrs, 3);
    // The scale gets extracted out, so the base transform is identity
    expect(metadata!.multiscale.baseInfo.baseTransform).toStrictEqual(
      new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    );
    console.log(metadata);
    const scales = metadata!.multiscale.coordinateSpace.scales;
    expect(scales[0]).toBeCloseTo(1e-5); // 10 micrometer in meters
    expect(scales[1]).toBeCloseTo(3e-7); // 0.3 micrometer in meters
    expect(scales[2]).toBeCloseTo(2e-6); // 2 micrometer in meters
  });

  it("should parse a translation transformation", () => {
    const attrs = makeOmeAttrsWithTransform({
      type: "translation",
      translation: [10, 20, 30],
      output: "physical",
    });
    const metadata = parseOmeMetadata("test://", attrs, 3);
    expect(metadata!.multiscale.baseInfo.baseTransform).toStrictEqual(
      new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1]),
    );
  });

  it("should parse a rotation transformation", () => {
    const attrs = makeOmeAttrsWithTransform({
      type: "rotation",
      rotation: [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
      ],
      output: "physical",
    });
    const metadata = parseOmeMetadata("test://", attrs, 3);
    expect(metadata!.multiscale.baseInfo.baseTransform).toStrictEqual(
      new Float64Array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]),
    );
  });

  it("should parse a mapAxis transformation", () => {
    const attrs = makeOmeAttrsWithTransform({
      type: "mapAxis",
      mapAxis: [1, 0, 2],
      output: "physical",
    });
    const metadata = parseOmeMetadata("test://", attrs, 3);
    expect(metadata!.multiscale.baseInfo.baseTransform).toStrictEqual(
      new Float64Array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    );
  });

  it("should parse an affine transformation", () => {
    const attrs = makeOmeAttrsWithTransform({
      type: "affine",
      affine: [
        [1, 0.4, 0.1, 4.5],
        [0.1, 0.8, 0.4, 6.9],
        [0.3, 0.2, 0.9, 8.1],
      ],
      output: "physical",
    });
    const metadata = parseOmeMetadata("test://", attrs, 3);
    // We need to back the scale out of the affine matrix
    // That scale is the lengths of the column vectors
    const expectedScales = [
      Math.sqrt(1 * 1 + 0.1 * 0.1 + 0.3 * 0.3),
      Math.sqrt(0.4 * 0.4 + 0.8 * 0.8 + 0.2 * 0.2),
      Math.sqrt(0.1 * 0.1 + 0.4 * 0.4 + 0.9 * 0.9),
    ];
    const scales = metadata!.multiscale.coordinateSpace.scales;
    expect(scales[0]).toBeCloseTo(expectedScales[0] * 1e-6);
    expect(scales[1]).toBeCloseTo(expectedScales[1] * 1e-6);
    expect(scales[2]).toBeCloseTo(expectedScales[2] * 1e-6);
    expect(metadata!.multiscale.baseInfo.baseTransform).toStrictEqual(
      new Float64Array([
        1 / expectedScales[0],
        0.1 / expectedScales[1],
        0.3 / expectedScales[2],
        0,
        0.4 / expectedScales[0],
        0.8 / expectedScales[1],
        0.2 / expectedScales[2],
        0,
        0.1 / expectedScales[0],
        0.4 / expectedScales[1],
        0.9 / expectedScales[2],
        0,
        4.5 / expectedScales[0],
        6.9 / expectedScales[1],
        8.1 / expectedScales[2],
        1,
      ]),
    );
  });

  it("should combine transforms in a sequence", () => {
    const attrs = makeOmeAttrsWithTransform({
      type: "sequence",
      output: "physical",
      input: "array",
      transformations: [
        { type: "scale", scale: [2, 3, 4] },
        { type: "translation", translation: [20, 30, 40] },
        {
          type: "rotation",
          rotation: [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
          ],
        },
      ],
    });
    const metadata = parseOmeMetadata("test://", attrs, 3);
    // The combined transform as an affine would be after the scale, then translation
    // [2, 0, 0, 20], [0, 3, 0, 30], [0, 0, 4, 40], [0, 0, 0, 1]
    // then apply rotation
    // [0, 3, 0, 30], [0, 0, 4, 40], [2, 0, 0, 20], [0, 0, 0, 1]
    // We back the scale out of that final matrix and compare to affine
    const expectedScales = [2, 3, 4];
    const scales = metadata!.multiscale.coordinateSpace.scales;
    expect(scales[0]).toBeCloseTo(expectedScales[0] * 1e-6);
    expect(scales[1]).toBeCloseTo(expectedScales[1] * 1e-6);
    expect(scales[2]).toBeCloseTo(expectedScales[2] * 1e-6);
    expect(metadata!.multiscale.baseInfo.baseTransform).toStrictEqual(
      new Float64Array([
        0,
        0,
        2 / expectedScales[2],
        0,
        3 / expectedScales[0],
        0,
        0,
        0,
        0,
        4 / expectedScales[1],
        0,
        0,
        30 / expectedScales[0],
        40 / expectedScales[1],
        20 / expectedScales[2],
        1,
      ]),
    );
  });

  it("should use the last coordinate system if multiple provided", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [
              {
                name: "first_system",
                axes: [{ type: "space", name: "x", unit: "micrometer" }],
              },
              regularCoordinateSystem,
            ],
            datasets: [
              {
                path: "array",
                coordinateTransformations: [
                  {
                    type: "identity",
                    output: "physical",
                  },
                ],
              },
            ],
          },
        ],
      },
    };

    const metadata = parseOmeMetadata("test://", attrs, 3);
    const space = metadata!.multiscale.coordinateSpace;
    expect(space.names).toStrictEqual(["z", "y", "x"]);
    expect(space.units).toStrictEqual(["m", "m", "m"]);
  });

  it("should throw an error for non-supported transformation types", () => {
    const attrs = makeOmeAttrsWithTransform({
      type: "non_existent_transform",
      output: "physical",
    });
    expect(() => parseOmeMetadata("test://", attrs, 3)).toThrow(
      'Error parsing "datasets" property: Unsupported coordinate transform type: "non_existent_transform"',
    );
  });
});

describe("OME-Zarr 0.6 sequence transformation validation", () => {
  it("should validate sequence transform with correct input/output", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [regularCoordinateSystem],
            datasets: [
              {
                path: "array",
                coordinateTransformations: [
                  {
                    type: "sequence",
                    output: "physical",
                    input: "array",
                    transformations: [
                      { type: "scale", scale: [4, 3, 2] },
                      { type: "translation", translation: [32, 21, 10] },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    };

    // This should not throw an error
    expect(() => parseOmeMetadata("test://", attrs, 3)).not.toThrow();
  });

  it("should accept transforms with empty string input/output (optional fields)", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [regularCoordinateSystem],
            datasets: [
              {
                path: "s0",
                coordinateTransformations: [
                  {
                    type: "scale",
                    output: "physical",
                    input: "", // Empty string means not specified
                    scale: [4, 3, 2],
                  },
                ],
              },
            ],
          },
        ],
      },
    };

    // This should not throw an error - empty strings are treated as "not specified"
    expect(() => parseOmeMetadata("test://", attrs, 3)).not.toThrow();
  });

  it("should reject sequence transform with wrong output coordinate system", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [regularCoordinateSystem],
            datasets: [
              {
                path: "array",
                coordinateTransformations: [
                  {
                    type: "sequence",
                    output: "wrong_system",
                    input: "array",
                    transformations: [{ type: "scale", scale: [4, 3, 2] }],
                  },
                ],
              },
            ],
          },
        ],
      },
    };

    // This should throw an error
    expect(() => parseOmeMetadata("test://", attrs, 3)).toThrow(
      /output is "wrong_system" but expected "physical"/,
    );
  });

  it("should reject sequence transform with wrong input", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [regularCoordinateSystem],
            datasets: [
              {
                path: "array",
                coordinateTransformations: [
                  {
                    type: "sequence",
                    output: "physical",
                    input: "wrong_path",
                    transformations: [{ type: "scale", scale: [4, 3, 2] }],
                  },
                ],
              },
            ],
          },
        ],
      },
    };

    // This should throw an error
    expect(() => parseOmeMetadata("test://", attrs, 3)).toThrow(
      /input is "wrong_path" but expected "array"/,
    );
  });

  it("should reject nested sequence transforms", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [regularCoordinateSystem],
            datasets: [
              {
                path: "array",
                coordinateTransformations: [
                  {
                    type: "sequence",
                    output: "physical",
                    input: "array",
                    transformations: [
                      {
                        type: "sequence",
                        transformations: [{ type: "scale", scale: [4, 3, 2] }],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    };

    // This should throw an error
    expect(() => parseOmeMetadata("test://", attrs, 3)).toThrow(
      /sequence transformation MUST NOT be part of another sequence transformation/,
    );
  });

  it("should validate chaining of inner transforms in sequence", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [
              {
                name: "intermediate",
                axes: [
                  { type: "space", name: "z", unit: "micrometer" },
                  { type: "space", name: "y", unit: "micrometer" },
                  { type: "space", name: "x", unit: "micrometer" },
                ],
              },
              regularCoordinateSystem,
            ],
            datasets: [
              {
                path: "array",
                coordinateTransformations: [
                  {
                    type: "sequence",
                    output: "physical",
                    input: "array",
                    transformations: [
                      {
                        type: "scale",
                        scale: [4, 3, 2],
                        input: "array",
                        output: "intermediate",
                      },
                      {
                        type: "translation",
                        translation: [32, 21, 10],
                        input: "intermediate",
                        output: "physical",
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    };

    // This should not throw an error as the chain is valid
    expect(() => parseOmeMetadata("test://", attrs, 3)).not.toThrow();
  });

  it("should reject broken chain in sequence transforms", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [
              {
                name: "intermediate",
                axes: [
                  { type: "space", name: "z", unit: "micrometer" },
                  { type: "space", name: "y", unit: "micrometer" },
                  { type: "space", name: "x", unit: "micrometer" },
                ],
              },
              regularCoordinateSystem,
            ],
            datasets: [
              {
                path: "array",
                coordinateTransformations: [
                  {
                    type: "sequence",
                    output: "physical",
                    input: "array",
                    transformations: [
                      {
                        type: "scale",
                        scale: [4, 3, 2],
                        input: "array",
                        output: "intermediate",
                      },
                      {
                        type: "translation",
                        translation: [32, 21, 10],
                        input: "wrong_system", // Wrong input - doesn't match previous output
                        output: "physical",
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    };

    // This should throw an error as the chain is broken
    expect(() => parseOmeMetadata("test://", attrs, 3)).toThrow(
      /transform 0 has output "intermediate" but transform 1 has input "wrong_system"/,
    );
  });
});
