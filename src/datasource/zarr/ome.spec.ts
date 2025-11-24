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

describe("OME-Zarr 0.6 coordinate transformations", () => {
  it("should validate sequence transform with correct input/output", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [
              {
                name: "physical",
                axes: [
                  { type: "space", name: "z", unit: "micrometer" },
                  { type: "space", name: "y", unit: "micrometer" },
                  { type: "space", name: "x", unit: "micrometer" },
                ],
              },
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
            coordinateSystems: [
              {
                name: "physical",
                axes: [
                  { type: "space", name: "z", unit: "micrometer" },
                  { type: "space", name: "y", unit: "micrometer" },
                  { type: "space", name: "x", unit: "micrometer" },
                ],
              },
            ],
            datasets: [
              {
                path: "s0",
                coordinateTransformations: [
                  {
                    type: "scale",
                    output: "physical",
                    input: "",  // Empty string means not specified
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
            coordinateSystems: [
              {
                name: "physical",
                axes: [
                  { type: "space", name: "z", unit: "micrometer" },
                  { type: "space", name: "y", unit: "micrometer" },
                  { type: "space", name: "x", unit: "micrometer" },
                ],
              },
            ],
            datasets: [
              {
                path: "array",
                coordinateTransformations: [
                  {
                    type: "sequence",
                    output: "wrong_system",
                    input: "array",
                    transformations: [
                      { type: "scale", scale: [4, 3, 2] },
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
      /output is "wrong_system" but expected "physical"/
    );
  });

  it("should reject sequence transform with wrong input", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [
              {
                name: "physical",
                axes: [
                  { type: "space", name: "z", unit: "micrometer" },
                  { type: "space", name: "y", unit: "micrometer" },
                  { type: "space", name: "x", unit: "micrometer" },
                ],
              },
            ],
            datasets: [
              {
                path: "array",
                coordinateTransformations: [
                  {
                    type: "sequence",
                    output: "physical",
                    input: "wrong_path",
                    transformations: [
                      { type: "scale", scale: [4, 3, 2] },
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
      /input is "wrong_path" but expected "array"/
    );
  });

  it("should reject nested sequence transforms", () => {
    const attrs = {
      ome: {
        version: "0.6",
        multiscales: [
          {
            name: "multiscales",
            coordinateSystems: [
              {
                name: "physical",
                axes: [
                  { type: "space", name: "z", unit: "micrometer" },
                  { type: "space", name: "y", unit: "micrometer" },
                  { type: "space", name: "x", unit: "micrometer" },
                ],
              },
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
                        type: "sequence",
                        transformations: [
                          { type: "scale", scale: [4, 3, 2] },
                        ],
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
      /sequence transformation MUST NOT be part of another sequence transformation/
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
              {
                name: "physical",
                axes: [
                  { type: "space", name: "z", unit: "micrometer" },
                  { type: "space", name: "y", unit: "micrometer" },
                  { type: "space", name: "x", unit: "micrometer" },
                ],
              },
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
              {
                name: "physical",
                axes: [
                  { type: "space", name: "z", unit: "micrometer" },
                  { type: "space", name: "y", unit: "micrometer" },
                  { type: "space", name: "x", unit: "micrometer" },
                ],
              },
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
                        input: "wrong_system",  // Wrong input - doesn't match previous output
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
      /transform 0 has output "intermediate" but transform 1 has input "wrong_system"/
    );
  });
});
