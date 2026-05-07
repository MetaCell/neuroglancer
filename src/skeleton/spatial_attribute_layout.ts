import { DataType } from "#src/util/data_type.js";

export const spatiallyIndexedSkeletonTextureAttributeSpecs = Object.freeze([
  { name: "position", dataType: DataType.FLOAT32, numComponents: 3 },
  { name: "segment", dataType: DataType.UINT64, numComponents: 1 },
]);
