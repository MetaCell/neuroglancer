/**
 * @license
 * This work is a derivative of the Google Neuroglancer project,
 * Copyright 2016 Google Inc.
 * The Derivative Work is covered by
 * Copyright 2020 Howard Hughes Medical Institute
 *
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

import type { SkeletonChunk } from "#src/skeleton/backend.js";

// https://swc-specification.readthedocs.io/en/latest/swc.html
interface SwcNode {
  readonly id: number;
  readonly type: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly radius: number;
  readonly parent: number;
}

const swcRootParent = -1;

export enum SwcValidation {
  STRICT,
  // Skips each problem item and warns once for each file.
  LENIENT,
}

class SwcProblemLog {
  private readonly countByProblem = new Map<string, number>();

  constructor(private readonly validation: SwcValidation) {}

  report(problem: string, detail: string) {
    if (this.validation === SwcValidation.STRICT) throw new Error(detail);
    this.countByProblem.set(
      problem,
      (this.countByProblem.get(problem) ?? 0) + 1,
    );
  }

  warnIfAny(objectId: bigint) {
    if (this.countByProblem.size === 0) return;
    const summary = Array.from(
      this.countByProblem,
      ([problem, count]) => `${count} ${problem}`,
    ).join(", ");
    console.warn(`SWC skeleton ${objectId}: ${summary}`);
  }
}

export function decodeSwcSkeletonChunk(
  chunk: SkeletonChunk,
  swcText: string,
  options: { validation: SwcValidation },
) {
  const problems = new SwcProblemLog(options.validation);
  const nodeById = new Map<number, SwcNode>();
  for (const node of parseSwc(swcText, problems)) {
    if (nodeById.has(node.id)) {
      problems.report(
        "duplicate nodes replaced by a later line",
        `SWC node ${node.id} is defined more than once`,
      );
    }
    nodeById.set(node.id, node);
  }
  const nodes = Array.from(nodeById.values());
  if (nodes.length === 0) {
    throw new Error("SWC file contains no nodes");
  }

  const vertexIndexById = new Map(
    nodes.map((node, vertexIndex) => [node.id, vertexIndex]),
  );
  const vertexPositions = new Float32Array(3 * nodes.length);
  const radii = new Float32Array(nodes.length);
  const types = new Float32Array(nodes.length);
  const edges: number[] = [];
  nodes.forEach((node, vertexIndex) => {
    vertexPositions[3 * vertexIndex] = node.x;
    vertexPositions[3 * vertexIndex + 1] = node.y;
    vertexPositions[3 * vertexIndex + 2] = node.z;
    radii[vertexIndex] = node.radius;
    types[vertexIndex] = node.type;
    if (node.parent === swcRootParent) return;
    const parentVertexIndex = vertexIndexById.get(node.parent);
    if (parentVertexIndex === undefined) {
      problems.report(
        "edges dropped because the parent is not in the file",
        `SWC node ${node.id} has parent ${node.parent}, which is not in the file`,
      );
      return;
    }
    edges.push(vertexIndex, parentVertexIndex);
  });
  problems.warnIfAny(chunk.objectId);

  chunk.vertexPositions = vertexPositions;
  chunk.indices = Uint32Array.from(edges);
  // In the order of `swcVertexAttributes`.
  chunk.vertexAttributes = [radii, types];
}

function parseSwc(swcText: string, problems: SwcProblemLog): SwcNode[] {
  const nodes: SwcNode[] = [];
  swcText.split(/\r?\n/).forEach((line, lineIndex) => {
    const content = line.trim();
    if (content === "" || content.startsWith("#")) return;
    const columns = content.split(/\s+/).map(Number);
    if (columns.length !== 7 || !columns.every(Number.isFinite)) {
      problems.report(
        "lines skipped because they are not seven numbers",
        `SWC line ${lineIndex + 1} is not seven numbers: ${JSON.stringify(line)}`,
      );
      return;
    }
    const [id, type, x, y, z, radius, parent] = columns;
    nodes.push({ id, type, x, y, z, radius, parent });
  });
  return nodes;
}
