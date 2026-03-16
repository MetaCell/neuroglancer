export interface CatmaidCacheConfiguration {
  cache_type: string;
  cell_width: number;
  cell_height: number;
  cell_depth: number;
  [key: string]: any;
}

export interface CatmaidStackInfo {
  dimension: { x: number; y: number; z: number };
  resolution: { x: number; y: number; z: number };
  translation: { x: number; y: number; z: number };
  orientation?: number;
  title?: string;
  metadata?: {
    cache_provider?: string;
    cache_configurations?: CatmaidCacheConfiguration[];
  };
}

import { fetchOkWithCredentials } from "#src/credentials_provider/http_request.js";
import type { CredentialsProvider } from "#src/credentials_provider/index.js";
import {
  EditableSpatiallyIndexedSkeletonSource,
  SpatiallyIndexedSkeletonBranchNavigationTarget,
  SpatiallyIndexedSkeletonNavigationTarget,
  SpatiallyIndexedSkeletonNode,
  SpatiallyIndexedSkeletonOpenLeaf,
} from "#src/skeleton/api.js";
import { HttpError } from "#src/util/http_request.js";
import { Unpackr } from "msgpackr";

export interface CatmaidToken {
  token?: string;
}

export const credentialsKey = "CATMAID";
const DEFAULT_CACHE_GRID_CELL_WIDTH = 25000;
const DEFAULT_CACHE_GRID_CELL_HEIGHT = 25000;
const DEFAULT_CACHE_GRID_CELL_DEPTH = 40;
const CATMAID_NOCHECK_STATE = JSON.stringify({ nocheck: true });
const CATMAID_NO_MATCHING_NODE_PROVIDER_ERROR =
  "Could not find matching node provider for request";

type CatmaidStatePayload = string | object;

export interface CatmaidAddNodeResult {
  treenodeId: number;
  skeletonId: number;
}

export interface CatmaidDeleteNodeOptions {
  parentNodeId?: number;
  childNodeIds?: readonly number[];
  state?: CatmaidStatePayload;
}

export interface CatmaidNodeNavigationTarget
  extends SpatiallyIndexedSkeletonNavigationTarget {}

export interface CatmaidBranchNavigationTarget
  extends SpatiallyIndexedSkeletonBranchNavigationTarget {}

export interface CatmaidOpenLeafTarget
  extends SpatiallyIndexedSkeletonOpenLeaf {}

export interface CatmaidMergeSkeletonResult {
  resultSkeletonId: number | undefined;
  deletedSkeletonId: number | undefined;
  stableAnnotationSwap: boolean;
}

export interface CatmaidSplitSkeletonResult {
  existingSkeletonId: number | undefined;
  newSkeletonId: number | undefined;
}

export const CATMAID_TRUE_END_LABEL = "ends";

function includesNoMatchingNodeProviderError(value: unknown): boolean {
  return (
    typeof value === "string" &&
    value.includes(CATMAID_NO_MATCHING_NODE_PROVIDER_ERROR)
  );
}

function isNoMatchingNodeProviderErrorPayload(payload: unknown): boolean {
  if (payload === null || typeof payload !== "object") return false;
  const value = payload as { error?: unknown; detail?: unknown };
  return (
    includesNoMatchingNodeProviderError(value.error) ||
    includesNoMatchingNodeProviderError(value.detail)
  );
}

async function tryReadJsonPayload(
  response: Response,
): Promise<unknown | undefined> {
  try {
    return await response.json();
  } catch {
    return undefined;
  }
}

async function tryReadErrorPayload(
  response: Response,
): Promise<unknown | undefined> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return tryReadJsonPayload(response);
  }
  try {
    const text = await response.text();
    if (!text) return undefined;
    try {
      return JSON.parse(text);
    } catch {
      return { error: text };
    }
  } catch {
    return undefined;
  }
}

export function getCatmaidProjectSpaceBounds(info: CatmaidStackInfo): {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
} {
  const { dimension, resolution, translation } = info;
  const offsetX = translation?.x ?? 0;
  const offsetY = translation?.y ?? 0;
  const offsetZ = translation?.z ?? 0;

  // CATMAID treenode coordinates and grid cache cell sizes are in project-space nanometers.
  return {
    min: { x: offsetX, y: offsetY, z: offsetZ },
    max: {
      x: offsetX + dimension.x * resolution.x,
      y: offsetY + dimension.y * resolution.y,
      z: offsetZ + dimension.z * resolution.z,
    },
  };
}

function normalizeBoundingBoxForNodeList(boundingBox: {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
}) {
  const left = Math.floor(boundingBox.min.x);
  const top = Math.floor(boundingBox.min.y);
  const z1 = Math.floor(boundingBox.min.z);

  // CATMAID treats right/bottom as inclusive and z2 as exclusive for grid-cell index filtering.
  // Use ceil and ensure a positive extent on each axis.
  const right = Math.max(left + 1, Math.ceil(boundingBox.max.x));
  const bottom = Math.max(top + 1, Math.ceil(boundingBox.max.y));
  const z2 = Math.max(z1 + 1, Math.ceil(boundingBox.max.z));

  return { left, top, z1, right, bottom, z2 };
}

function appendNodeUpdateRows(
  body: URLSearchParams,
  key: string,
  rows: Array<[number, number, number, number]>,
) {
  // CATMAID get_request_list parses nested lists from bracketed keys
  // (e.g. t[0][0]=id, t[0][1]=x, ...), not from a JSON string.
  for (let rowIndex = 0; rowIndex < rows.length; ++rowIndex) {
    const row = rows[rowIndex];
    for (let colIndex = 0; colIndex < row.length; ++colIndex) {
      body.append(`${key}[${rowIndex}][${colIndex}]`, row[colIndex].toString());
    }
  }
}

function appendCatmaidState(
  body: URLSearchParams,
  state?: CatmaidStatePayload,
) {
  if (state === undefined) {
    body.append("state", CATMAID_NOCHECK_STATE);
    return;
  }
  if (typeof state === "string") {
    const normalizedState = state.trim();
    body.append(
      "state",
      normalizedState.length > 0 ? normalizedState : CATMAID_NOCHECK_STATE,
    );
    return;
  }
  body.append("state", JSON.stringify(state));
}

function fetchWithCatmaidCredentials(
  credentialsProvider: CredentialsProvider<CatmaidToken>,
  input: string,
  init: RequestInit,
): Promise<Response> {
  return fetchOkWithCredentials(
    credentialsProvider,
    input,
    init,
    (credentials: CatmaidToken, init: RequestInit) => {
      const newInit: RequestInit = { ...init };
      if (credentials.token) {
        newInit.headers = {
          ...newInit.headers,
          Authorization: `Token ${credentials.token}`,
        };
      }
      return newInit;
    },
    (error) => {
      const { status } = error;
      if (status === 403 || status === 401) {
        // Authorization needed.  Retry with refreshed token.
        return "refresh";
      }
      throw error;
    },
  );
}

export class CatmaidClient implements EditableSpatiallyIndexedSkeletonSource {
  private metadataInfoPromiseByKey = new Map<
    string,
    Promise<CatmaidStackInfo | null>
  >();
  private readonly msgpackUnpackr = new Unpackr({
    mapsAsObjects: false,
    int64AsType: "number",
  });

  constructor(
    public baseUrl: string,
    public projectId: number,
    public token?: string,
    public credentialsProvider?: CredentialsProvider<CatmaidToken>,
  ) {}

  private async fetch(
    endpoint: string,
    options: RequestInit = {},
    expectMsgpack: boolean = false,
  ): Promise<any> {
    // Ensure baseUrl doesn't have trailing slash and endpoint doesn't have leading slash
    const baseUrl = this.baseUrl.replace(/\/$/, "");
    const url = `${baseUrl}/${this.projectId}/${endpoint}`;
    const headers = new Headers(options.headers);
    if (this.token) {
      headers.append("X-Authorization", `Token ${this.token}`);
    }
    // CATMAID API often expects form-encoded data for POST
    if (options.method === "POST" && options.body instanceof URLSearchParams) {
      headers.append("Content-Type", "application/x-www-form-urlencoded");
    }

    let response: Response;
    if (this.credentialsProvider) {
      response = await fetchWithCatmaidCredentials(
        this.credentialsProvider,
        url,
        { ...options, headers },
      );
    } else {
      response = await fetch(url, { ...options, headers });
      if (!response.ok) {
        throw HttpError.fromResponse(response);
      }
    }

    if (expectMsgpack) {
      const contentType = response.headers.get("content-type") ?? "";
      if (contentType.includes("application/json")) {
        return response.json();
      }
      const buffer = await response.arrayBuffer();
      try {
        return this.msgpackUnpackr.unpack(new Uint8Array(buffer));
      } catch (error) {
        // Some CATMAID deployments return a JSON error body with a msgpack request.
        try {
          return JSON.parse(new TextDecoder().decode(buffer));
        } catch {
          throw error;
        }
      }
    }

    return response.json();
  }

  private async isNoMatchingNodeProviderHttpError(
    error: unknown,
  ): Promise<boolean> {
    if (!(error instanceof HttpError) || error.response === undefined) {
      return false;
    }
    const payload = await tryReadErrorPayload(error.response.clone());
    return isNoMatchingNodeProviderErrorPayload(payload);
  }

  private parseNavigationTarget(
    value: unknown,
  ): CatmaidNodeNavigationTarget | undefined {
    if (!Array.isArray(value) || value.length < 4) {
      return undefined;
    }
    const nodeId = Number(value[0]);
    const x = Number(value[1]);
    const y = Number(value[2]);
    const z = Number(value[3]);
    if (
      !Number.isFinite(nodeId) ||
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      !Number.isFinite(z)
    ) {
      return undefined;
    }
    return {
      nodeId: Math.round(nodeId),
      x,
      y,
      z,
    };
  }

  async listSkeletons(): Promise<number[]> {
    return this.fetch("skeletons/");
  }

  private async listStacks(): Promise<{ id: number; title: string }[]> {
    return this.fetch("stacks");
  }

  private async getStackInfo(stackId: number): Promise<CatmaidStackInfo> {
    return this.fetch(`stack/${stackId}/info`);
  }

  private getMetadataCacheKey(stackId?: number) {
    return stackId === undefined ? "default" : `stack:${stackId}`;
  }

  private async loadMetadataInfo(
    stackId?: number,
  ): Promise<CatmaidStackInfo | null> {
    let effectiveStackId: number;

    if (stackId !== undefined) {
      effectiveStackId = stackId;
    } else {
      try {
        const stacks = await this.listStacks();
        if (!stacks || stacks.length === 0) return null;
        effectiveStackId = stacks[0].id;
      } catch (e) {
        console.warn("Failed to fetch stack info:", e);
        return null;
      }
    }

    return this.getStackInfo(effectiveStackId);
  }

  private getMetadataInfo(stackId?: number): Promise<CatmaidStackInfo | null> {
    const key = this.getMetadataCacheKey(stackId);
    let promise = this.metadataInfoPromiseByKey.get(key);
    if (promise === undefined) {
      promise = this.loadMetadataInfo(stackId);
      this.metadataInfoPromiseByKey.set(key, promise);
      promise.catch(() => {
        if (this.metadataInfoPromiseByKey.get(key) === promise) {
          this.metadataInfoPromiseByKey.delete(key);
        }
      });
    }
    return promise;
  }

  async getDimensions(): Promise<{
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  } | null> {
    const info = await this.getMetadataInfo();
    if (!info) return null;
    return getCatmaidProjectSpaceBounds(info);
  }

  async getResolution(): Promise<{ x: number; y: number; z: number } | null> {
    const info = await this.getMetadataInfo();
    return info ? info.resolution : null;
  }

  async getGridCellSizes(): Promise<
    Array<{ x: number; y: number; z: number }>
  > {
    const info = await this.getMetadataInfo();
    const gridSizes: Array<{ x: number; y: number; z: number }> = [];

    // Try to get all grid cell sizes from metadata
    if (info?.metadata?.cache_configurations) {
      for (const config of info.metadata.cache_configurations) {
        if (config.cache_type === "grid") {
          gridSizes.push({
            x: config.cell_width,
            y: config.cell_height,
            z: config.cell_depth,
          });
        }
      }
    }

    // If no grid configs found, use default
    if (gridSizes.length === 0) {
      gridSizes.push({
        x: DEFAULT_CACHE_GRID_CELL_WIDTH,
        y: DEFAULT_CACHE_GRID_CELL_HEIGHT,
        z: DEFAULT_CACHE_GRID_CELL_DEPTH,
      });
    }

    return gridSizes;
  }

  async getCacheProvider(): Promise<string | undefined> {
    const info = await this.getMetadataInfo();
    return info?.metadata?.cache_provider;
  }

  async getSkeleton(
    skeletonId: number,
  ): Promise<SpatiallyIndexedSkeletonNode[]> {
    const data = await this.fetch(
      `skeletons/${skeletonId}/compact-detail?with_tags=true`,
    );
    const nodes = Array.isArray(data?.[0]) ? data[0] : [];
    const labels = data?.[2] ?? {};
    return nodes.map((n: any[]) => ({
      id: n[0],
      parent_id: n[1],
      x: n[3],
      y: n[4],
      z: n[5],
      skeleton_id: skeletonId,
      radius: Number.isFinite(n[6]) ? n[6] : undefined,
      confidence: Number.isFinite(n[7]) ? n[7] : undefined,
      labels: Array.isArray(labels?.[n[0]]) ? labels[n[0]] : undefined,
    }));
  }

  async getSkeletonRootNode(
    skeletonId: number,
  ): Promise<CatmaidNodeNavigationTarget> {
    const response = await this.fetch(`skeletons/${skeletonId}/root`);
    const nodeId = Number(response?.root_id);
    const x = Number(response?.x);
    const y = Number(response?.y);
    const z = Number(response?.z);
    if (
      !Number.isFinite(nodeId) ||
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      !Number.isFinite(z)
    ) {
      throw new Error(
        "CATMAID skeleton root endpoint returned an unexpected response format.",
      );
    }
    return {
      nodeId: Math.round(nodeId),
      x,
      y,
      z,
    };
  }

  async getPreviousBranchOrRoot(
    nodeId: number,
    options: {
      alt?: boolean;
    } = {},
  ): Promise<CatmaidNodeNavigationTarget> {
    const body = new URLSearchParams({
      alt: options.alt ? "1" : "0",
    });
    const response = await this.fetch(
      `treenodes/${nodeId}/previous-branch-or-root`,
      {
        method: "POST",
        body,
      },
    );
    const target = this.parseNavigationTarget(response);
    if (target === undefined) {
      throw new Error(
        "CATMAID previous-branch-or-root endpoint returned an unexpected response format.",
      );
    }
    return target;
  }

  async getNextBranchOrEnd(
    nodeId: number,
  ): Promise<CatmaidBranchNavigationTarget[]> {
    const response = await this.fetch(
      `treenodes/${nodeId}/next-branch-or-end`,
      {
        method: "POST",
        body: new URLSearchParams(),
      },
    );
    if (!Array.isArray(response)) {
      throw new Error(
        "CATMAID next-branch-or-end endpoint returned an unexpected response format.",
      );
    }
    const branches: CatmaidBranchNavigationTarget[] = [];
    for (const branch of response) {
      if (!Array.isArray(branch) || branch.length < 3) continue;
      const child = this.parseNavigationTarget(branch[0]);
      const branchStartOrEnd = this.parseNavigationTarget(branch[1]);
      const branchEnd = this.parseNavigationTarget(branch[2]);
      if (
        child === undefined ||
        branchStartOrEnd === undefined ||
        branchEnd === undefined
      ) {
        continue;
      }
      branches.push({
        child,
        branchStartOrEnd,
        branchEnd,
      });
    }
    return branches;
  }

  async getOpenLeaves(
    skeletonId: number,
    nodeId: number,
  ): Promise<CatmaidOpenLeafTarget[]> {
    const body = new URLSearchParams({
      treenode_id: nodeId.toString(),
    });
    const response = await this.fetch(`skeletons/${skeletonId}/open-leaves`, {
      method: "POST",
      body,
    });
    if (!Array.isArray(response)) {
      throw new Error(
        "CATMAID open-leaves endpoint returned an unexpected response format.",
      );
    }
    const leaves: CatmaidOpenLeafTarget[] = [];
    for (const leaf of response) {
      if (!Array.isArray(leaf) || leaf.length < 3 || !Array.isArray(leaf[1])) {
        continue;
      }
      const nodeIdValue = Number(leaf[0]);
      const x = Number(leaf[1][0]);
      const y = Number(leaf[1][1]);
      const z = Number(leaf[1][2]);
      const distance = Number(leaf[2]);
      if (
        !Number.isFinite(nodeIdValue) ||
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        !Number.isFinite(z) ||
        !Number.isFinite(distance)
      ) {
        continue;
      }
      leaves.push({
        nodeId: Math.round(nodeIdValue),
        x,
        y,
        z,
        distance,
        creationTime:
          typeof leaf[3] === "string" && leaf[3].length > 0
            ? leaf[3]
            : undefined,
      });
    }
    return leaves;
  }

  async fetchNodes(
    boundingBox: {
      min: { x: number; y: number; z: number };
      max: { x: number; y: number; z: number };
    },
    lod: number = 0,
    options: {
      cacheProvider?: string;
      signal?: AbortSignal;
      includeLabels?: boolean;
    } = {},
  ): Promise<SpatiallyIndexedSkeletonNode[]> {
    const { cacheProvider, signal, includeLabels = false } = options;
    const normalizedBoundingBox = normalizeBoundingBoxForNodeList(boundingBox);
    const params = new URLSearchParams({
      left: normalizedBoundingBox.left.toString(),
      top: normalizedBoundingBox.top.toString(),
      z1: normalizedBoundingBox.z1.toString(),
      right: normalizedBoundingBox.right.toString(),
      bottom: normalizedBoundingBox.bottom.toString(),
      z2: normalizedBoundingBox.z2.toString(),
      lod_type: "percent",
      lod: lod.toString(),
      format: "msgpack",
      labels: includeLabels ? "true" : "false",
    });

    // Add cache provider if available
    if (cacheProvider) {
      params.append("src", cacheProvider);
    }

    let data: any;
    try {
      data = await this.fetch(
        `node/list?${params.toString()}`,
        { signal },
        true,
      );
    } catch (error) {
      if (await this.isNoMatchingNodeProviderHttpError(error)) {
        return [];
      }
      throw error;
    }

    if (isNoMatchingNodeProviderErrorPayload(data)) {
      return [];
    }

    if (!Array.isArray(data) || !Array.isArray(data[0])) {
      throw new Error(
        "CATMAID node/list endpoint returned an unexpected response format.",
      );
    }

    // Check if limit was reached for the first LOD level
    if (data[3]) {
      console.warn(
        "CATMAID node/list endpoint returned limit_reached=true. Some nodes may be missing.",
      );
    }

    // Process first LOD level (data[0])
    const labels = data[2] ?? {};
    const nodes: SpatiallyIndexedSkeletonNode[] = data[0].map((n: any[]) => ({
      id: n[0],
      parent_id: n[1],
      x: n[2],
      y: n[3],
      z: n[4],
      radius: Number.isFinite(n[5]) ? n[5] : undefined,
      confidence: Number.isFinite(n[6]) ? n[6] : undefined,
      skeleton_id: n[7],
      labels: Array.isArray(labels?.[n[0]]) ? labels[n[0]] : undefined,
    }));

    // Process additional LOD levels (data[5] - extraNodes)
    const extraNodes = data[5];
    if (Array.isArray(extraNodes)) {
      for (const lodLevel of extraNodes) {
        // Each lodLevel is [treenodes, connectors, labels, limit_reached, relations]
        if (lodLevel[3]) {
          console.warn(
            "CATMAID node/list endpoint returned limit_reached=true for an extra LOD level. Some nodes may be missing.",
          );
        }
        const treenodes = lodLevel[0];
        const lodLabels = lodLevel[2] ?? {};
        if (Array.isArray(treenodes)) {
          for (const n of treenodes) {
            nodes.push({
              id: n[0],
              parent_id: n[1],
              x: n[2],
              y: n[3],
              z: n[4],
              radius: Number.isFinite(n[5]) ? n[5] : undefined,
              confidence: Number.isFinite(n[6]) ? n[6] : undefined,
              skeleton_id: n[7],
              labels: Array.isArray(lodLabels?.[n[0]])
                ? lodLabels[n[0]]
                : undefined,
            });
          }
        }
      }
    }

    return nodes;
  }

  async moveNode(
    nodeId: number,
    x: number,
    y: number,
    z: number,
    state?: CatmaidStatePayload,
  ): Promise<void> {
    const body = new URLSearchParams();
    appendNodeUpdateRows(body, "t", [[nodeId, x, y, z]]);
    appendCatmaidState(body, state);

    await this.fetch(`node/update`, {
      method: "POST",
      body: body,
    });
  }

  private async splitSkeletonAtNode(
    nodeId: number,
  ): Promise<CatmaidSplitSkeletonResult> {
    const body = new URLSearchParams({
      treenode_id: nodeId.toString(),
    });
    const response = await this.fetch(`skeleton/split`, {
      method: "POST",
      body,
    });
    const existingSkeletonId = Number(response?.existing_skeleton_id);
    const newSkeletonId = Number(response?.new_skeleton_id);
    return {
      existingSkeletonId: Number.isFinite(existingSkeletonId)
        ? Math.round(existingSkeletonId)
        : undefined,
      newSkeletonId: Number.isFinite(newSkeletonId)
        ? Math.round(newSkeletonId)
        : undefined,
    };
  }

  async rerootSkeleton(
    nodeId: number,
    state?: CatmaidStatePayload,
  ): Promise<void> {
    const body = new URLSearchParams({
      treenode_id: nodeId.toString(),
    });
    appendCatmaidState(body, state);
    await this.fetch(`skeleton/reroot`, {
      method: "POST",
      body,
    });
  }

  async deleteNode(
    nodeId: number,
    options: CatmaidDeleteNodeOptions = {},
  ): Promise<void> {
    const { parentNodeId, childNodeIds = [], state } = options;
    const normalizedParentNodeId =
      parentNodeId === undefined || parentNodeId === null
        ? undefined
        : Math.round(Number(parentNodeId));
    const normalizedChildIds = [
      ...new Set(
        childNodeIds
          .map((value) => Number(value))
          .filter((value) => Number.isFinite(value))
          .map((value) => Math.round(value)),
      ),
    ].sort((a, b) => a - b);
    if (normalizedParentNodeId === undefined && normalizedChildIds.length > 0) {
      await this.rerootSkeleton(normalizedChildIds[0], state);
    }
    const body = new URLSearchParams({
      treenode_id: nodeId.toString(),
    });
    appendCatmaidState(body, state);
    const response = await this.fetch(`treenode/delete`, {
      method: "POST",
      body: body,
    });
    if (response?.success === undefined) {
      throw new Error("Delete endpoint returned an unexpected response.");
    }
  }

  async addNode(
    skeletonId: number,
    x: number,
    y: number,
    z: number,
    parentId?: number,
    state?: CatmaidStatePayload,
  ): Promise<CatmaidAddNodeResult> {
    const body = new URLSearchParams({
      x: x.toString(),
      y: y.toString(),
      z: z.toString(),
      parent_id: (parentId ?? -1).toString(),
    });
    if (Number.isSafeInteger(skeletonId) && skeletonId > 0) {
      body.append("skeleton_id", skeletonId.toString());
    }
    appendCatmaidState(body, state);

    const res = await this.fetch(`treenode/create`, {
      method: "POST",
      body: body,
    });
    const treenodeId = Number(res?.treenode_id);
    const nextSkeletonId = Number(res?.skeleton_id);
    if (!Number.isFinite(treenodeId)) {
      throw new Error(
        "CATMAID treenode/create did not return a valid treenode_id.",
      );
    }
    if (!Number.isFinite(nextSkeletonId)) {
      throw new Error(
        "CATMAID treenode/create did not return a valid skeleton_id.",
      );
    }
    return {
      treenodeId,
      skeletonId: nextSkeletonId,
    };
  }

  private async updateNodeLabelWithFallback(
    nodeId: number,
    endpoint: "update" | "remove",
    bodyFactories: Array<() => URLSearchParams>,
  ) {
    let lastError: unknown;
    for (const makeBody of bodyFactories) {
      try {
        await this.fetch(`label/treenode/${nodeId}/${endpoint}`, {
          method: "POST",
          body: makeBody(),
        });
        return;
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError ?? new Error(`Failed to ${endpoint} treenode label.`);
  }

  async addNodeLabel(nodeId: number, label: string): Promise<void> {
    const normalizedLabel = label.trim();
    if (normalizedLabel.length === 0) {
      throw new Error("Node label must not be empty.");
    }
    await this.updateNodeLabelWithFallback(nodeId, "update", [
      () => new URLSearchParams({ tags: normalizedLabel }),
      () => new URLSearchParams({ tag: normalizedLabel }),
    ]);
  }

  async removeNodeLabel(nodeId: number, label: string): Promise<void> {
    const normalizedLabel = label.trim();
    if (normalizedLabel.length === 0) {
      throw new Error("Node label must not be empty.");
    }
    await this.updateNodeLabelWithFallback(nodeId, "remove", [
      () => new URLSearchParams({ tag: normalizedLabel }),
      () => new URLSearchParams({ label: normalizedLabel }),
      () => new URLSearchParams({ tags: normalizedLabel }),
    ]);
  }

  async mergeSkeletons(
    fromNodeId: number,
    toNodeId: number,
  ): Promise<CatmaidMergeSkeletonResult> {
    const body = new URLSearchParams({
      from_id: fromNodeId.toString(),
      to_id: toNodeId.toString(),
    });
    const response = await this.fetch(`skeleton/join`, {
      method: "POST",
      body,
    });
    const resultSkeletonId = Number(response?.result_skeleton_id);
    const deletedSkeletonId = Number(response?.deleted_skeleton_id);
    return {
      resultSkeletonId: Number.isFinite(resultSkeletonId)
        ? Math.round(resultSkeletonId)
        : undefined,
      deletedSkeletonId: Number.isFinite(deletedSkeletonId)
        ? Math.round(deletedSkeletonId)
        : undefined,
      stableAnnotationSwap: Boolean(response?.stable_annotation_swap),
    };
  }

  async splitSkeleton(nodeId: number): Promise<CatmaidSplitSkeletonResult> {
    return this.splitSkeletonAtNode(nodeId);
  }
}
