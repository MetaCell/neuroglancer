
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
import { SpatiallyIndexedSkeletonNode, SpatiallyIndexedSkeletonSource } from "#src/skeleton/api.js";
import { HttpError } from "#src/util/http_request.js";
import { Unpackr } from "msgpackr";

export interface CatmaidToken {
    token?: string;
}

export const credentialsKey = "CATMAID";
const DEFAULT_CACHE_GRID_CELL_WIDTH = 25000;
const DEFAULT_CACHE_GRID_CELL_HEIGHT = 25000;
const DEFAULT_CACHE_GRID_CELL_DEPTH = 40;
const CATMAID_NO_MATCHING_NODE_PROVIDER_ERROR = "Could not find matching node provider for request";

function includesNoMatchingNodeProviderError(value: unknown): boolean {
    return typeof value === "string" && value.includes(CATMAID_NO_MATCHING_NODE_PROVIDER_ERROR);
}

function isNoMatchingNodeProviderErrorPayload(payload: unknown): boolean {
    if (payload === null || typeof payload !== "object") return false;
    const value = payload as { error?: unknown; detail?: unknown };
    return (
        includesNoMatchingNodeProviderError(value.error) ||
        includesNoMatchingNodeProviderError(value.detail)
    );
}

async function tryReadJsonPayload(response: Response): Promise<unknown | undefined> {
    try {
        return await response.json();
    } catch {
        return undefined;
    }
}

async function tryReadErrorPayload(response: Response): Promise<unknown | undefined> {
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

export class CatmaidClient implements SpatiallyIndexedSkeletonSource {
    private metadataInfoPromiseByKey = new Map<string, Promise<CatmaidStackInfo | null>>();
    private readonly msgpackUnpackr = new Unpackr({
        mapsAsObjects: false,
        int64AsType: "number",
    });

    constructor(
        public baseUrl: string,
        public projectId: number,
        public token?: string,
        public credentialsProvider?: CredentialsProvider<CatmaidToken>,
    ) { }

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
            response = await fetchWithCatmaidCredentials(this.credentialsProvider, url, { ...options, headers });
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

    private async isNoMatchingNodeProviderHttpError(error: unknown): Promise<boolean> {
        if (!(error instanceof HttpError) || error.response === undefined) {
            return false;
        }
        const payload = await tryReadErrorPayload(error.response.clone());
        return isNoMatchingNodeProviderErrorPayload(payload);
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

    private async loadMetadataInfo(stackId?: number): Promise<CatmaidStackInfo | null> {
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

    async getDimensions(): Promise<{ min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } } | null> {
        const info = await this.getMetadataInfo();
        if (!info) return null;
        return getCatmaidProjectSpaceBounds(info);
    }

    async getResolution(): Promise<{ x: number; y: number; z: number } | null> {
        const info = await this.getMetadataInfo();
        return info ? info.resolution : null;
    }

    async getGridCellSizes(): Promise<Array<{ x: number; y: number; z: number }>> {
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

    async getSkeleton(skeletonId: number): Promise<SpatiallyIndexedSkeletonNode[]> {
        const data = await this.fetch(`skeletons/${skeletonId}/compact-detail`);
        const nodes = data[0];
        return nodes.map((n: any[]) => ({
            id: n[0],
            parent_id: n[1],
            x: n[3],
            y: n[4],
            z: n[5],
            skeleton_id: skeletonId,
        }));
    }

    async fetchNodes(
        boundingBox: {
            min: { x: number; y: number; z: number };
            max: { x: number; y: number; z: number };
        },
        lod: number = 0,
        cacheProvider?: string,
    ): Promise<SpatiallyIndexedSkeletonNode[]> {
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
        });

        // Add cache provider if available
        if (cacheProvider) {
            params.append("src", cacheProvider);
        }

        let data: any;
        try {
            data = await this.fetch(`node/list?${params.toString()}`, {}, true);
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
            throw new Error("CATMAID node/list endpoint returned an unexpected response format.");
        }

        // Check if limit was reached for the first LOD level
        if (data[3]) {
            console.warn("CATMAID node/list endpoint returned limit_reached=true. Some nodes may be missing.");
        }

        // Process first LOD level (data[0])
        const nodes: SpatiallyIndexedSkeletonNode[] = data[0].map((n: any[]) => ({
            id: n[0],
            parent_id: n[1],
            x: n[2],
            y: n[3],
            z: n[4],
            skeleton_id: n[7],
        }));

        // Process additional LOD levels (data[5] - extraNodes)
        const extraNodes = data[5];
        if (Array.isArray(extraNodes)) {
            for (const lodLevel of extraNodes) {
                // Each lodLevel is [treenodes, connectors, labels, limit_reached, relations]
                if (lodLevel[3]) {
                    console.warn("CATMAID node/list endpoint returned limit_reached=true for an extra LOD level. Some nodes may be missing.");
                }
                const treenodes = lodLevel[0];
                if (Array.isArray(treenodes)) {
                    for (const n of treenodes) {
                        nodes.push({
                            id: n[0],
                            parent_id: n[1],
                            x: n[2],
                            y: n[3],
                            z: n[4],
                            skeleton_id: n[7],
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
    ): Promise<void> {
        const body = new URLSearchParams({
            x: x.toString(),
            y: y.toString(),
            z: z.toString(),
            treenode_id: nodeId.toString(),
        });

        await this.fetch(`node/update`, {
            method: "POST",
            body: body,
        });
    }

    async deleteNode(nodeId: number): Promise<void> {
        const body = new URLSearchParams({
            treenode_id: nodeId.toString(),
        });
        await this.fetch(`treenode/delete`, {
            method: "POST",
            body: body,
        });
    }

    async addNode(
        skeletonId: number,
        x: number,
        y: number,
        z: number,
        parentId?: number,
    ): Promise<number> {
        const body = new URLSearchParams({
            x: x.toString(),
            y: y.toString(),
            z: z.toString(),
            skeleton_id: skeletonId.toString(),
        });
        if (parentId) {
            body.append("parent_id", parentId.toString());
        }

        const res = await this.fetch(`treenode/create`, {
            method: "POST",
            body: body,
        });
        return res.treenode_id;
    }

    async mergeSkeletons(
        skeletonId1: number,
        skeletonId2: number,
    ): Promise<void> {
        void skeletonId1;
        void skeletonId2;
        throw new Error("mergeSkeletons not implemented.");
    }
    async splitSkeleton(nodeId: number): Promise<void> {
        void nodeId;
        throw new Error("splitSkeleton not implemented.");
    }
}
