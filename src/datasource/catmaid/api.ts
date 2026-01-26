
interface CatmaidCacheConfiguration {
    cache_type: string;
    cell_width: number;
    cell_height: number;
    cell_depth: number;
    [key: string]: any;
}

interface CatmaidStackInfo {
    dimension: { x: number; y: number; z: number };
    resolution: { x: number; y: number; z: number };
    translation: { x: number; y: number; z: number };
    title?: string;
    metadata?: {
        cache_configurations?: CatmaidCacheConfiguration[];
    };
}


import { fetchOkWithCredentials } from "#src/credentials_provider/http_request.js";
import type { CredentialsProvider } from "#src/credentials_provider/index.js";
import { SpatiallyIndexedSkeletonNode, SpatiallyIndexedSkeletonSource } from "#src/skeleton/api.js";
import { Unpackr } from "msgpackr";

export interface CatmaidToken {
    token?: string;
}

export const credentialsKey = "CATMAID";

// Default CATMAID cache grid cell dimensions
const DEFAULT_CACHE_GRID_CELL_WIDTH = 25000;
const DEFAULT_CACHE_GRID_CELL_HEIGHT = 25000;
const DEFAULT_CACHE_GRID_CELL_DEPTH = 40;

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
    private stackInfoCache: { stackId: number; promise: Promise<CatmaidStackInfo | null> } | null = null;
    private listStacksCache: Promise<{ id: number; title: string }[]> | null = null;
    
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
                throw new Error(`CATMAID request failed: ${response.statusText}`);
            }
        }
        
        if (expectMsgpack) {
            const buffer = await response.arrayBuffer();
            const unpackr = new Unpackr({
                mapsAsObjects: false,
                int64AsType: "number",
            });
            return unpackr.unpack(new Uint8Array(buffer));
        }
        
        return response.json();
    }

    async listSkeletons(): Promise<number[]> {
        return this.fetch("skeletons/");
    }

    private async listStacks(): Promise<{ id: number; title: string }[]> {
        if (this.listStacksCache === null) {
            this.listStacksCache = this.fetch("stacks");
        }
        return this.listStacksCache;
    }

    private async getStackInfo(stackId: number): Promise<CatmaidStackInfo> {
        return this.fetch(`stack/${stackId}/info`);
    }

    private async getMetadataInfo(stackId?: number): Promise<CatmaidStackInfo | null> {
        // If no stackId is provided and we already have cached stack info,
        // reuse it directly without listing stacks again.
        if (stackId === undefined && this.stackInfoCache !== null) {
            return this.stackInfoCache.promise;
        }
    
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
    
        // Use cache only if it's for the same stackId
        if (
            this.stackInfoCache !== null &&
            this.stackInfoCache.stackId === effectiveStackId
        ) {
            return this.stackInfoCache.promise;
        }
    
        const promise = this.getStackInfo(effectiveStackId);
        this.stackInfoCache = { stackId: effectiveStackId, promise };
        return promise;
    }
    

    async getDimensions(): Promise<{ min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } } | null> {
        const info = await this.getMetadataInfo();
        if (!info) return null;

        // Return voxel-space bounds (dimension from stack info)
        // The "dimension" field in the stack info represents the size in voxels
        const { dimension, translation } = info;

        // Use translation if available, otherwise 0,0,0
        const offX = translation?.x ?? 0;
        const offY = translation?.y ?? 0;
        const offZ = translation?.z ?? 0;

        const min = {
            x: offX,
            y: offY,
            z: offZ,
        };
        const max = {
            x: offX + dimension.x,
            y: offY + dimension.y,
            z: offZ + dimension.z,
        };
        return { min, max };
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
    ): Promise<SpatiallyIndexedSkeletonNode[]> {
        const params = new URLSearchParams({
            left: boundingBox.min.x.toString(),
            top: boundingBox.min.y.toString(),
            z1: boundingBox.min.z.toString(),
            right: boundingBox.max.x.toString(),
            bottom: boundingBox.max.y.toString(),
            z2: boundingBox.max.z.toString(),
            lod_type: 'percent',
            lod: lod.toString(),
            format: 'msgpack',
        });

        const data = await this.fetch(`node/list?${params.toString()}`, {}, true);

        // Check if limit was reached for the first LOD level
        if (data[4]) {
            console.warn("CATMAID node/list endpoint returned limit_reached=true. Some nodes may be missing.");
        }

        // Process first LOD level (data[0])
        const nodes: SpatiallyIndexedSkeletonNode[] = data[0].map((n: any[]) => ({
            id: n[0],
            parent_id: n[1],
            x: n[2],
            y: n[3],
            z: n[4],
            confidence: n[5],
            radius: n[6],
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
                            confidence: n[5],
                            radius: n[6],
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
