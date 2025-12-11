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

export interface CatmaidNode {
    id: number;
    parent_id: number | null;
    x: number;
    y: number;
    z: number;
    radius: number;
    confidence: number;
    skeleton_id: number;
}

export interface SkeletonSummary {
    skeleton_id: number;
    num_nodes: number;
    cable_length: number;
}

export interface CatmaidStackInfo {
    dimension: { x: number; y: number; z: number };
    resolution: { x: number; y: number; z: number };
    translation: { x: number; y: number; z: number };
    title?: string;
}

export interface CatmaidSource {
    listSkeletons(): Promise<number[]>;
    getSkeleton(skeletonId: number): Promise<CatmaidNode[]>;
    getDimensions(): Promise<{ min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } } | null>;
    getResolution(): Promise<{ x: number; y: number; z: number } | null>;
    fetchNodes(boundingBox: { min: { x: number, y: number, z: number }, max: { x: number, y: number, z: number } }): Promise<CatmaidNode[]>;
    addNode(
        skeletonId: number,
        x: number,
        y: number,
        z: number,
        parentId?: number,
    ): Promise<number>;
    moveNode(nodeId: number, x: number, y: number, z: number): Promise<void>;
    deleteNode(nodeId: number): Promise<void>;

    mergeSkeletons(skeletonId1: number, skeletonId2: number): Promise<void>;
    splitSkeleton(nodeId: number): Promise<void>;
}

export class CatmaidClient implements CatmaidSource {
    constructor(
        public baseUrl: string,
        public projectId: number,
        public token?: string,
    ) { }

    private async fetch(
        endpoint: string,
        options: RequestInit = {},
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

        const response = await fetch(url, { ...options, headers });
        if (!response.ok) {
            throw new Error(`CATMAID request failed: ${response.statusText}`);
        }
        return response.json();
    }

    async listSkeletons(): Promise<number[]> {
        return this.fetch("skeletons/");
    }

    async listStacks(): Promise<{ id: number; title: string }[]> {
        return this.fetch("stacks");
    }

    async getStackInfo(stackId: number): Promise<CatmaidStackInfo> {
        return this.fetch(`stack/${stackId}/info`);
    }

    private async getMetadataInfo(stackId?: number): Promise<CatmaidStackInfo | null> {
        if (stackId !== undefined) {
            return this.getStackInfo(stackId);
        }
        try {
            const stacks = await this.listStacks();
            if (!stacks || stacks.length === 0) return null;
            const targetId = stacks[0].id;
            return this.getStackInfo(targetId);
        } catch (e) {
            console.warn("Failed to fetch stack info:", e);
            return null;
        }
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

    async getSkeleton(skeletonId: number): Promise<CatmaidNode[]> {
        const data = await this.fetch(`skeletons/${skeletonId}/compact-detail`);
        const nodes = data[0];
        return nodes.map((n: any[]) => ({
            id: n[0],
            parent_id: n[1],
            x: n[3],
            y: n[4],
            z: n[5],
            radius: n[6],
            confidence: n[7],
            skeleton_id: skeletonId,
        }));
    }

    async fetchNodes(
        boundingBox: {
            min: { x: number; y: number; z: number };
            max: { x: number; y: number; z: number };
        },
    ): Promise<CatmaidNode[]> {
        const body = new URLSearchParams({
            left: boundingBox.min.x.toString(),
            top: boundingBox.min.y.toString(),
            z1: boundingBox.min.z.toString(),
            right: boundingBox.max.x.toString(),
            bottom: boundingBox.max.y.toString(),
            z2: boundingBox.max.z.toString(),
        });

        const data = await this.fetch("nodes/", {
            method: "POST",
            body: body,
        });

        return data.map((n: any[]) => ({
            id: n[0],
            parent_id: n[1],
            x: n[2],
            y: n[3],
            z: n[4],
            confidence: n[5],
            radius: n[6],
            skeleton_id: n[7],
        }));
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
