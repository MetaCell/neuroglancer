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

export interface CatmaidSource {
    listSkeletons(): Promise<number[]>;
    getSkeleton(skeletonId: number): Promise<CatmaidNode[]>;
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

    async getSkeleton(skeletonId: number): Promise<CatmaidNode[]> {
        const data = await this.fetch(`skeletons/${skeletonId}/compact-detail`);
        // data[0] contains the nodes
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
        // Placeholder implementation; parameters referenced to satisfy noUnusedParameters.
        void skeletonId1;
        void skeletonId2;
        throw new Error("mergeSkeletons not implemented.");
    }
    async splitSkeleton(nodeId: number): Promise<void> {
        void nodeId;
        throw new Error("splitSkeleton not implemented.");
    }
}
