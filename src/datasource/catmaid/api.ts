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


import { vec3 } from "#src/util/geom.js";

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

export class CatmaidClient {
    // Mock data storage for testing without a real CATMAID server
    private static mockNodes: Map<number, CatmaidNode> = new Map();
    private static nextNodeId = 1000;
    private static useMockData = true; // Set to false to use real CATMAID server

    constructor(
        public baseUrl: string,
        public projectId: number,
        public token?: string
    ) {
        // Initialize mock data if not already done
        if (CatmaidClient.mockNodes.size === 0) {
            this.initializeMockData();
        }
    }

    private initializeMockData() {
        console.log('Initializing CATMAID mock data...');
        
        // Create a sample neuron with multiple branches
        // Main trunk along z-axis - positioned at [500, 500, 500] for easy viewing
        const nodes: CatmaidNode[] = [
            // Root node
            { id: 1, parent_id: null, x: 500, y: 500, z: 500, radius: 8, confidence: 5, skeleton_id: 1 },
            // Main trunk
            { id: 2, parent_id: 1, x: 500, y: 500, z: 600, radius: 6, confidence: 5, skeleton_id: 1 },
            { id: 3, parent_id: 2, x: 500, y: 500, z: 700, radius: 5, confidence: 5, skeleton_id: 1 },
            { id: 4, parent_id: 3, x: 500, y: 500, z: 800, radius: 5, confidence: 5, skeleton_id: 1 },
            { id: 5, parent_id: 4, x: 500, y: 500, z: 900, radius: 4, confidence: 5, skeleton_id: 1 },
            
            // Branch 1 - extending in +x direction
            { id: 6, parent_id: 3, x: 600, y: 500, z: 700, radius: 4, confidence: 5, skeleton_id: 1 },
            { id: 7, parent_id: 6, x: 700, y: 500, z: 700, radius: 3, confidence: 5, skeleton_id: 1 },
            { id: 8, parent_id: 7, x: 800, y: 500, z: 750, radius: 3, confidence: 4, skeleton_id: 1 },
            
            // Branch 2 - extending in -x direction
            { id: 9, parent_id: 3, x: 400, y: 500, z: 700, radius: 4, confidence: 5, skeleton_id: 1 },
            { id: 10, parent_id: 9, x: 300, y: 500, z: 700, radius: 3, confidence: 5, skeleton_id: 1 },
            
            // Branch 3 - extending in +y direction from node 4
            { id: 11, parent_id: 4, x: 500, y: 600, z: 800, radius: 4, confidence: 5, skeleton_id: 1 },
            { id: 12, parent_id: 11, x: 500, y: 700, z: 850, radius: 3, confidence: 5, skeleton_id: 1 },
            { id: 13, parent_id: 12, x: 550, y: 750, z: 900, radius: 2, confidence: 4, skeleton_id: 1 },
            
            // Branch 4 - extending in -y direction from node 4
            { id: 14, parent_id: 4, x: 500, y: 400, z: 800, radius: 4, confidence: 5, skeleton_id: 1 },
            { id: 15, parent_id: 14, x: 500, y: 300, z: 850, radius: 3, confidence: 5, skeleton_id: 1 },
        ];

        // Add a second neuron nearby
        const neuron2Nodes: CatmaidNode[] = [
            { id: 100, parent_id: null, x: 1000, y: 1000, z: 500, radius: 7, confidence: 5, skeleton_id: 2 },
            { id: 101, parent_id: 100, x: 1000, y: 1000, z: 600, radius: 6, confidence: 5, skeleton_id: 2 },
            { id: 102, parent_id: 101, x: 950, y: 950, z: 700, radius: 5, confidence: 5, skeleton_id: 2 },
            { id: 103, parent_id: 102, x: 900, y: 900, z: 800, radius: 4, confidence: 5, skeleton_id: 2 },
            { id: 104, parent_id: 103, x: 850, y: 850, z: 900, radius: 3, confidence: 4, skeleton_id: 2 },
        ];

        [...nodes, ...neuron2Nodes].forEach(node => {
            CatmaidClient.mockNodes.set(node.id, node);
        });

        CatmaidClient.nextNodeId = 200;
        console.log(`Initialized ${CatmaidClient.mockNodes.size} mock nodes`);
    }

    private async fetch(
        endpoint: string,
        options: RequestInit = {}
    ): Promise<any> {
        const url = `${this.baseUrl}/${this.projectId}/${endpoint}`;
        const headers = new Headers(options.headers);
        if (this.token) {
            headers.append("X-Authorization", `Token ${this.token}`);
        }
        const response = await fetch(url, { ...options, headers });
        if (!response.ok) {
            throw new Error(`CATMAID request failed: ${response.statusText}`);
        }
        return response.json();
    }

    async getSkeleton(skeletonId: number): Promise<CatmaidNode[]> {
        if (CatmaidClient.useMockData) {
            console.log(`Mock: Getting skeleton ${skeletonId}`);
            const nodes = Array.from(CatmaidClient.mockNodes.values())
                .filter(n => n.skeleton_id === skeletonId);
            console.log(`Mock: Found ${nodes.length} nodes for skeleton ${skeletonId}`);
            return nodes;
        }

        // Real implementation
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

    async boxQuery(min: vec3, max: vec3): Promise<CatmaidNode[]> {
        if (CatmaidClient.useMockData) {
            console.log(`Mock: Box query [${min[0]}, ${min[1]}, ${min[2]}] to [${max[0]}, ${max[1]}, ${max[2]}]`);
            const nodes = Array.from(CatmaidClient.mockNodes.values()).filter(node => {
                return node.x >= min[0] && node.x <= max[0] &&
                       node.y >= min[1] && node.y <= max[1] &&
                       node.z >= min[2] && node.z <= max[2];
            });
            console.log(`Mock: Found ${nodes.length} nodes in box`);
            return nodes;
        }

        // Real implementation
        const body = new URLSearchParams({
            left: min[0].toString(),
            top: min[1].toString(),
            z1: min[2].toString(),
            right: max[0].toString(),
            bottom: max[1].toString(),
            z2: max[2].toString(),
        });

        const data = await this.fetch(`nodes/`, {
            method: "POST",
            body: body,
        });

        const nodes = data[0];
        return nodes.map((n: any[]) => ({
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
        z: number
    ): Promise<void> {
        if (CatmaidClient.useMockData) {
            console.log(`Mock: Moving node ${nodeId} to [${x}, ${y}, ${z}]`);
            const node = CatmaidClient.mockNodes.get(nodeId);
            if (node) {
                node.x = x;
                node.y = y;
                node.z = z;
                console.log(`Mock: Node ${nodeId} moved successfully`);
            } else {
                console.warn(`Mock: Node ${nodeId} not found`);
            }
            return;
        }

        // Real implementation
        await this.fetch(`node/${nodeId}/move`, {
            method: "POST",
            body: JSON.stringify({ x, y, z }),
        });
    }

    async deleteNode(nodeId: number): Promise<void> {
        if (CatmaidClient.useMockData) {
            console.log(`Mock: Deleting node ${nodeId}`);
            const deleted = CatmaidClient.mockNodes.delete(nodeId);
            if (deleted) {
                console.log(`Mock: Node ${nodeId} deleted successfully`);
            } else {
                console.warn(`Mock: Node ${nodeId} not found`);
            }
            return;
        }

        // Real implementation
        await this.fetch(`node/${nodeId}/delete`, {
            method: "POST",
        });
    }

    async addNode(
        skeletonId: number,
        x: number,
        y: number,
        z: number,
        parentId?: number
    ): Promise<number> {
        if (CatmaidClient.useMockData) {
            const nodeId = CatmaidClient.nextNodeId++;
            console.log(`Mock: Adding node ${nodeId} at [${x}, ${y}, ${z}] with parent ${parentId} to skeleton ${skeletonId}`);
            
            const newNode: CatmaidNode = {
                id: nodeId,
                parent_id: parentId || null,
                x,
                y,
                z,
                radius: 3,
                confidence: 5,
                skeleton_id: skeletonId,
            };
            
            CatmaidClient.mockNodes.set(nodeId, newNode);
            console.log(`Mock: Node ${nodeId} added successfully`);
            return nodeId;
        }

        // Real implementation
        const res = await this.fetch(`node/add`, {
            method: "POST",
            body: JSON.stringify({
                skeleton_id: skeletonId,
                x,
                y,
                z,
                parent_id: parentId,
            }),
        });
        return res.node_id;
    }
}
