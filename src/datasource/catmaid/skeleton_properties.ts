/**
 * @license
 * Copyright 2026 Google Inc.
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

import type { CatmaidClient } from "#src/datasource/catmaid/api.js";
import type { InlineSegmentStringProperty } from "#src/segmentation_display_state/property_map.js";
import {
  SegmentPropertyMap,
  normalizeInlineSegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import type { SpatialSkeletonSegmentPropertySource } from "#src/skeleton/api.js";
import { WatchableValue } from "#src/trackable_value.js";

export type CatmaidSkeletonPropertyClient = Pick<
  CatmaidClient,
  "listSkeletons" | "fetchNeuronNames" | "fetchSkeletonAnnotations"
>;

interface CatmaidSkeletonRecord {
  readonly neuronName: string;
  readonly annotationNames: readonly string[];
}

const EMPTY_SKELETON_RECORD: CatmaidSkeletonRecord = {
  neuronName: "",
  annotationNames: [],
};

interface CatmaidSkeletonPropertyColumn {
  readonly id: string;
  readonly type: InlineSegmentStringProperty["type"];
  readonly read: (record: CatmaidSkeletonRecord) => string;
}

const CATMAID_SKELETON_PROPERTY_COLUMNS: readonly CatmaidSkeletonPropertyColumn[] =
  [
    { id: "neuron_name", type: "label", read: (record) => record.neuronName },
    {
      id: "annotations",
      type: "string",
      read: (record) => record.annotationNames.join(" "),
    },
  ];

function makeSkeletonRecord(
  neuronName: string,
  annotationNames: readonly string[] | undefined,
): CatmaidSkeletonRecord {
  return { neuronName, annotationNames: [...(annotationNames ?? [])].sort() };
}

function buildSegmentPropertyMap(
  records: ReadonlyMap<number, CatmaidSkeletonRecord>,
): SegmentPropertyMap {
  const ids = new BigUint64Array(records.size);
  const columnValues = CATMAID_SKELETON_PROPERTY_COLUMNS.map(
    () => new Array<string>(records.size),
  );
  const skeletonIds = [...records.keys()].sort((left, right) => left - right);
  skeletonIds.forEach((skeletonId, index) => {
    ids[index] = BigInt(skeletonId);
    const record = records.get(skeletonId)!;
    CATMAID_SKELETON_PROPERTY_COLUMNS.forEach((column, columnIndex) => {
      columnValues[columnIndex][index] = column.read(record);
    });
  });
  return new SegmentPropertyMap({
    inlineProperties: normalizeInlineSegmentPropertyMap({
      ids,
      properties: CATMAID_SKELETON_PROPERTY_COLUMNS.map(
        (column, columnIndex) => ({
          id: column.id,
          type: column.type,
          values: columnValues[columnIndex],
        }),
      ),
    }),
  });
}

async function fetchOrWarn<T>(
  description: string,
  fetch: () => Promise<T>,
  fallback: T,
): Promise<T> {
  try {
    return await fetch();
  } catch (error) {
    console.warn(`Failed to fetch CATMAID ${description}:`, error);
    return fallback;
  }
}

export class CatmaidSkeletonPropertyProvider
  implements SpatialSkeletonSegmentPropertySource
{
  readonly segmentPropertyMap = new WatchableValue<
    SegmentPropertyMap | undefined
  >(undefined);
  private readonly records = new Map<number, CatmaidSkeletonRecord>();
  private listPromise: Promise<void> | undefined;
  private refreshChain: Promise<void> = Promise.resolve();
  private pendingLoad:
    | { readonly refreshedSkeletonIds: Set<number> }
    | undefined;

  constructor(private readonly client: CatmaidSkeletonPropertyClient) {}

  // Resolves once the skeletons are listed. Names and annotations follow.
  load(): Promise<void> {
    return (this.listPromise ??= this.listSkeletons());
  }

  refreshSegmentProperties(skeletonIds: readonly number[]): Promise<void> {
    const refresh = this.refreshChain.then(() => this.runRefresh(skeletonIds));
    this.refreshChain = refresh.catch(() => {});
    return refresh;
  }

  private async listSkeletons(): Promise<void> {
    const pendingLoad = { refreshedSkeletonIds: new Set<number>() };
    this.pendingLoad = pendingLoad;
    let skeletonIds: number[];
    try {
      skeletonIds = await this.client.listSkeletons();
    } catch (error) {
      this.pendingLoad = undefined;
      throw error;
    }
    for (const skeletonId of skeletonIds) {
      if (!pendingLoad.refreshedSkeletonIds.has(skeletonId)) {
        this.records.set(skeletonId, EMPTY_SKELETON_RECORD);
      }
    }
    this.publish();
    void this.loadNamesAndAnnotations(skeletonIds, pendingLoad);
  }

  private async loadNamesAndAnnotations(
    skeletonIds: readonly number[],
    pendingLoad: { readonly refreshedSkeletonIds: Set<number> },
  ): Promise<void> {
    try {
      const [neuronNames, annotationNames] = await Promise.all([
        fetchOrWarn(
          "neuron names",
          () => this.client.fetchNeuronNames(skeletonIds),
          new Map<number, string>(),
        ),
        fetchOrWarn(
          "skeleton annotations",
          () => this.client.fetchSkeletonAnnotations(skeletonIds),
          new Map<number, string[]>(),
        ),
      ]);
      for (const skeletonId of skeletonIds) {
        if (pendingLoad.refreshedSkeletonIds.has(skeletonId)) {
          continue;
        }
        this.records.set(
          skeletonId,
          makeSkeletonRecord(
            neuronNames.get(skeletonId) ?? "",
            annotationNames.get(skeletonId),
          ),
        );
      }
      this.publish();
    } finally {
      this.pendingLoad = undefined;
    }
  }

  private async runRefresh(skeletonIds: readonly number[]): Promise<void> {
    const [neuronNames, annotationNames] = await Promise.all([
      this.client.fetchNeuronNames(skeletonIds),
      fetchOrWarn(
        "skeleton annotations",
        () => this.client.fetchSkeletonAnnotations(skeletonIds),
        undefined,
      ),
    ]);
    for (const skeletonId of skeletonIds) {
      this.pendingLoad?.refreshedSkeletonIds.add(skeletonId);
      const neuronName = neuronNames.get(skeletonId);
      if (neuronName === undefined) {
        this.records.delete(skeletonId);
        continue;
      }
      this.records.set(
        skeletonId,
        makeSkeletonRecord(
          neuronName,
          annotationNames === undefined
            ? this.records.get(skeletonId)?.annotationNames
            : (annotationNames.get(skeletonId) ?? []),
        ),
      );
    }
    this.publish();
  }

  private publish() {
    this.segmentPropertyMap.value = buildSegmentPropertyMap(this.records);
  }
}
