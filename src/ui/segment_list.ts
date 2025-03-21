/**
 * @license
 * Copyright 2020 Google Inc.
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

import "#src/ui/segment_list.css";

import type { DebouncedFunc } from "lodash-es";
import { debounce, throttle } from "lodash-es";
import type {
  SegmentationUserLayer,
  SegmentationUserLayerGroupState,
} from "#src/layer/segmentation/index.js";
import type { SegmentationDisplayState } from "#src/segmentation_display_state/frontend.js";
import {
  registerCallbackWhenSegmentationDisplayStateChanged,
  SegmentWidgetWithExtraColumnsFactory,
} from "#src/segmentation_display_state/frontend.js";
import type {
  ExplicitIdQuery,
  FilterQuery,
  InlineSegmentNumericalProperty,
  NumericalPropertyConstraint,
  PreprocessedSegmentPropertyMap,
  PropertyHistogram,
  QueryResult,
} from "#src/segmentation_display_state/property_map.js";
import {
  changeTagConstraintInSegmentQuery,
  executeSegmentQuery,
  findQueryResultIntersectionSize,
  forEachQueryResultSegmentIdGenerator,
  isQueryUnconstrained,
  parseSegmentQuery,
  queryIncludesColumn,
  unparseSegmentQuery,
  updatePropertyHistograms,
} from "#src/segmentation_display_state/property_map.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { observeWatchable, WatchableValue } from "#src/trackable_value.js";
import { getDefaultSelectBindings } from "#src/ui/default_input_event_bindings.js";
import { SELECT_SEGMENTS_TOOLS_ID } from "#src/ui/segment_select_tools.js";
import {
  ANNOTATE_MERGE_SEGMENTS_TOOL_ID,
  ANNOTATE_SPLIT_SEGMENTS_TOOL_ID,
} from "#src/ui/segment_split_merge_tools.js";
import { makeToolButton } from "#src/ui/tool.js";
import { animationFrameDebounce } from "#src/util/animation_frame_debounce.js";
import type { ArraySpliceOp } from "#src/util/array.js";
import { getFixedOrderMergeSplices } from "#src/util/array.js";
import { bigintCompare } from "#src/util/bigint.js";
import { setClipboard } from "#src/util/clipboard.js";
import { RefCounted } from "#src/util/disposable.js";
import { removeChildren, updateInputFieldWidth } from "#src/util/dom.js";
import {
  EventActionMap,
  KeyboardEventBinder,
  registerActionListener,
} from "#src/util/keyboard_bindings.js";
import type { DataTypeInterval } from "#src/util/lerp.js";
import {
  clampToInterval,
  computeInvlerp,
  dataTypeCompare,
  dataTypeIntervalEqual,
  getClampedInterval,
  getIntervalBoundsEffectiveFraction,
  parseDataTypeValue,
} from "#src/util/lerp.js";
import { MouseEventBinder } from "#src/util/mouse_bindings.js";
import { neverSignal, NullarySignal, Signal } from "#src/util/signal.js";
import { CheckboxIcon } from "#src/widget/checkbox_icon.js";
import { makeCopyButton } from "#src/widget/copy_button.js";
import { DependentViewWidget } from "#src/widget/dependent_view_widget.js";
import { makeEyeButton } from "#src/widget/eye_button.js";
import type { RangeAndWindowIntervals } from "#src/widget/invlerp.js";
import {
  CdfController,
  getUpdatedRangeAndWindowParameters,
} from "#src/widget/invlerp.js";
import { makeStarButton } from "#src/widget/star_button.js";
import { Tab } from "#src/widget/tab_view.js";
import type { VirtualListSource } from "#src/widget/virtual_list.js";
import { VirtualList } from "#src/widget/virtual_list.js";

abstract class SegmentListSource
  extends RefCounted
  implements VirtualListSource
{
  length: number;
  changed = new Signal<(splices: readonly Readonly<ArraySpliceOp>[]) => void>();

  // The segment list is the concatenation of two lists: the `explicitSegments` list, specified as
  // explicit uint64 ids, and the `matches`, list, specifying the indices into the
  // `segmentPropertyMap` of the matching segments.
  explicitSegments: bigint[] | undefined;

  debouncedUpdate = debounce(() => this.update(), 0);

  constructor(
    public segmentationDisplayState: SegmentationDisplayState,
    public parentElement: HTMLElement,
  ) {
    super();
  }

  abstract update(): void;

  private updateRendering(element: HTMLElement) {
    this.segmentWidgetFactory.update(element);
  }

  segmentWidgetFactory: SegmentWidgetWithExtraColumnsFactory;

  abstract render(index: number): HTMLDivElement;

  updateRenderedItems(list: VirtualList) {
    list.forEachRenderedItem((element) => {
      this.updateRendering(element);
    });
  }
}

class StarredSegmentsListSource extends SegmentListSource {
  constructor(
    public segmentationDisplayState: SegmentationDisplayState,
    public parentElement: HTMLElement,
  ) {
    super(segmentationDisplayState, parentElement);
    this.update();
    this.registerDisposer(
      segmentationDisplayState.segmentationGroupState.value.selectedSegments.changed.add(
        this.debouncedUpdate,
      ),
    );
  }

  update() {
    const splices: ArraySpliceOp[] = [];
    const { selectedSegments } =
      this.segmentationDisplayState.segmentationGroupState.value;
    const newSelectedSegments = [...selectedSegments];
    const { explicitSegments } = this;
    if (explicitSegments === undefined) {
      splices.push({
        retainCount: 0,
        insertCount: newSelectedSegments.length,
        deleteCount: 0,
      });
    } else {
      splices.push(
        ...getFixedOrderMergeSplices(
          explicitSegments,
          newSelectedSegments,
          (a, b) => a === b,
        ),
      );
    }
    this.explicitSegments = newSelectedSegments;
    this.length = newSelectedSegments.length;
    this.changed.dispatch(splices);
  }

  render = (index: number) => {
    const { explicitSegments } = this;
    const id = explicitSegments![index];
    return this.segmentWidgetFactory.get(id);
  };
}

class SegmentQueryListSource extends SegmentListSource {
  prevQuery: string | undefined;
  queryResult = new WatchableValue<QueryResult | undefined>(undefined);
  prevQueryResult = new WatchableValue<QueryResult | undefined>(undefined);
  statusText = new WatchableValue<string>("");
  selectedMatches = 0;
  visibleMatches = 0;
  matchStatusTextPrefix = "";
  selectedSegmentsGeneration = -1;
  visibleSegmentsGeneration = -1;

  get numMatches() {
    return this.queryResult.value?.count ?? 0;
  }

  update() {
    const query = this.query.value;
    const { segmentPropertyMap } = this;
    this.prevQueryResult.value = this.queryResult.value;
    const prevQueryResult = this.prevQueryResult.value;
    let queryResult: QueryResult;
    if (this.prevQuery === query) {
      queryResult = prevQueryResult!;
    } else {
      const queryParseResult = parseSegmentQuery(segmentPropertyMap, query);
      queryResult = executeSegmentQuery(segmentPropertyMap, queryParseResult);
    }

    const splices: ArraySpliceOp[] = [];
    let changed = false;
    let matchStatusTextPrefix = "";
    const unconstrained = isQueryUnconstrained(queryResult.query);
    if (!unconstrained) {
      if (this.explicitSegments !== undefined) {
        splices.push({
          deleteCount: this.explicitSegments.length,
          retainCount: 0,
          insertCount: 0,
        });
        this.explicitSegments = undefined;
        changed = true;
      }
    }

    const { explicitIds } = queryResult;
    if (explicitIds !== undefined) {
      this.explicitSegments = explicitIds;
    } else {
      this.explicitSegments = undefined;
    }

    if (prevQueryResult !== queryResult) {
      splices.push({
        retainCount: 0,
        deleteCount: prevQueryResult?.count ?? 0,
        insertCount: queryResult.count,
      });
      changed = true;
      this.queryResult.value = queryResult;
    }

    if (queryResult.explicitIds !== undefined) {
      matchStatusTextPrefix = `${queryResult.count} ids`;
    } else if (unconstrained) {
      matchStatusTextPrefix = `${queryResult.count} total ids`;
    } else if (queryResult.total > 0) {
      matchStatusTextPrefix = `${queryResult.count} match /${queryResult.total} total ids`;
    }

    const { selectedSegments, visibleSegments } =
      this.segmentationDisplayState.segmentationGroupState.value;
    const selectedSegmentsGeneration = selectedSegments.changed.count;
    const visibleSegmentsGeneration = visibleSegments.changed.count;
    const prevSelectedSegmentsGeneration = this.selectedSegmentsGeneration;
    const prevVisibleSegmentsGeneration = this.visibleSegmentsGeneration;
    const queryChanged = prevQueryResult !== queryResult;
    const selectedChanged =
      prevSelectedSegmentsGeneration !== selectedSegmentsGeneration ||
      queryChanged;
    const visibleChanged =
      prevVisibleSegmentsGeneration !== visibleSegmentsGeneration ||
      queryChanged;
    this.selectedSegmentsGeneration = selectedSegmentsGeneration;
    this.visibleSegmentsGeneration = visibleSegmentsGeneration;

    if (selectedChanged) {
      this.selectedMatches =
        queryResult.count > 0
          ? findQueryResultIntersectionSize(
              segmentPropertyMap,
              queryResult,
              selectedSegments,
            )
          : 0;
    }

    if (visibleChanged) {
      this.visibleMatches =
        queryResult.count > 0
          ? findQueryResultIntersectionSize(
              segmentPropertyMap,
              queryResult,
              visibleSegments,
            )
          : 0;
    }

    let fullStatusText = matchStatusTextPrefix;
    if (this.selectedMatches > 0) {
      if (this.selectedMatches === this.visibleMatches) {
        fullStatusText = `${this.selectedMatches} vis/${fullStatusText}`;
      } else if (this.visibleMatches > 0) {
        fullStatusText = `${this.visibleMatches} vis/${this.selectedMatches} star/${fullStatusText}`;
      } else {
        fullStatusText = `${this.selectedMatches} star/${fullStatusText}`;
      }
    }

    this.statusText.value = fullStatusText;

    this.prevQuery = query;
    this.matchStatusTextPrefix = matchStatusTextPrefix;
    this.length = queryResult.count;
    if (changed) {
      this.changed.dispatch(splices);
    }
  }

  constructor(
    public query: WatchableValueInterface<string>,
    public segmentPropertyMap: PreprocessedSegmentPropertyMap | undefined,
    public segmentationDisplayState: SegmentationDisplayState,
    public parentElement: HTMLElement,
  ) {
    super(segmentationDisplayState, parentElement);
    this.update();
    this.registerDisposer(
      segmentationDisplayState.segmentationGroupState.value.selectedSegments.changed.add(
        this.debouncedUpdate,
      ),
    ); // to update statusText
    this.registerDisposer(
      segmentationDisplayState.segmentationGroupState.value.visibleSegments.changed.add(
        this.debouncedUpdate,
      ),
    ); // to update statusText
    if (query) {
      this.registerDisposer(query.changed.add(this.debouncedUpdate));
    }
  }

  render = (index: number) => {
    const { explicitSegments } = this;
    let id: bigint;
    if (explicitSegments !== undefined) {
      id = explicitSegments[index];
    } else {
      const propIndex = this.queryResult.value!.indices![index];
      const { ids } =
        this.segmentPropertyMap!.segmentPropertyMap.inlineProperties!;
      id = ids[propIndex];
    }
    return this.segmentWidgetFactory.get(id);
  };
}

const keyMap = EventActionMap.fromObject({
  enter: { action: "toggle-listed" },
  "shift+enter": { action: "hide-listed" },
  "control+enter": { action: "hide-all" },
  escape: { action: "cancel" },
});

interface NumericalBoundElements {
  container: HTMLElement;
  inputs: [HTMLInputElement, HTMLInputElement];
  spacers: [HTMLElement, HTMLElement, HTMLElement] | undefined;
}

interface NumericalPropertySummary {
  element: HTMLElement;
  controller: CdfController<RangeAndWindowIntervals>;
  property: InlineSegmentNumericalProperty;
  boundElements: {
    window: NumericalBoundElements;
    range: NumericalBoundElements;
  };
  plotImg: HTMLImageElement;
  propertyHistogram: PropertyHistogram | undefined;
  bounds: RangeAndWindowIntervals;
  columnCheckbox: HTMLInputElement;
  sortIcon: HTMLElement;
}

function updateInputBoundWidth(inputElement: HTMLInputElement) {
  updateInputFieldWidth(
    inputElement,
    Math.max(1, inputElement.value.length + 0.1),
  );
}

function updateInputBoundValue(inputElement: HTMLInputElement, bound: number) {
  let boundString: string;
  if (Number.isInteger(bound)) {
    boundString = bound.toString();
  } else {
    const sFull = bound.toString();
    const sPrecision = bound.toPrecision(6);
    boundString = sFull.length < sPrecision.length ? sFull : sPrecision;
  }
  inputElement.value = boundString;
  updateInputBoundWidth(inputElement);
}

function createBoundInput(boundType: "range" | "window", endpointIndex: 0 | 1) {
  const e = document.createElement("input");
  e.addEventListener("focus", () => {
    e.select();
  });
  e.classList.add(
    `neuroglancer-segment-query-result-numerical-plot-${boundType}-bound`,
  );
  e.classList.add("neuroglancer-segment-query-result-numerical-plot-bound");
  e.type = "text";
  e.spellcheck = false;
  e.autocomplete = "off";
  e.title =
    (endpointIndex === 0 ? "Lower" : "Upper") +
    " bound " +
    (boundType === "range" ? "range" : "for distribution");
  e.addEventListener("input", () => {
    updateInputBoundWidth(e);
  });
  return e;
}

function toggleIncludeColumn(
  queryResult: QueryResult | undefined,
  setQuery: (query: FilterQuery) => void,
  fieldId: string,
) {
  if (queryResult === undefined) return;
  if (queryResult.indices === undefined) return;
  const query = queryResult.query as FilterQuery;
  let { sortBy, includeColumns } = query;
  const included = queryIncludesColumn(query, fieldId);
  if (included) {
    sortBy = sortBy.filter((x) => x.fieldId !== fieldId);
    includeColumns = includeColumns.filter((x) => x !== fieldId);
  } else {
    includeColumns.push(fieldId);
  }
  setQuery({ ...query, sortBy, includeColumns });
}

function toggleSortOrder(
  queryResult: QueryResult | undefined,
  setQuery: (query: FilterQuery) => void,
  id: string,
) {
  const query = queryResult?.query;
  const sortBy = query?.sortBy;
  if (sortBy === undefined) return;
  const { includeColumns } = query as FilterQuery;
  const prevOrder = sortBy.find((x) => x.fieldId === id)?.order;
  const newOrder = prevOrder === "<" ? ">" : "<";
  const newIncludeColumns = includeColumns.filter((x) => x !== id);
  for (const s of sortBy) {
    if (s.fieldId !== "id" && s.fieldId !== "label" && s.fieldId !== id) {
      newIncludeColumns.push(s.fieldId);
    }
  }
  setQuery({
    ...(query as FilterQuery),
    sortBy: [{ fieldId: id, order: newOrder }],
    includeColumns: newIncludeColumns,
  });
}

function updateColumnSortIcon(
  queryResult: QueryResult | undefined,
  sortIcon: HTMLElement,
  id: string,
) {
  const sortBy = queryResult?.query?.sortBy;
  const order = sortBy?.find((s) => s.fieldId === id)?.order;
  sortIcon.textContent = order === ">" ? "▼" : "▲";
  sortIcon.style.visibility = order === undefined ? "" : "visible";
  sortIcon.title = `Sort by ${id} in ${
    order === "<" ? "descending" : "ascending"
  } order`;
}

class NumericalPropertiesSummary extends RefCounted {
  listElement: HTMLElement | undefined;
  properties: NumericalPropertySummary[];
  propertyHistograms: PropertyHistogram[] = [];
  bounds = {
    window: new WatchableValue<DataTypeInterval[]>([]),
    range: new WatchableValue<DataTypeInterval[]>([]),
  };

  throttledUpdate = this.registerCancellable(
    throttle(() => this.updateHistograms(), 100),
  );
  debouncedRender = this.registerCancellable(
    animationFrameDebounce(() => this.updateHistogramRenderings()),
  );
  debouncedSetQuery = this.registerCancellable(
    debounce(() => this.setQueryFromBounds(), 200),
  );

  constructor(
    public segmentPropertyMap: PreprocessedSegmentPropertyMap | undefined,
    public queryResult: WatchableValueInterface<QueryResult | undefined>,
    public setQuery: (query: FilterQuery) => void,
  ) {
    super();
    const properties = segmentPropertyMap?.numericalProperties;
    const propertySummaries: NumericalPropertySummary[] = [];
    let listElement: HTMLElement | undefined;
    if (properties !== undefined && properties.length > 0) {
      listElement = document.createElement("details");
      const summaryElement = document.createElement("summary");
      summaryElement.textContent = `${properties.length} numerical propert${
        properties.length > 1 ? "ies" : "y"
      }`;
      listElement.appendChild(summaryElement);
      listElement.classList.add(
        "neuroglancer-segment-query-result-numerical-list",
      );
      const windowBounds = this.bounds.window.value;
      for (
        let i = 0, numProperties = properties.length;
        i < numProperties;
        ++i
      ) {
        const property = properties[i];
        const summary = this.makeNumericalPropertySummary(i, property);
        propertySummaries.push(summary);
        listElement.appendChild(summary.element);
        windowBounds[i] = property.bounds;
      }
    }
    this.listElement = listElement;
    this.properties = propertySummaries;
    this.registerDisposer(
      this.queryResult.changed.add(() => {
        this.handleNewQueryResult();
      }),
    );
    // When window bounds change, we need to recompute histograms.  Throttle this to avoid
    // excessive computation time.
    this.registerDisposer(this.bounds.window.changed.add(this.throttledUpdate));
    // When window bounds or constraint bounds change, re-render the plot on the next animation
    // frame.
    this.registerDisposer(this.bounds.window.changed.add(this.debouncedRender));
    this.registerDisposer(this.bounds.range.changed.add(this.debouncedRender));
    this.registerDisposer(
      this.bounds.range.changed.add(this.debouncedSetQuery),
    );
    this.handleNewQueryResult();
  }

  private setQueryFromBounds() {
    const queryResult = this.queryResult.value;
    if (queryResult === undefined) return;
    if (queryResult.indices === undefined) return;
    const query = queryResult.query as FilterQuery;
    const numericalConstraints: NumericalPropertyConstraint[] = [];
    const constraintBounds = this.bounds.range.value;
    const { properties } = this;
    for (let i = 0, numProperties = properties.length; i < numProperties; ++i) {
      const property = properties[i].property;
      numericalConstraints.push({
        fieldId: property.id,
        bounds: constraintBounds[i],
      });
    }
    this.setQuery({ ...query, numericalConstraints });
  }

  private getBounds(propertyIndex: number) {
    const { bounds } = this;
    return {
      range: bounds.range.value[propertyIndex],
      window: bounds.window.value[propertyIndex],
    };
  }

  private setBounds(propertyIndex: number, value: RangeAndWindowIntervals) {
    const { property } = this.properties[propertyIndex];
    let newRange = getClampedInterval(property.bounds, value.range);
    if (dataTypeCompare(newRange[0], newRange[1]) > 0) {
      newRange = [newRange[1], newRange[0]] as DataTypeInterval;
    }
    const newWindow = getClampedInterval(property.bounds, value.window);
    const oldValue = this.getBounds(propertyIndex);
    if (!dataTypeIntervalEqual(newWindow, oldValue.window)) {
      this.bounds.window.value[propertyIndex] = newWindow;
      this.bounds.window.changed.dispatch();
    }
    if (!dataTypeIntervalEqual(newRange, oldValue.range)) {
      this.bounds.range.value[propertyIndex] = newRange;
      this.bounds.range.changed.dispatch();
    }
  }

  private setBound(
    boundType: "range" | "window",
    endpoint: 0 | 1,
    propertyIndex: number,
    value: number,
  ) {
    const property =
      this.segmentPropertyMap!.numericalProperties[propertyIndex];
    const baseBounds = property.bounds;
    value = clampToInterval(baseBounds, value) as number;
    const params = this.getBounds(propertyIndex);
    const newParams = getUpdatedRangeAndWindowParameters(
      params,
      boundType,
      endpoint,
      value,
      /*fitRangeInWindow=*/ true,
    );
    this.setBounds(propertyIndex, newParams);
  }

  private handleNewQueryResult() {
    const queryResult = this.queryResult.value;
    const { listElement } = this;
    if (listElement === undefined) return;
    if (queryResult?.indices !== undefined) {
      const { numericalConstraints } = queryResult!.query as FilterQuery;
      const { numericalProperties } = this.segmentPropertyMap!;
      const constraintBounds = this.bounds.range.value;
      const numConstraints = numericalConstraints.length;
      const numProperties = numericalProperties.length;
      constraintBounds.length = numProperties;
      for (let i = 0; i < numProperties; ++i) {
        constraintBounds[i] = numericalProperties[i].bounds;
      }
      for (let i = 0; i < numConstraints; ++i) {
        const constraint = numericalConstraints[i];
        const propertyIndex = numericalProperties.findIndex(
          (p) => p.id === constraint.fieldId,
        );
        constraintBounds[propertyIndex] = constraint.bounds;
      }
    }
    this.updateHistograms();
    this.throttledUpdate.cancel();
  }

  private updateHistograms() {
    const queryResult = this.queryResult.value;
    const { listElement } = this;
    if (listElement === undefined) return;
    updatePropertyHistograms(
      this.segmentPropertyMap,
      queryResult,
      this.propertyHistograms,
      this.bounds.window.value,
    );
    this.updateHistogramRenderings();
  }

  private updateHistogramRenderings() {
    this.debouncedRender.cancel();
    const { listElement } = this;
    if (listElement === undefined) return;
    const { propertyHistograms } = this;
    if (propertyHistograms.length === 0) {
      listElement.style.display = "none";
      return;
    }
    listElement.style.display = "";
    const { properties } = this;
    for (let i = 0, n = properties.length; i < n; ++i) {
      this.updateNumericalPropertySummary(
        i,
        properties[i],
        propertyHistograms[i],
      );
    }
  }

  private makeNumericalPropertySummary(
    propertyIndex: number,
    property: InlineSegmentNumericalProperty,
  ): NumericalPropertySummary {
    const plotContainer = document.createElement("div");
    plotContainer.classList.add(
      "neuroglancer-segment-query-result-numerical-plot-container",
    );
    const plotImg = document.createElement("img");
    plotImg.classList.add("neuroglancer-segment-query-result-numerical-plot");
    const controller = new CdfController(
      plotImg,
      property.dataType,
      () => this.getBounds(propertyIndex),
      (bounds) => this.setBounds(propertyIndex, bounds),
    );
    const sortIcon = document.createElement("span");
    sortIcon.classList.add(
      "neuroglancer-segment-query-result-numerical-plot-sort",
    );
    const columnCheckbox = document.createElement("input");
    columnCheckbox.type = "checkbox";
    columnCheckbox.addEventListener("click", () => {
      toggleIncludeColumn(this.queryResult.value, this.setQuery, property.id);
    });
    const makeBoundElements = (
      boundType: "window" | "range",
    ): NumericalBoundElements => {
      const container = document.createElement("div");
      container.classList.add(
        "neuroglancer-segment-query-result-numerical-plot-bounds",
      );
      container.classList.add(
        `neuroglancer-segment-query-result-numerical-plot-bounds-${boundType}`,
      );
      const makeBoundElement = (endpointIndex: 0 | 1) => {
        const e = createBoundInput(boundType, endpointIndex);
        e.addEventListener("change", () => {
          const existingBounds = this.bounds[boundType].value[propertyIndex];
          if (existingBounds === undefined) return;
          try {
            const value = parseDataTypeValue(property.dataType, e.value);
            this.setBound(
              boundType,
              endpointIndex,
              propertyIndex,
              value as number,
            );
            this.bounds[boundType].changed.dispatch();
          } catch {
            // Ignore invalid input.
          }
          updateInputBoundValue(
            e,
            this.bounds[boundType].value[propertyIndex][
              endpointIndex
            ] as number,
          );
        });
        return e;
      };
      const inputs: [HTMLInputElement, HTMLInputElement] = [
        makeBoundElement(0),
        makeBoundElement(1),
      ];

      let spacers: [HTMLElement, HTMLElement, HTMLElement] | undefined;
      if (boundType === "range") {
        spacers = [
          document.createElement("div"),
          document.createElement("div"),
          document.createElement("div"),
        ];
        spacers[1].classList.add(
          "neuroglancer-segment-query-result-numerical-plot-bound-constraint-spacer",
        );
        spacers[1].appendChild(columnCheckbox);
        const label = document.createElement("span");
        label.classList.add(
          "neuroglancer-segment-query-result-numerical-plot-label",
        );
        label.appendChild(document.createTextNode(property.id));
        label.appendChild(sortIcon);
        label.addEventListener("click", () => {
          toggleSortOrder(this.queryResult.value, this.setQuery, property.id);
        });
        spacers[1].appendChild(label);
        const { description } = property;
        if (description) {
          spacers[1].title = description;
        }
        container.appendChild(spacers[0]);
        container.appendChild(inputs[0]);
        const lessEqual1 = document.createElement("div");
        lessEqual1.textContent = "≤";
        lessEqual1.classList.add(
          "neuroglancer-segment-query-result-numerical-plot-bound-constraint-symbol",
        );
        container.appendChild(lessEqual1);
        container.appendChild(spacers[1]);
        const lessEqual2 = document.createElement("div");
        lessEqual2.textContent = "≤";
        lessEqual2.classList.add(
          "neuroglancer-segment-query-result-numerical-plot-bound-constraint-symbol",
        );
        container.appendChild(lessEqual2);
        container.appendChild(inputs[1]);
        container.appendChild(spacers[2]);
      } else {
        container.appendChild(inputs[0]);
        container.appendChild(inputs[1]);
      }
      return { container, spacers, inputs };
    };
    const boundElements = {
      range: makeBoundElements("range"),
      window: makeBoundElements("window"),
    };
    plotContainer.appendChild(boundElements.range.container);
    plotContainer.appendChild(plotImg);
    plotContainer.appendChild(boundElements.window.container);
    return {
      property,
      controller,
      element: plotContainer,
      plotImg,
      boundElements,
      bounds: {
        window: [NaN, NaN],
        range: [NaN, NaN],
      },
      propertyHistogram: undefined,
      columnCheckbox,
      sortIcon,
    };
  }

  private updateNumericalPropertySummary(
    propertyIndex: number,
    summary: NumericalPropertySummary,
    propertyHistogram: PropertyHistogram,
  ) {
    const prevWindowBounds = summary.bounds.window;
    const windowBounds = this.bounds.window.value[propertyIndex]!;
    const prevConstraintBounds = summary.bounds.range;
    const constraintBounds = this.bounds.range.value[propertyIndex]!;
    const { property } = summary;
    const queryResult = this.queryResult.value;
    const isIncluded = queryIncludesColumn(queryResult?.query, property.id);
    summary.columnCheckbox.checked = isIncluded;
    summary.columnCheckbox.title = isIncluded
      ? "Remove column from result table"
      : "Add column to result table";
    updateColumnSortIcon(queryResult, summary.sortIcon, property.id);
    // Check if we need to update the image.
    if (
      summary.propertyHistogram === propertyHistogram &&
      dataTypeIntervalEqual(prevWindowBounds, windowBounds) &&
      dataTypeIntervalEqual(prevConstraintBounds, constraintBounds)
    ) {
      return;
    }
    const { histogram } = propertyHistogram;
    const svgNs = "http://www.w3.org/2000/svg";
    const plotElement = document.createElementNS(svgNs, "svg");
    plotElement.setAttribute("width", "1");
    plotElement.setAttribute("height", "1");
    plotElement.setAttribute("preserveAspectRatio", "none");
    const rect = document.createElementNS(svgNs, "rect");
    const constraintStartX = computeInvlerp(windowBounds, constraintBounds[0]);
    const constraintEndX = computeInvlerp(windowBounds, constraintBounds[1]);
    rect.setAttribute("x", `${constraintStartX}`);
    rect.setAttribute("y", "0");
    rect.setAttribute("width", `${constraintEndX - constraintStartX}`);
    rect.setAttribute("height", "1");
    rect.setAttribute("fill", "#4f4f4f");
    plotElement.appendChild(rect);
    const numBins = histogram.length;
    const makeCdfLine = (
      startBinIndex: number,
      endBinIndex: number,
      endBinIndexForTotal: number,
    ) => {
      const polyLine = document.createElementNS(svgNs, "polyline");
      let points = "";
      let totalCount = 0;
      for (let i = startBinIndex; i < endBinIndexForTotal; ++i) {
        totalCount += histogram[i];
      }
      if (totalCount === 0) return undefined;
      const startBinX = computeInvlerp(
        windowBounds,
        propertyHistogram.window[0],
      );
      const endBinX = computeInvlerp(windowBounds, propertyHistogram.window[1]);
      const addPoint = (i: number, height: number) => {
        const fraction = i / (numBins - 2);
        const x = startBinX * (1 - fraction) + endBinX * fraction;
        points += ` ${x},${1 - height}`;
      };
      if (startBinIndex !== 0) {
        addPoint(startBinIndex, 0);
      }
      let cumSum = 0;
      for (let i = startBinIndex; i < endBinIndex; ++i) {
        const count = histogram[i];
        cumSum += count;
        addPoint(i, cumSum / totalCount);
      }
      polyLine.setAttribute("fill", "none");
      polyLine.setAttribute("stroke-width", "1px");
      polyLine.setAttribute("points", points);
      polyLine.setAttribute("vector-effect", "non-scaling-stroke");
      return polyLine;
    };

    {
      const polyLine = makeCdfLine(0, numBins - 1, numBins);
      if (polyLine !== undefined) {
        polyLine.setAttribute("stroke", "cyan");
        plotElement.appendChild(polyLine);
      }
    }

    if (!dataTypeIntervalEqual(property.bounds, constraintBounds)) {
      // Also plot CDF restricted to data that satisfies the constraint.
      const constraintStartBin = Math.floor(
        Math.max(
          0,
          Math.min(
            1,
            computeInvlerp(propertyHistogram.window, constraintBounds[0]),
          ),
        ) *
          (numBins - 2),
      );
      const constraintEndBin = Math.ceil(
        Math.max(
          0,
          Math.min(
            1,
            computeInvlerp(propertyHistogram.window, constraintBounds[1]),
          ),
        ) *
          (numBins - 2),
      );
      const polyLine = makeCdfLine(
        constraintStartBin,
        constraintEndBin,
        constraintEndBin,
      );
      if (polyLine !== undefined) {
        polyLine.setAttribute("stroke", "white");
        plotElement.appendChild(polyLine);
      }
    }

    // Embed the svg as an img rather than embedding it directly, in order to
    // allow it to be scaled using CSS.
    const xml = new XMLSerializer().serializeToString(plotElement);
    summary.plotImg.src = `data:image/svg+xml;base64,${btoa(xml)}`;
    summary.propertyHistogram = propertyHistogram;
    for (let endpointIndex = 0; endpointIndex < 2; ++endpointIndex) {
      prevWindowBounds[endpointIndex] = windowBounds[endpointIndex];
      updateInputBoundValue(
        summary.boundElements.window.inputs[endpointIndex],
        windowBounds[endpointIndex] as number,
      );
      prevConstraintBounds[endpointIndex] = constraintBounds[endpointIndex];
      updateInputBoundValue(
        summary.boundElements.range.inputs[endpointIndex],
        constraintBounds[endpointIndex] as number,
      );
    }

    const spacers = summary.boundElements.range.spacers!;
    const clampedRange = getClampedInterval(windowBounds, constraintBounds);
    const effectiveFraction = getIntervalBoundsEffectiveFraction(
      property.dataType,
      windowBounds,
    );
    const leftOffset =
      computeInvlerp(windowBounds, clampedRange[0]) * effectiveFraction;
    const rightOffset =
      computeInvlerp(windowBounds, clampedRange[1]) * effectiveFraction +
      (1 - effectiveFraction);
    spacers[0].style.width = `${leftOffset * 100}%`;
    spacers[2].style.width = `${(1 - rightOffset) * 100}%`;
  }
}

function renderTagSummary(
  queryResult: QueryResult,
  setQuery: (query: FilterQuery) => void,
): HTMLElement | undefined {
  const { tags } = queryResult;
  if (tags === undefined || tags.length === 0) return undefined;
  const filterQuery = queryResult.query as FilterQuery;
  const tagList = document.createElement("div");
  tagList.classList.add("neuroglancer-segment-query-result-tag-list");
  for (const { tag, count, desc } of tags) {
    const tagElement = document.createElement("div");
    tagElement.classList.add("neuroglancer-segment-query-result-tag");
    const tagName = document.createElement("span");
    tagName.classList.add("neuroglancer-segment-query-result-tag-name");
    // if the tag is different than desc, show both
    if (tag !== desc && desc !== undefined && desc !== "") {
      tagName.textContent = tag + " (" + desc + ")";
    } else {
      tagName.textContent = tag;
    }

    tagList.appendChild(tagElement);
    const included = filterQuery.includeTags.includes(tag);
    const excluded = filterQuery.excludeTags.includes(tag);
    let toggleTooltip: string;
    if (included) {
      toggleTooltip = "Remove tag from required set";
    } else if (excluded) {
      toggleTooltip = "Remove tag from excluded set";
    } else {
      toggleTooltip = "Add tag to required set";
    }
    tagName.addEventListener("click", () => {
      setQuery(
        changeTagConstraintInSegmentQuery(
          filterQuery,
          tag,
          true,
          !included && !excluded,
        ),
      );
    });
    tagName.title = toggleTooltip;
    const inQuery = included || excluded;
    const addIncludeExcludeButton = (include: boolean) => {
      const includeExcludeCount = include ? count : queryResult.count - count;
      const includeElement = document.createElement("div");
      includeElement.classList.add(
        "neuroglancer-segment-query-result-tag-toggle",
      );
      includeElement.classList.add(
        `neuroglancer-segment-query-result-tag-${
          include ? "include" : "exclude"
        }`,
      );
      tagElement.appendChild(includeElement);
      if (!inQuery && includeExcludeCount === 0) return;
      const selected = include ? included : excluded;
      includeElement.appendChild(
        new CheckboxIcon(
          {
            get value() {
              return selected;
            },
            set value(value: boolean) {
              setQuery(
                changeTagConstraintInSegmentQuery(
                  filterQuery,
                  tag,
                  include,
                  value,
                ),
              );
            },
            changed: neverSignal,
          },
          {
            text: include ? "+" : "-",
            enableTitle: `Add tag to ${include ? "required" : "exclusion"} set`,
            disableTitle: `Remove tag from ${
              include ? "required" : "exclusion"
            } set`,
            backgroundScheme: "dark",
          },
        ).element,
      );
    };
    addIncludeExcludeButton(true);
    addIncludeExcludeButton(false);
    tagElement.appendChild(tagName);
    const numElement = document.createElement("span");
    numElement.classList.add("neuroglancer-segment-query-result-tag-count");
    if (!inQuery) {
      numElement.textContent = count.toString();
    }
    tagElement.appendChild(numElement);
  }
  return tagList;
}

abstract class SegmentListGroupBase extends RefCounted {
  element = document.createElement("div");

  selectionStatusContainer = document.createElement("span");
  starAllButton: HTMLElement;
  selectionStatusMessage = document.createElement("span");
  copyAllSegmentsButton: HTMLElement;
  copyVisibleSegmentsButton: HTMLElement;
  visibilityToggleAllButton: HTMLElement;

  statusChanged = new NullarySignal();

  private debouncedUpdateStatus = debounce(() => this.updateStatus(), 0);

  makeSegmentsVisible(visible: boolean) {
    const { visibleSegments } = this.group;
    const segments = Array.from(this.listSegments());
    visibleSegments.set(segments, visible);
  }

  invertVisibility() {
    const markVisible: bigint[] = [];
    const markNonVisible: bigint[] = [];
    const { visibleSegments } = this.group;
    for (const segment of this.listSegments()) {
      if (visibleSegments.has(segment)) {
        markNonVisible.push(segment);
      } else {
        markVisible.push(segment);
      }
    }
    visibleSegments.set(markVisible, true);
    visibleSegments.set(markNonVisible, false);
  }

  selectSegments(select: boolean, changeVisibility = false) {
    const { selectedSegments, visibleSegments } = this.group;
    const segments = Array.from(this.listSegments());
    if (select || !changeVisibility) {
      selectedSegments.set(segments, select);
    }
    if (changeVisibility) {
      visibleSegments.set(segments, select);
    }
  }

  copySegments(onlyVisible = false) {
    let ids = [...this.listSegments()];
    if (onlyVisible) {
      ids = ids.filter((segment) => this.group.visibleSegments.has(segment));
    }
    ids.sort(bigintCompare);
    setClipboard(ids.join(", "));
  }

  constructor(
    protected listSource: SegmentListSource,
    protected group: SegmentationUserLayerGroupState,
  ) {
    super();

    const { element } = this;
    element.style.display = "contents";
    this.starAllButton = makeStarButton({
      title:
        "Click to toggle star status, shift+click to unstar non-visible segments.",
      onClick: (event) => {
        const starred = this.starAllButton.classList.contains(
          "neuroglancer-starred",
        );
        if (event.shiftKey) {
          const nonVisibleSegments: bigint[] = [];
          for (const segment of this.group.selectedSegments) {
            if (!this.group.visibleSegments.has(segment)) {
              nonVisibleSegments.push(segment);
            }
          }
          this.group.selectedSegments.delete(nonVisibleSegments);
          return;
        }
        this.selectSegments(!starred, false);
      },
    });
    this.copyAllSegmentsButton = makeCopyButton({
      title: "Copy all segment IDs",
      onClick: () => {
        this.copySegments(false);
      },
    });
    this.copyVisibleSegmentsButton = makeCopyButton({
      title: "Copy visible segment IDs",
      onClick: () => {
        this.copySegments(true);
      },
    });
    this.visibilityToggleAllButton = makeEyeButton({
      onClick: (event) => {
        if (event.shiftKey) {
          this.invertVisibility();
          return;
        }
        this.makeSegmentsVisible(
          !this.visibilityToggleAllButton.classList.contains(
            "neuroglancer-visible",
          ),
        );
      },
    });
    const { selectionStatusContainer } = this;
    this.selectionStatusMessage.classList.add(
      "neuroglancer-segment-list-status-message",
    );
    selectionStatusContainer.classList.add("neuroglancer-segment-list-status");
    selectionStatusContainer.appendChild(this.copyAllSegmentsButton);
    selectionStatusContainer.appendChild(this.starAllButton);
    selectionStatusContainer.appendChild(this.visibilityToggleAllButton);
    selectionStatusContainer.appendChild(this.copyVisibleSegmentsButton);
    selectionStatusContainer.appendChild(this.selectionStatusMessage);
    this.element.appendChild(selectionStatusContainer);
    this.registerDisposer(
      group.visibleSegments.changed.add(this.debouncedUpdateStatus),
    );
    this.registerDisposer(
      group.selectedSegments.changed.add(this.debouncedUpdateStatus),
    );
    this.registerDisposer(listSource.changed.add(this.debouncedUpdateStatus));
  }

  *listSegments(): IterableIterator<bigint> {}

  updateStatus() {
    const {
      listSource,
      group,
      starAllButton,
      selectionStatusMessage,
      copyAllSegmentsButton,
      copyVisibleSegmentsButton,
      visibilityToggleAllButton,
    } = this;
    listSource.debouncedUpdate.flush();
    const { visibleSegments, selectedSegments } = group;
    let queryVisibleCount = 0;
    let querySelectedCount = 0;
    let numMatches = 0;
    let statusMessage = "";
    const selectedCount = selectedSegments.size;
    const visibleCount = visibleSegments.size;
    if (listSource instanceof SegmentQueryListSource) {
      numMatches = listSource.numMatches;
      queryVisibleCount = listSource.visibleMatches;
      querySelectedCount = listSource.selectedMatches;
      statusMessage = listSource.statusText.value;
    } else {
      statusMessage = `${visibleCount}/${selectedCount} visible`;
    }
    const visibleDisplayedCount = numMatches ? queryVisibleCount : visibleCount;
    const visibleSelectedCount = numMatches
      ? querySelectedCount
      : selectedCount;
    const totalDisplayed = numMatches || selectedCount;
    starAllButton.classList.toggle(
      "neuroglancer-starred",
      visibleSelectedCount === totalDisplayed,
    );
    starAllButton.classList.toggle(
      "neuroglancer-indeterminate",
      visibleSelectedCount > 0 && visibleSelectedCount !== totalDisplayed,
    );
    selectionStatusMessage.textContent = statusMessage;
    copyAllSegmentsButton.title = `Copy all ${totalDisplayed} ${
      numMatches ? "matching" : "starred"
    } segment(s)`;
    copyVisibleSegmentsButton.title = `Copy ${visibleDisplayedCount} ${
      numMatches ? "visible matching" : "visible"
    } segment(s)`;
    copyAllSegmentsButton.style.visibility = totalDisplayed
      ? "visible"
      : "hidden";
    copyVisibleSegmentsButton.style.visibility = visibleDisplayedCount
      ? "visible"
      : "hidden";
    starAllButton.style.visibility = totalDisplayed ? "visible" : "hidden";
    visibilityToggleAllButton.style.visibility = totalDisplayed
      ? "visible"
      : "hidden";
    const allVisible = visibleDisplayedCount === totalDisplayed;
    visibilityToggleAllButton.classList.toggle(
      "neuroglancer-visible",
      allVisible,
    );
    const visibleIndeterminate =
      visibleDisplayedCount > 0 && visibleDisplayedCount !== totalDisplayed;
    visibilityToggleAllButton.classList.toggle(
      "neuroglancer-indeterminate",
      visibleIndeterminate,
    );
    let visibleToggleTitle: string;
    if (!allVisible) {
      visibleToggleTitle = `Click to show ${
        totalDisplayed - visibleDisplayedCount
      } segment ID(s).`;
    } else {
      visibleToggleTitle = `Click to hide ${totalDisplayed} segment ID(s).`;
    }
    if (visibleIndeterminate) {
      visibleToggleTitle += "  Shift+click to invert visibility.";
    }
    visibilityToggleAllButton.title = visibleToggleTitle;
    this.statusChanged.dispatch();
  }
}

class SegmentListGroupSelected extends SegmentListGroupBase {
  constructor(
    protected listSource: SegmentListSource,
    protected group: SegmentationUserLayerGroupState,
  ) {
    super(listSource, group);
  }

  listSegments() {
    return this.group.selectedSegments[Symbol.iterator](); // TODO, better way to call the iterator?
  }
}

class SegmentListGroupQuery extends SegmentListGroupBase {
  updateQuery() {
    const { listSource, debouncedUpdateQueryModel } = this;
    debouncedUpdateQueryModel();
    debouncedUpdateQueryModel.flush();
    listSource.debouncedUpdate.flush();
  }

  listSegments(): IterableIterator<bigint> {
    const { listSource, segmentPropertyMap } = this;
    this.updateQuery();
    const queryResult = listSource.queryResult.value;
    return forEachQueryResultSegmentIdGenerator(
      segmentPropertyMap,
      queryResult,
    );
  }

  constructor(
    list: VirtualList,
    protected listSource: SegmentQueryListSource,
    group: SegmentationUserLayerGroupState,
    private segmentPropertyMap: PreprocessedSegmentPropertyMap | undefined,
    segmentQuery: WatchableValueInterface<string>,
    queryElement: HTMLInputElement,
    private debouncedUpdateQueryModel: DebouncedFunc<() => void>,
  ) {
    super(listSource, group);
    const setQuery = (newQuery: ExplicitIdQuery | FilterQuery) => {
      queryElement.focus();
      queryElement.select();
      const value = unparseSegmentQuery(segmentPropertyMap, newQuery);
      document.execCommand("insertText", false, value);
      segmentQuery.value = value;
      queryElement.select();
    };
    const queryStatisticsContainer = document.createElement("div");
    queryStatisticsContainer.classList.add(
      "neuroglancer-segment-query-result-statistics",
    );
    const queryStatisticsSeparator = document.createElement("div");
    queryStatisticsSeparator.classList.add(
      "neuroglancer-segment-query-result-statistics-separator",
    );
    const queryErrors = document.createElement("ul");
    queryErrors.classList.add("neuroglancer-segment-query-errors");
    // push them in front of the base list elements
    this.element.prepend(
      queryErrors,
      queryStatisticsContainer,
      queryStatisticsSeparator,
    );
    this.registerEventListener(queryElement, "input", () => {
      debouncedUpdateQueryModel();
    });
    this.registerDisposer(
      registerActionListener(queryElement, "cancel", () => {
        queryElement.focus();
        queryElement.select();
        document.execCommand("delete");
        queryElement.blur();
        queryElement.value = "";
        segmentQuery.value = "";
      }),
    );
    this.registerDisposer(
      registerActionListener(queryElement, "toggle-listed", () => {
        this.toggleMatches();
      }),
    );
    this.registerDisposer(
      registerActionListener(queryElement, "hide-all", () => {
        group.visibleSegments.clear();
      }),
    );
    this.registerDisposer(
      registerActionListener(queryElement, "hide-listed", () => {
        debouncedUpdateQueryModel();
        debouncedUpdateQueryModel.flush();
        listSource.debouncedUpdate.flush();
        const { visibleSegments } = group;
        if (this.listSource instanceof StarredSegmentsListSource) {
          visibleSegments.clear();
        } else {
          visibleSegments.delete(Array.from(this.listSegments()));
        }
      }),
    );
    const numericalPropertySummaries = this.registerDisposer(
      new NumericalPropertiesSummary(
        segmentPropertyMap,
        listSource.queryResult,
        setQuery,
      ),
    );
    {
      const { listElement } = numericalPropertySummaries;
      if (listElement !== undefined) {
        queryStatisticsContainer.appendChild(listElement);
      }
    }
    const updateQueryErrors = (queryResult: QueryResult | undefined) => {
      const errors = queryResult?.errors;
      removeChildren(queryErrors);
      if (errors === undefined) return;
      for (const error of errors) {
        const errorElement = document.createElement("li");
        errorElement.textContent = error.message;
        queryErrors.appendChild(errorElement);
      }
    };

    let tagSummary: HTMLElement | undefined = undefined;
    observeWatchable((queryResult: QueryResult | undefined) => {
      listSource.segmentWidgetFactory =
        new SegmentWidgetWithExtraColumnsFactory(
          listSource.segmentationDisplayState,
          listSource.parentElement,
          (property) => queryIncludesColumn(queryResult?.query, property.id),
        );
      list.scrollToTop();
      removeChildren(list.header);
      if (segmentPropertyMap !== undefined) {
        const header = listSource.segmentWidgetFactory.getHeader();
        header.container.classList.add("neuroglancer-segment-list-header");
        for (const headerLabel of header.propertyLabels) {
          const { label, sortIcon, id } = headerLabel;
          label.addEventListener("click", () => {
            toggleSortOrder(listSource.queryResult.value, setQuery, id);
          });
          updateColumnSortIcon(queryResult, sortIcon, id);
        }
        list.header.appendChild(header.container);
      }
      updateQueryErrors(queryResult);
      queryStatisticsSeparator.style.display = "none";
      tagSummary?.remove();
      if (queryResult === undefined) return;
      const { query } = queryResult;
      if (query.errors !== undefined || query.ids !== undefined) return;
      tagSummary = renderTagSummary(queryResult, setQuery);
      if (tagSummary !== undefined) {
        queryStatisticsContainer.appendChild(tagSummary);
      }
      if (
        numericalPropertySummaries.properties.length > 0 ||
        tagSummary !== undefined
      ) {
        queryStatisticsSeparator.style.display = "";
      }
    }, listSource.queryResult);
  }

  toggleMatches() {
    const { listSource } = this;
    this.updateQuery();
    listSource.debouncedUpdate.flush();
    const queryResult = listSource.queryResult.value;
    if (queryResult === undefined) return;
    const { visibleMatches } = listSource;
    const shouldSelect = visibleMatches !== queryResult.count;
    this.selectSegments(shouldSelect, true);
    return true;
  }
}

export class SegmentDisplayTab extends Tab {
  constructor(public layer: SegmentationUserLayer) {
    super();
    const { element } = this;
    element.classList.add("neuroglancer-segment-display-tab");
    element.appendChild(
      this.registerDisposer(
        new DependentViewWidget(
          layer.displayState.segmentationGroupState.value.graph,
          (graph, parent, context) => {
            if (graph === undefined) return;
            if (graph.tabContents) {
              return;
            }
            const toolbox = document.createElement("div");
            toolbox.className = "neuroglancer-segmentation-toolbox";
            toolbox.appendChild(
              makeToolButton(context, layer.toolBinder, {
                toolJson: ANNOTATE_MERGE_SEGMENTS_TOOL_ID,
                label: "Merge",
                title: "Merge segments",
              }),
            );
            toolbox.appendChild(
              makeToolButton(context, layer.toolBinder, {
                toolJson: ANNOTATE_SPLIT_SEGMENTS_TOOL_ID,
                label: "Split",
                title: "Split segments",
              }),
            );
            parent.appendChild(toolbox);
          },
        ),
      ).element,
    );

    const toolbox = document.createElement("div");
    toolbox.className = "neuroglancer-segmentation-toolbox";
    toolbox.appendChild(
      makeToolButton(this, layer.toolBinder, {
        toolJson: SELECT_SEGMENTS_TOOLS_ID,
        label: "Select",
        title: "Select/Deselect segments",
      }),
    );
    element.appendChild(toolbox);

    const queryElement = document.createElement("input");
    queryElement.classList.add("neuroglancer-segment-list-query");
    queryElement.addEventListener("focus", () => {
      queryElement.select();
    });
    const keyboardHandler = this.registerDisposer(
      new KeyboardEventBinder(queryElement, keyMap),
    );
    keyboardHandler.allShortcutsAreGlobal = true;
    const { segmentQuery } = this.layer.displayState;
    const debouncedUpdateQueryModel = this.registerCancellable(
      debounce(() => {
        segmentQuery.value = queryElement.value;
      }, 200),
    );
    queryElement.autocomplete = "off";
    queryElement.title = keyMap.describe();
    queryElement.spellcheck = false;
    queryElement.placeholder = "Enter ID, name prefix or /regexp";
    this.registerDisposer(
      observeWatchable((q) => {
        queryElement.value = q;
      }, segmentQuery),
    );
    this.registerDisposer(
      observeWatchable((t) => {
        if (Date.now() - t < 100) {
          setTimeout(() => {
            queryElement.focus();
          }, 0);
          this.layer.segmentQueryFocusTime.value = Number.NEGATIVE_INFINITY;
        }
      }, this.layer.segmentQueryFocusTime),
    );

    element.appendChild(queryElement);
    element.appendChild(
      this.registerDisposer(
        new DependentViewWidget(
          // segmentLabelMap is guaranteed to change if segmentationGroupState changes.
          layer.displayState.segmentPropertyMap,
          (segmentPropertyMap, parent, context) => {
            const listSource = context.registerDisposer(
              new SegmentQueryListSource(
                segmentQuery,
                segmentPropertyMap,
                layer.displayState,
                parent,
              ),
            );
            const selectedSegmentsListSource = context.registerDisposer(
              new StarredSegmentsListSource(layer.displayState, parent),
            );
            const list = context.registerDisposer(
              new VirtualList({ source: listSource, horizontalScroll: true }),
            );
            const selectedSegmentsList = context.registerDisposer(
              new VirtualList({
                source: selectedSegmentsListSource,
                horizontalScroll: true,
              }),
            );

            const group = layer.displayState.segmentationGroupState.value;

            const segList = context.registerDisposer(
              new SegmentListGroupQuery(
                list,
                listSource,
                group,
                segmentPropertyMap,
                segmentQuery,
                queryElement,
                debouncedUpdateQueryModel,
              ),
            );
            segList.element.appendChild(list.element);
            parent.appendChild(segList.element);
            const segList2 = context.registerDisposer(
              new SegmentListGroupSelected(selectedSegmentsListSource, group),
            );
            segList2.element.appendChild(selectedSegmentsList.element);
            parent.appendChild(segList2.element);

            const updateListDisplayState = () => {
              const showQueryResultsList =
                listSource.query.value !== "" || listSource.numMatches > 0;
              const showStarredSegmentsList =
                selectedSegmentsListSource.length > 0 || !showQueryResultsList;
              segList.element.style.display = showQueryResultsList
                ? "contents"
                : "none";
              segList2.element.style.display = showStarredSegmentsList
                ? "contents"
                : "none";
            };
            context.registerDisposer(
              segList.statusChanged.add(updateListDisplayState),
            );
            context.registerDisposer(
              segList2.statusChanged.add(updateListDisplayState),
            );
            segList.updateStatus();
            segList2.updateStatus();

            const updateListItems = context.registerCancellable(
              animationFrameDebounce(() => {
                listSource.updateRenderedItems(list);
                selectedSegmentsListSource.updateRenderedItems(
                  selectedSegmentsList,
                );
              }),
            );
            const { displayState } = this.layer;
            registerCallbackWhenSegmentationDisplayStateChanged(
              displayState,
              context,
              updateListItems,
            );
            context.registerDisposer(
              displayState.segmentationGroupState.value.selectedSegments.changed.add(
                updateListItems,
              ),
            );
            list.element.classList.add("neuroglancer-segment-list");
            list.element.classList.add("neuroglancer-preview-list");
            selectedSegmentsList.element.classList.add(
              "neuroglancer-segment-list",
            );
            context.registerDisposer(layer.bindSegmentListWidth(list.element));
            context.registerDisposer(
              layer.bindSegmentListWidth(selectedSegmentsList.element),
            );
            context.registerDisposer(
              new MouseEventBinder(list.element, getDefaultSelectBindings()),
            );
            context.registerDisposer(
              new MouseEventBinder(
                selectedSegmentsList.element,
                getDefaultSelectBindings(),
              ),
            );

            // list2 doesn't depend on queryResult, maybe move this into class
            selectedSegmentsListSource.segmentWidgetFactory =
              new SegmentWidgetWithExtraColumnsFactory(
                selectedSegmentsListSource.segmentationDisplayState,
                selectedSegmentsListSource.parentElement,
                (property) => queryIncludesColumn(undefined, property.id),
              );
            selectedSegmentsList.scrollToTop();
            removeChildren(selectedSegmentsList.header);
          },
        ),
      ).element,
    );
  }
}
