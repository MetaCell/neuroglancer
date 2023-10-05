import {TrackableEnum} from 'neuroglancer/util/trackable_enum';
import {glsl_CHUNK_NUMBER_SHADER, glsl_MAX_PROJECTION_SHADER, glsl_USER_DEFINED_RAY_TRAVERSAL, glsl_FRONT_TO_BACK_COMPOSITING_SHADER} from 'src/neuroglancer/volume_rendering/glsl';

export enum SHADER_MODES {
  DEFAULT = 0,
  FRONT_TO_BACK_COMPOSITING = 1,
  MAX_PROJECTION = 2,
  CHUNK_VISUALIZATION = 3,
}

export const SHADER_FUNCTIONS = new Map([
  [SHADER_MODES.DEFAULT, glsl_USER_DEFINED_RAY_TRAVERSAL],
  [SHADER_MODES.FRONT_TO_BACK_COMPOSITING, glsl_FRONT_TO_BACK_COMPOSITING_SHADER],
  [SHADER_MODES.MAX_PROJECTION, glsl_MAX_PROJECTION_SHADER],
  [SHADER_MODES.CHUNK_VISUALIZATION, glsl_CHUNK_NUMBER_SHADER],
]);

export type TrackableShaderModeValue = TrackableEnum<SHADER_MODES>;

export function trackableShaderModeValue(initialValue = SHADER_MODES.DEFAULT) {
  return new TrackableEnum(SHADER_MODES, initialValue);
}