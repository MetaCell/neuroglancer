import { AnnotationType } from "#src/annotation/index.js";

export function addAnnotationClassToIcon(
  icon: HTMLElement,
  annotationType: AnnotationType,
) {
  icon.classList.remove(
    "annotation-point",
    "annotation-line",
    "annotation-box",
    "annotation-ellipsoid",
  );
  switch (annotationType) {
    case AnnotationType.POINT:
      icon.classList.add("annotation-point");
      break;
    case AnnotationType.LINE:
      icon.classList.add("annotation-line");
      break;
    case AnnotationType.AXIS_ALIGNED_BOUNDING_BOX:
      icon.classList.add("annotation-box");
      break;
    case AnnotationType.ELLIPSOID:
      icon.classList.add("annotation-ellipsoid");
      break;
    /* TODO add to neuroglass theme handling for annotation-polyline and change this */
    case AnnotationType.POLYLINE:
      icon.classList.add("annotation-line");
      break;
    default:
      icon.classList.add("annotation-point");
  }
}
