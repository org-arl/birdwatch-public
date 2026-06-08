/** YOLO box in normalized image coordinates (center x/y, width/height, all in [0, 1]). */
export interface BoundingBox {
  classId: number;
  x: number;
  y: number;
  w: number;
  h: number;
  confidence?: number;
}

export enum BoxType {
  TP_PRED = 'TP_PRED',
  TP_GT = 'TP_GT',
  FP = 'FP',
  FN = 'FN',
}

export interface RenderBox extends BoundingBox {
  type: BoxType;
  color: string;
  dashed: boolean;
}

export interface BoxStyle {
  color: string;
  dashed: boolean;
}

export type AspectRatio = '16:9' | '4:3' | '1:1' | 'auto';

/** How prediction–GT box overlap is scored for matching (TP/FP and PR curves). */
export type MatchOverlapMetric = 'iou' | 'iomin';

export interface VisualizationConfig {
  matchOverlapMetric: MatchOverlapMetric;
  /** Minimum overlap score vs. a same-class GT to count as a match (IoU or IoMin, depending on metric). */
  matchOverlapThreshold: number;
  /**
   * Per-class greedy NMS on predictions: boxes with pairwise IoU above this value are suppressed (lower confidence dropped).
   * Applied after the confidence threshold; independent of match overlap metric/threshold vs. ground truth.
   */
  nmsIouThreshold: number;
  confThreshold: number;
  styles: {
    tpPred: BoxStyle;
    tpGt: BoxStyle;
    fp: BoxStyle;
    fn: BoxStyle;
  };
  lineWidth: number;
  labelFontSize: number;
  gridSize: 9 | 16;
  aspectRatio: AspectRatio;
  zoomLevel: number;
  editHighlightColor?: string;
  audio?: {
    minFreq?: number;
    maxFreq?: number;
    highlightColor?: string;
    clipSec?: number;
    playbackSpeed?: number;
  };
  showLabels?: boolean;
  showPredictions?: boolean;
}

export interface FileMap {
  [filename: string]: File | FileSystemFileHandle;
}

export interface LabelMap {
  [filename: string]: BoundingBox[];
}

export interface ImageItem {
  name: string;
  file: File | FileSystemFileHandle;
  gtData?: BoundingBox[];
  predictions?: { sourceId: string, boxes: BoundingBox[], color: string, visible: boolean }[];
  isModified?: boolean;
  isSaved?: boolean;
}

export interface FileCollection {
  id: string;
  name: string;
  type: 'images' | 'labels';
  files: FileMap;
  labels?: LabelMap;
  count: number;
}

export interface PredictionSource {
  id: string;
  name: string;
  path: string;
  color: string;
  visible: boolean;
  labels: LabelMap;
  groupId?: string;
}

export interface Project {
  id: string;
  name: string;
  config: VisualizationConfig;
  imageCollectionId: string | null;
  gtCollectionId: string | null;
  predictionSources: PredictionSource[];
  imagePath?: string;
  gtPath?: string;
  audioPath?: string;
}

export interface RenderResult {
  stats: {
    tp: number;
    fp: number;
    fn: number;
  };
  boxes: RenderBox[];
}