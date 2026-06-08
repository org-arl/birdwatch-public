import { BoundingBox, BoxType, MatchOverlapMetric, RenderBox, VisualizationConfig, ImageItem } from '../types';

/** Parse a YOLO label file (class cx cy w h [confidence]). */
export const parseYoloFile = async (file: File | FileSystemFileHandle): Promise<BoundingBox[]> => {
  let text = '';
  if (file instanceof File) {
    text = await file.text();
  } else {
    const f = await (file as FileSystemFileHandle).getFile();
    text = await f.text();
  }

  const lines = text.trim().split('\n');

  return lines
    .map((line): BoundingBox | null => {
      const parts = line.trim().split(/\s+/);
      if (parts.length < 5) return null;
      return {
        classId: parseInt(parts[0], 10),
        x: parseFloat(parts[1]),
        y: parseFloat(parts[2]),
        w: parseFloat(parts[3]),
        h: parseFloat(parts[4]),
        confidence: parts[5] ? parseFloat(parts[5]) : 1.0,
      };
    })
    .filter((box): box is BoundingBox => box !== null);
};

/** Load all .txt files in a map into a label map. */
export const preloadLabels = async (fileMap: { [name: string]: File | FileSystemFileHandle }): Promise<{ [name: string]: BoundingBox[] }> => {
  const labelMap: { [name: string]: BoundingBox[] } = {};
  await Promise.all(Object.entries(fileMap).map(async ([name, file]) => {
    if (name.endsWith('.txt')) {
      labelMap[name] = await parseYoloFile(file);
    }
  }));
  return labelMap;
};

const intersectAndAreas = (
  pred: BoundingBox,
  gt: BoundingBox
): { intersectionArea: number; predArea: number; gtArea: number } => {
  const b1_x1 = pred.x - pred.w / 2;
  const b1_y1 = pred.y - pred.h / 2;
  const b1_x2 = pred.x + pred.w / 2;
  const b1_y2 = pred.y + pred.h / 2;

  const b2_x1 = gt.x - gt.w / 2;
  const b2_y1 = gt.y - gt.h / 2;
  const b2_x2 = gt.x + gt.w / 2;
  const b2_y2 = gt.y + gt.h / 2;

  const x1 = Math.max(b1_x1, b2_x1);
  const y1 = Math.max(b1_y1, b2_y1);
  const x2 = Math.min(b1_x2, b2_x2);
  const y2 = Math.min(b1_y2, b2_y2);

  const predArea = pred.w * pred.h;
  const gtArea = gt.w * gt.h;

  if (x2 < x1 || y2 < y1) {
    return { intersectionArea: 0, predArea, gtArea };
  }

  const intersectionArea = (x2 - x1) * (y2 - y1);
  return { intersectionArea, predArea, gtArea };
};

/** Standard IoU = intersection / union. */
const calculateIoU = (pred: BoundingBox, gt: BoundingBox): number => {
  const { intersectionArea, predArea, gtArea } = intersectAndAreas(pred, gt);
  const unionArea = predArea + gtArea - intersectionArea;
  if (unionArea <= 0) return 0;
  return intersectionArea / unionArea;
};

/**
 * Greedy per-class NMS: sort by confidence (high first), keep a box unless it has IoU > threshold
 * with any already-kept box of the same class.
 */
export const applyNmsIou = (boxes: BoundingBox[], iouThreshold: number): BoundingBox[] => {
  if (boxes.length === 0) return [];
  const byClass = new Map<number, BoundingBox[]>();
  for (const b of boxes) {
    const list = byClass.get(b.classId) ?? [];
    list.push(b);
    byClass.set(b.classId, list);
  }
  const out: BoundingBox[] = [];
  for (const classBoxes of byClass.values()) {
    const sorted = [...classBoxes].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
    const kept: BoundingBox[] = [];
    for (const cand of sorted) {
      let suppress = false;
      for (const k of kept) {
        if (calculateIoU(cand, k) > iouThreshold) {
          suppress = true;
          break;
        }
      }
      if (!suppress) kept.push(cand);
    }
    out.push(...kept);
  }
  return out;
};

/**
 * IoMin = intersection / min(Area(Pred), Area(GT)).
 * Penalizes misses less when one box is much larger than the other.
 */
const calculateIoMin = (pred: BoundingBox, gt: BoundingBox): number => {
  const { intersectionArea, predArea, gtArea } = intersectAndAreas(pred, gt);
  const minArea = Math.min(predArea, gtArea);
  if (minArea === 0) return 0;
  return intersectionArea / minArea;
};

const overlapScore = (metric: MatchOverlapMetric, pred: BoundingBox, gt: BoundingBox): number =>
  metric === 'iou' ? calculateIoU(pred, gt) : calculateIoMin(pred, gt);

/**
 * Assign TP/FP/FN labels for drawing.
 * Predictions pass conf threshold and NMS; any overlap with same-class GT counts as TP.
 */
export const calculateMatches = (
  gtBoxes: BoundingBox[],
  predBoxes: BoundingBox[],
  config: VisualizationConfig
): RenderBox[] => {
  const result: RenderBox[] = [];
  const matchedGtIndices = new Set<number>();
  const { matchOverlapMetric, matchOverlapThreshold } = config;

  const validPreds = predBoxes.filter(p => (p.confidence || 1) >= config.confThreshold);
  const afterNms = applyNmsIou(validPreds, config.nmsIouThreshold);
  const sortedPreds = [...afterNms].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));

  sortedPreds.forEach((pred) => {
    let isTp = false;

    gtBoxes.forEach((gt, gtIdx) => {
      if (gt.classId !== pred.classId) return;

      const score = overlapScore(matchOverlapMetric, pred, gt);

      if (score >= matchOverlapThreshold) {
        isTp = true;
        matchedGtIndices.add(gtIdx);
      }
    });

    if (isTp) {
      result.push({
        ...pred,
        type: BoxType.TP_PRED,
        color: config.styles.tpPred.color,
        dashed: config.styles.tpPred.dashed
      });
    } else {
      result.push({
        ...pred,
        type: BoxType.FP,
        color: config.styles.fp.color,
        dashed: config.styles.fp.dashed
      });
    }
  });

  gtBoxes.forEach((gt, idx) => {
    if (matchedGtIndices.has(idx)) {
      result.push({
        ...gt,
        type: BoxType.TP_GT,
        color: config.styles.tpGt.color,
        dashed: config.styles.tpGt.dashed
      });
    } else {
      result.push({
        ...gt,
        type: BoxType.FN,
        color: config.styles.fn.color,
        dashed: config.styles.fn.dashed
      });
    }
  });

  return result;
};


export interface PRPoint {
  confidence: number;
  precision: number;
  recall: number;
  f1: number;
}

type PRStatsConfigPick = Pick<VisualizationConfig, 'matchOverlapMetric' | 'matchOverlapThreshold' | 'nmsIouThreshold'>;

const primaryPredictionBoxes = (item: ImageItem): BoundingBox[] =>
  item.predictions?.filter(p => p.visible)[0]?.boxes ?? [];

/**
 * Build a precision–recall curve from the primary visible prediction source per image.
 * Duplicate predictions on the same GT are ignored; one prediction may match multiple GTs.
 * The curve is monotonized (non-increasing precision vs. recall).
 */
export const calculatePRStats = async (
  items: ImageItem[],
  prConfig: PRStatsConfigPick
): Promise<PRPoint[]> => {
  const { matchOverlapMetric, matchOverlapThreshold, nmsIouThreshold } = prConfig;
  const dataset = items.map((item, imgIdx) => {
    const gts = item.gtData || [];
    return {
      imgIdx,
      gts: gts.map(g => ({ ...g, used: false })),
      rawPreds: primaryPredictionBoxes(item),
    };
  });

  const totalGtCount = dataset.reduce((acc, d) => acc + d.gts.length, 0);

  if (totalGtCount === 0) return [];

  const steps = 50;
  const rawResults: PRPoint[] = [];

  for (let i = 0; i <= steps; i++) {
    const confThreshold = i / steps;

    const predsAtStep: (BoundingBox & { imgIdx: number })[] = [];
    for (const d of dataset) {
      const passed = d.rawPreds.filter(p => (p.confidence || 0) >= confThreshold);
      for (const p of applyNmsIou(passed, nmsIouThreshold)) {
        predsAtStep.push({ ...p, imgIdx: d.imgIdx });
      }
    }
    predsAtStep.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));

    const detectedGtsMap = new Map<number, Set<number>>();

    let tp = 0;
    let fp = 0;

    for (const pred of predsAtStep) {
      const imgIdx = pred.imgIdx;
      const gtsInImage = dataset[imgIdx].gts;

      if (!detectedGtsMap.has(imgIdx)) {
        detectedGtsMap.set(imgIdx, new Set());
      }
      const detectedSet = detectedGtsMap.get(imgIdx)!;

      let matchedAnyGt = false;
      let isNewDiscovery = false;

      gtsInImage.forEach((gt, gtIdx) => {
        if (gt.classId !== pred.classId) return;

        const score = overlapScore(matchOverlapMetric, pred, gt);

        if (score >= matchOverlapThreshold) {
          matchedAnyGt = true;
          if (!detectedSet.has(gtIdx)) {
            detectedSet.add(gtIdx);
            isNewDiscovery = true;
          }
        }
      });

      if (matchedAnyGt) {
        if (isNewDiscovery) {
          tp++;
        }
      } else {
        fp++;
      }
    }

    const precision = (tp + fp) === 0 ? 1 : tp / (tp + fp);
    const recall = tp / totalGtCount;
    const f1 = (precision + recall) === 0 ? 0 : 2 * (precision * recall) / (precision + recall);

    rawResults.push({
      confidence: confThreshold,
      precision,
      recall,
      f1
    });
  }

  rawResults.sort((a, b) => b.confidence - a.confidence);

  let maxPrecision = 0;
  for (let i = rawResults.length - 1; i >= 0; i--) {
    maxPrecision = Math.max(maxPrecision, rawResults[i].precision);
    rawResults[i].precision = maxPrecision;
  }

  if (rawResults[0].recall > 0) {
    rawResults.unshift({
      confidence: 1.1,
      precision: 1.0,
      recall: 0.0,
      f1: 0.0
    });
  }

  return rawResults;
};
