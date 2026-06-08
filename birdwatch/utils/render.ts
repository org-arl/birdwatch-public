import { ImageItem, VisualizationConfig, BoxType, RenderResult, RenderBox } from '../types';
import { calculateMatches } from './yolo';

export interface RenderOptions {
  fontSize?: number;
  forceLineWidth?: number;
  highlightType?: BoxType | null;
  preCalculatedBoxes?: RenderBox[];
}

/** Draw one image with letterboxed aspect fit; stats are returned for React overlays. */
export const drawVisualization = async (
  ctx: CanvasRenderingContext2D,
  item: ImageItem,
  config: VisualizationConfig,
  targetWidth: number,
  targetHeight: number,
  img?: HTMLImageElement,
  options?: RenderOptions
): Promise<RenderResult> => {
  let imageElement = img;
  if (!imageElement) {
    imageElement = new Image();
    const file = item.file;
    const url = file instanceof File ? URL.createObjectURL(file) : URL.createObjectURL(await (file as FileSystemFileHandle).getFile());
    imageElement.src = url;
    await new Promise((resolve) => {
      imageElement!.onload = resolve;
    });
  }

  const imgW = imageElement.naturalWidth;
  const imgH = imageElement.naturalHeight;

  const targetRatio = targetWidth / targetHeight;
  const imgRatio = imgW / imgH;
  const isExactMatch = Math.abs(targetRatio - imgRatio) < 0.005;

  let drawW, drawH, offsetX, offsetY;

  if (isExactMatch) {
    drawW = targetWidth;
    drawH = targetHeight;
    offsetX = 0;
    offsetY = 0;
  } else {
    const scale = Math.min(targetWidth / imgW, targetHeight / imgH);
    drawW = imgW * scale;
    drawH = imgH * scale;
    offsetX = (targetWidth - drawW) / 2;
    offsetY = (targetHeight - drawH) / 2;

    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, targetWidth, targetHeight);
  }

  ctx.globalAlpha = 1.0;
  ctx.drawImage(imageElement, offsetX, offsetY, drawW, drawH);

  let renderBoxes: RenderBox[] = [];
  if (options?.preCalculatedBoxes) {
    renderBoxes = options.preCalculatedBoxes;
  } else {
    const gtBoxes = item.gtData || [];
    renderBoxes = calculateMatches(gtBoxes, [], config);
  }

  const lineWidth = options?.forceLineWidth ?? config.lineWidth;
  const baseFontSize = options?.fontSize ?? config.labelFontSize ?? 14;
  const highlightType = options?.highlightType;

  ctx.lineWidth = lineWidth;
  ctx.font = `bold ${baseFontSize}px sans-serif`;

  const scaleX = drawW / imgW;
  const scaleY = drawH / imgH;

  renderBoxes.forEach((box) => {
    let alpha = 1.0;

    if (config.showPredictions === false && (box.type === BoxType.TP_PRED || box.type === BoxType.FP)) {
      return;
    }

    if (highlightType) {
      if (box.type === highlightType) {
        alpha = 1.0;
      } else if (highlightType === BoxType.TP_PRED && box.type === BoxType.TP_GT) {
        alpha = 1.0;
      } else {
        alpha = 0.0;
      }
    }

    ctx.globalAlpha = alpha;

    const boxX_img = (box.x - box.w / 2) * imgW;
    const boxY_img = (box.y - box.h / 2) * imgH;
    const boxW_img = box.w * imgW;
    const boxH_img = box.h * imgH;

    const x = offsetX + boxX_img * scaleX;
    const y = offsetY + boxY_img * scaleY;
    const w = boxW_img * scaleX;
    const h = boxH_img * scaleY;

    ctx.strokeStyle = box.color;
    ctx.setLineDash(box.dashed ? [lineWidth * 3, lineWidth * 2] : []);

    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.stroke();

    ctx.setLineDash([]);

    if (config.showLabels !== false) {
      let label = '';
      if (box.type === BoxType.FN) {
        label = `GT ${box.classId}`;
      } else if (box.type === BoxType.TP_GT) {
        label = `GT ${box.classId}`;
      } else {
        const conf = box.confidence !== undefined ? box.confidence.toFixed(2) : '1.00';
        label = `${box.type.replace('_PRED', '')} ${conf}`;
      }

      const padding = Math.max(2, baseFontSize * 0.2);

      ctx.save();
      ctx.font = `bold ${baseFontSize}px sans-serif`;
      ctx.textBaseline = 'top';
      ctx.lineJoin = 'round';

      ctx.lineWidth = Math.max(2, baseFontSize * 0.2);
      ctx.strokeStyle = '#000000';
      ctx.strokeText(label, x + padding, y + padding);

      ctx.fillStyle = box.color;
      ctx.fillText(label, x + padding, y + padding);

      ctx.restore();
    }
  });

  ctx.globalAlpha = 1.0;

  if (!img) {
    URL.revokeObjectURL(imageElement.src);
  }

  const stats = {
    tp: renderBoxes.filter((b) => b.type === BoxType.TP_PRED).length,
    fp: renderBoxes.filter((b) => b.type === BoxType.FP).length,
    fn: renderBoxes.filter((b) => b.type === BoxType.FN).length,
  };

  return { stats, boxes: renderBoxes };
};
