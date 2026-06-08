import React, { useEffect, useRef, useState, useLayoutEffect } from 'react';
import { Download, Loader2, Trash2, Check, Copy } from 'lucide-react';
import { ImageItem, VisualizationConfig, BoxType, RenderBox, BoundingBox } from '../types';
import { drawVisualization } from '../utils/render';
import { calculateMatches, applyNmsIou } from '../utils/yolo';
import { getAudioFilename, extractStartTimeFromFilename } from '../utils/audio';

interface ImageViewerProps {
  item: ImageItem;
  config: VisualizationConfig;
  externalHighlight?: BoxType | null;
  isEditMode?: boolean;
  onUpdateGt?: (fileName: string, newBoxes: BoundingBox[]) => void;
  audioPlayer?: any;
  audioFiles?: Record<string, File | FileSystemFileHandle>;
  activePlayback?: { id: number; fileName: string } | null;
  onSetGlobalPlayback?: (pb: { id: number; fileName: string } | null) => void;
  isFocused?: boolean;
  onFocusToggle?: () => void;
  onSetFocus?: () => void;
  onRecoverOriginalGt?: (fileName: string) => void;
}

interface DragState {
  mode: 'move' | 'resize' | 'create';
  boxIndex: number;
  startX: number;
  startY: number;
  initialBox?: BoundingBox;
  handle?: 'tl' | 'tr' | 'bl' | 'br';
  potentialSelect?: number;
}

const ImageViewer: React.FC<ImageViewerProps> = ({ item, config, externalHighlight, isEditMode, onUpdateGt, audioPlayer, audioFiles, activePlayback, onSetGlobalPlayback, isFocused, onFocusToggle, onSetFocus, onRecoverOriginalGt }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const playbackIdRef = useRef<number>(0);

  const [localGtBoxes, setLocalGtBoxes] = useState<BoundingBox[]>(item.gtData || []);
  const [selectedBoxIdx, setSelectedBoxIdx] = useState<number | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [playingBox, setPlayingBox] = useState<BoundingBox | null>(null);
  const [hidePlayingBorder, setHidePlayingBorder] = useState(false);
  const [playhead, setPlayhead] = useState<number | null>(null);
  const [tempAudioBox, setTempAudioBox] = useState<BoundingBox | null>(null);
  const [contextPredBox, setContextPredBox] = useState<BoundingBox | null>(null);
  const localPlaybackIdRef = useRef<number>(0);
  const activePlaybackRef = useRef(activePlayback);

  useEffect(() => {
    activePlaybackRef.current = activePlayback;
  }, [activePlayback]);

  useEffect(() => {
    if (activePlayback && activePlayback.fileName !== item.name) {
      setPlayingBox(null);
      setPlayhead(null);
      playbackIdRef.current += 1;
      localPlaybackIdRef.current = -1;
    }
  }, [activePlayback, item.name]);

  const [cachedData, setCachedData] = useState<{
    img: HTMLImageElement;
    boxes: RenderBox[];
  } | null>(null);

  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const [hoveredStat, setHoveredStat] = useState<BoxType | null>(null);

  const [computedStats, setComputedStats] = useState({ tp: 0, fp: 0, fn: 0 });

  const [lockedStat, setLockedStat] = useState<BoxType | null>(null);

  const [hoverCoords, setHoverCoords] = useState<{ x: number, y: number, timePx: number, freqHz: number } | null>(null);

  useEffect(() => {
    setLocalGtBoxes(item.gtData || []);
  }, [item.gtData]);

  useEffect(() => {
    setSelectedBoxIdx(null);
    setPlayingBox(null);
    setHidePlayingBorder(false);
    setTempAudioBox(null);
  }, [item.name]);

  useEffect(() => {
    const closeMenu = () => {
      setContextMenu(null);
      setContextPredBox(null);
    };
    if (contextMenu) window.addEventListener('click', closeMenu);
    return () => window.removeEventListener('click', closeMenu);
  }, [contextMenu]);

  useLayoutEffect(() => {
    let active = true;

    const render = async () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      let img = cachedData?.img;
      if (!img || img.getAttribute('data-name') !== item.name) {
        setLoading(true);
        img = new Image();
        img.setAttribute('data-name', item.name);
        const file = item.file;
        const url = file instanceof File ? URL.createObjectURL(file) : URL.createObjectURL(await (file as FileSystemFileHandle).getFile());
        img.src = url;
        await new Promise((resolve) => {
          img!.onload = resolve;
        });
        URL.revokeObjectURL(url);
      }

      if (!active) return;

      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');

      if (ctx) {
        const currentGt = isEditMode ? localGtBoxes : (item.gtData || []);
        const shouldShowPreds = !(isEditMode && config.showPredictions === false);
        const visibleSources = item.predictions?.filter(p => p.visible) || [];

        let renderBoxes: RenderBox[] = [];

        if (shouldShowPreds && visibleSources.length > 0) {
          const primaryMatches = calculateMatches(currentGt, visibleSources[0].boxes, config);
          primaryMatches.forEach(m => {
            if (m.type === BoxType.TP_PRED || m.type === BoxType.FP) {
              m.color = visibleSources[0].color;
            }
          });
          renderBoxes.push(...primaryMatches);

          for (let i = 1; i < visibleSources.length; i++) {
            const src = visibleSources[i];
            const filtered = src.boxes.filter((b) => (b.confidence || 1) >= config.confThreshold);
            applyNmsIou(filtered, config.nmsIouThreshold).forEach((b) => {
              renderBoxes.push({
                ...b,
                type: BoxType.TP_PRED,
                color: src.color,
                dashed: false
              });
            });
          }
        } else {
          renderBoxes = calculateMatches(currentGt, [], config);
        }

        const activeHighlightType = lockedStat || hoveredStat || externalHighlight;

        const result = await drawVisualization(ctx, item, config, img.naturalWidth, img.naturalHeight, img, {
          preCalculatedBoxes: renderBoxes,
          highlightType: activeHighlightType
        });

        setCachedData({
          img,
          boxes: renderBoxes,
        });

        setComputedStats(result.stats);

        if (isEditMode) {
          drawEditorOverlay(ctx, renderBoxes, selectedBoxIdx);
        }

        if (playingBox && !hidePlayingBorder) {
          drawPlayingHighlight(ctx, playingBox);
        } else if (playhead !== null) {
          drawPlayingHighlight(ctx, null as any);
        }
      } else if (playhead !== null) {
        drawPlayingHighlight(ctx, null as any);
      }

      if (tempAudioBox) {
        drawPlayingHighlight(ctx, tempAudioBox);
      }

      setLoading(false);
    };

    render();

    return () => { active = false; };
  }, [
    item,
    config,
    config.matchOverlapMetric,
    config.matchOverlapThreshold,
    config.confThreshold,
    config.showLabels,
    localGtBoxes,
    isEditMode,
    hoveredStat,
    lockedStat,
    externalHighlight,
    playingBox,
    playhead,
    tempAudioBox,
    hidePlayingBorder,
  ]);

  useEffect(() => {
    if (!isEditMode || selectedBoxIdx === null) return;

    const onKey = (e: KeyboardEvent) => {
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') {
        return;
      }

      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        e.stopPropagation();
        const newBoxes = localGtBoxes.filter((_, i) => i !== selectedBoxIdx);
        setLocalGtBoxes(newBoxes);
        setSelectedBoxIdx(null);
        onUpdateGt?.(item.name, newBoxes);
      }
    };

    document.addEventListener('keydown', onKey, { capture: true });
    return () => document.removeEventListener('keydown', onKey, { capture: true });
  }, [isEditMode, selectedBoxIdx, localGtBoxes, item, onUpdateGt]);

  useEffect(() => {
    if (!isFocused) return;
    const onSpace = (e: KeyboardEvent) => {
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') return;
      if (e.key === ' ') {
        e.preventDefault();
        if (playhead !== null) {
          audioPlayer?.togglePause();
        } else {
          const fullImageAudioBox: BoundingBox = {
            classId: 0, x: 0.5, y: 0.5, w: 1.0, h: 1.0, confidence: 1.0
          };
          playAudioForBox(fullImageAudioBox, false, true);
        }
      }
    };
    document.addEventListener('keydown', onSpace);
    return () => document.removeEventListener('keydown', onSpace);
  }, [isFocused, playhead, audioPlayer, item]);

  const drawPlayingHighlight = (ctx: CanvasRenderingContext2D, box: BoundingBox | null) => {
    const { width, height } = ctx.canvas;

    if (box) {
      const x = box.x * width;
      const y = box.y * height;
      const w = box.w * width;
      const h = box.h * height;

      const highlightColor = config.audio?.highlightColor ?? '#00ff00';

      ctx.save();
      ctx.strokeStyle = highlightColor;
      ctx.lineWidth = 4;
      ctx.shadowColor = highlightColor;
      ctx.shadowBlur = 10;
      ctx.strokeRect(x - w / 2, y - h / 2, w, h);
      ctx.restore();
    }

    ctx.save();

    if (playhead !== null) {
      const actualW = box ? (box.w * width) : width;
      const actualX = box ? ((box.x * width) - actualW / 2) : 0;
      const actualY = box ? (box.y * height) : (height / 2);
      const actualH = box ? (box.h * height) : height;

      const playX = actualX + (actualW * playhead);
      ctx.beginPath();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.shadowBlur = 0;
      ctx.setLineDash([2, 5]);
      ctx.moveTo(playX, actualY - actualH / 2);
      ctx.lineTo(playX, actualY + actualH / 2);
      ctx.stroke();
    }

    ctx.restore();
  };


  const drawEditorOverlay = (ctx: CanvasRenderingContext2D, boxes: RenderBox[], selectedIdx: number | null) => {
    if (selectedIdx === null) return;
    const box = localGtBoxes[selectedIdx];
    if (!box) return;

    const { width, height } = ctx.canvas;
    const x = box.x * width;
    const y = box.y * height;
    const w = box.w * width;
    const h = box.h * height;

    const lx = x - w / 2;
    const ly = y - h / 2;

    ctx.save();

    const highlightColor = config.editHighlightColor || '#fbbf24';

    const r = parseInt(highlightColor.slice(1, 3), 16);
    const g = parseInt(highlightColor.slice(3, 5), 16);
    const b = parseInt(highlightColor.slice(5, 7), 16);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.15)`;
    ctx.fillRect(lx, ly, w, h);

    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(lx, ly, w, h);

    ctx.setLineDash([]);
    ctx.strokeStyle = highlightColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(lx, ly, w, h);

    const handleSize = 10;
    const handles = [
      { x: lx, y: ly },
      { x: lx + w, y: ly },
      { x: lx, y: ly + h },
      { x: lx + w, y: ly + h }
    ];

    ctx.fillStyle = highlightColor;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    handles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });

    ctx.restore();
  };

  const getImgCoords = (e: React.MouseEvent | MouseEvent) => {
    if (!canvasRef.current || !cachedData) return { x: 0, y: 0, rawX: 0, rawY: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;

    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;

    const nx = Math.max(0, Math.min(1, cx / canvasRef.current.width));
    const ny = Math.max(0, Math.min(1, cy / canvasRef.current.height));

    const clampedRawX = Math.max(0, Math.min(canvasRef.current.width, cx));
    const clampedRawY = Math.max(0, Math.min(canvasRef.current.height, cy));

    return { x: nx, y: ny, rawX: clampedRawX, rawY: clampedRawY };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0 && !(e.button === 2 && isEditMode)) return;

    containerRef.current?.focus();

    const coords = getImgCoords(e);

    if (isEditMode && e.button === 2) {
      setDragState({
        mode: 'create',
        boxIndex: -2,
        startX: coords.x,
        startY: coords.y
      });
      audioPlayer?.stop();
      setPlayingBox(null);
      setPlayhead(null);
      playbackIdRef.current += 1;
      setTempAudioBox({ classId: 0, x: coords.x, y: coords.y, w: 0, h: 0, confidence: 1 });
      return;
    }
    const { width, height } = canvasRef.current!;

    if (!isEditMode) {
      if (!cachedData) return;
      let hit: BoundingBox | null = null;
      let minArea = Infinity;

      for (const b of cachedData.boxes) {
        const bx1 = b.x - b.w / 2;
        const bx2 = b.x + b.w / 2;
        const by1 = b.y - b.h / 2;
        const by2 = b.y + b.h / 2;

        if (coords.x >= bx1 && coords.x <= bx2 && coords.y >= by1 && coords.y <= by2) {
          const area = b.w * b.h;
          if (area < minArea) {
            minArea = area;
            hit = b;
          }
        }
      }

      setDragState({
        mode: 'create',
        boxIndex: -2,
        startX: coords.x,
        startY: coords.y,
        initialBox: hit ? { ...hit } : undefined
      });
      audioPlayer?.stop();
      setPlayingBox(null);
      setPlayhead(null);
      playbackIdRef.current += 1;
      setTempAudioBox({ classId: 0, x: coords.x, y: coords.y, w: 0, h: 0, confidence: 1 });
      return;
    }

    for (let i = localGtBoxes.length - 1; i >= 0; i--) {
      if (i !== selectedBoxIdx) continue;
      const b = localGtBoxes[i];
      const lx = (b.x - b.w / 2) * width;
      const ly = (b.y - b.h / 2) * height;
      const rw = b.w * width;
      const rh = b.h * height;

      const handles: Record<'tl' | 'tr' | 'bl' | 'br', { x: number, y: number }> = {
        tl: { x: lx, y: ly },
        tr: { x: lx + rw, y: ly },
        bl: { x: lx, y: ly + rh },
        br: { x: lx + rw, y: ly + rh }
      };

      let bestHandle = null;
      let minDistance = Infinity;

      for (const [key, p] of Object.entries(handles)) {
        const dx = coords.rawX - p.x;
        const dy = coords.rawY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 30 && dist < minDistance) {
          minDistance = dist;
          bestHandle = key;
        }
      }

      if (bestHandle) {
        setSelectedBoxIdx(i);
        setDragState({
          mode: 'resize',
          handle: bestHandle as any,
          boxIndex: i,
          startX: coords.x,
          startY: coords.y,
          initialBox: { ...b }
        });
        return;
      }
    }

    let hit = -1;
    let minArea = Infinity;

    for (let i = localGtBoxes.length - 1; i >= 0; i--) {
      const b = localGtBoxes[i];
      const bx1 = b.x - b.w / 2;
      const bx2 = b.x + b.w / 2;
      const by1 = b.y - b.h / 2;
      const by2 = b.y + b.h / 2;

      if (coords.x >= bx1 && coords.x <= bx2 && coords.y >= by1 && coords.y <= by2) {
        const area = b.w * b.h;
        if (area < minArea) {
          minArea = area;
          hit = i;
        }
      }
    }

    if (hit !== -1 && hit === selectedBoxIdx) {
      setDragState({
        mode: 'move',
        boxIndex: hit,
        startX: coords.x,
        startY: coords.y,
        initialBox: { ...localGtBoxes[hit] }
      });
    } else {
      const newBox: BoundingBox = {
        classId: 0,
        x: coords.x,
        y: coords.y,
        w: 0,
        h: 0,
        confidence: 1.0
      };

      const newBoxes = [...localGtBoxes, newBox];
      setLocalGtBoxes(newBoxes);
      const newIndex = newBoxes.length - 1;

      setSelectedBoxIdx(newIndex);
      setDragState({
        mode: 'create',
        boxIndex: newIndex,
        startX: coords.x,
        startY: coords.y,
        potentialSelect: hit !== -1 ? hit : undefined
      });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getImgCoords(e);

    if (dragState) {
      e.preventDefault();

      if (dragState.mode === 'create' && dragState.boxIndex === -2) {
        const startX = dragState.startX;
        const startY = dragState.startY;
        const currentX = coords.x;
        const currentY = coords.y;

        const minX = Math.min(startX, currentX);
        const maxX = Math.max(startX, currentX);
        const minY = Math.min(startY, currentY);
        const maxY = Math.max(startY, currentY);

        const w = maxX - minX;
        const h = maxY - minY;
        const x = minX + w / 2;
        const y = minY + h / 2;

        setTempAudioBox({ classId: 0, x, y, w, h, confidence: 1 });
        return;
      }

      if (!isEditMode) return;

      if (dragState.mode === 'move' && dragState.initialBox) {
        const dx = coords.x - dragState.startX;
        const dy = coords.y - dragState.startY;

        let newX = dragState.initialBox.x + dx;
        let newY = dragState.initialBox.y + dy;
        const halfW = dragState.initialBox.w / 2;
        const halfH = dragState.initialBox.h / 2;

        newX = Math.max(halfW, Math.min(1 - halfW, newX));
        newY = Math.max(halfH, Math.min(1 - halfH, newY));

        const newBoxes = [...localGtBoxes];
        newBoxes[dragState.boxIndex] = {
          ...dragState.initialBox,
          x: newX,
          y: newY
        };
        setLocalGtBoxes(newBoxes);
      } else if (dragState.mode === 'resize' && dragState.initialBox && dragState.handle) {
        const b = dragState.initialBox;
        let x1 = b.x - b.w / 2;
        let y1 = b.y - b.h / 2;
        let x2 = b.x + b.w / 2;
        let y2 = b.y + b.h / 2;

        if (dragState.handle === 'tl') { x1 = coords.x; y1 = coords.y; }
        else if (dragState.handle === 'tr') { x2 = coords.x; y1 = coords.y; }
        else if (dragState.handle === 'bl') { x1 = coords.x; y2 = coords.y; }
        else if (dragState.handle === 'br') { x2 = coords.x; y2 = coords.y; }

        const w = Math.abs(x2 - x1);
        const h = Math.abs(y2 - y1);
        const x = (x1 + x2) / 2;
        const y = (y1 + y2) / 2;

        const newBoxes = [...localGtBoxes];
        newBoxes[dragState.boxIndex] = { ...b, x, y, w, h };
        setLocalGtBoxes(newBoxes);
      } else if (dragState.mode === 'create') {
        const startX = dragState.startX;
        const startY = dragState.startY;
        const currentX = coords.x;
        const currentY = coords.y;

        const minX = Math.min(startX, currentX);
        const maxX = Math.max(startX, currentX);
        const minY = Math.min(startY, currentY);
        const maxY = Math.max(startY, currentY);

        const w = maxX - minX;
        const h = maxY - minY;
        const x = minX + w / 2;
        const y = minY + h / 2;

        const newBoxes = [...localGtBoxes];
        if (newBoxes[dragState.boxIndex]) {
          newBoxes[dragState.boxIndex] = {
            ...newBoxes[dragState.boxIndex],
            x, y, w, h
          };
          setLocalGtBoxes(newBoxes);
        }
      }
      return;
    }

    if (!cachedData || !canvasRef.current) return;

    const imageStartTime = extractStartTimeFromFilename(item.name);
    const clipSec = config.audio?.clipSec ?? 6.0;
    const TIME_TOTAL = clipSec * 1000;
    const timeAtCursor = imageStartTime + (coords.x * TIME_TOTAL);

    const minF = config.audio?.minFreq ?? 500;
    const maxF = config.audio?.maxFreq ?? 12000;
    const freqAtCursor = maxF - (coords.y * (maxF - minF));

    let cursorX = coords.rawX;
    let cursorY = coords.rawY;
    if (containerRef.current) {
      const containerRect = containerRef.current.getBoundingClientRect();
      cursorX = e.clientX - containerRect.left;
      cursorY = e.clientY - containerRect.top;
    }

    setHoverCoords({
      x: cursorX,
      y: cursorY,
      timePx: timeAtCursor / 1000,
      freqHz: freqAtCursor
    });

  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (dragState) {
      if (dragState.mode === 'create' && dragState.boxIndex === -2 && tempAudioBox) {
        if (tempAudioBox.w > 0.005 && tempAudioBox.h > 0.005) {
          playAudioForBox(tempAudioBox, e.shiftKey);
        } else {
          if (!isEditMode) {
            if (dragState.initialBox) {
              playAudioForBox(dragState.initialBox, e.shiftKey);
            } else {
              const fullImageAudioBox: BoundingBox = {
                classId: 0,
                x: 0.5, y: 0.5, w: 1.0, h: 1.0, confidence: 1.0
              };
              playAudioForBox(fullImageAudioBox, false, true);
            }
          } else {
            openContextMenuInEditMode(e.clientX, e.clientY);
          }
        }
        setTempAudioBox(null);
        setDragState(null);
        return;
      }

      if (!isEditMode) {
        setDragState(null);
        return;
      }

      if (dragState.mode === 'create') {
        const box = localGtBoxes[dragState.boxIndex];
        if (box && (box.w < 0.001 || box.h < 0.001)) {
          const newBoxes = localGtBoxes.filter((_, i) => i !== dragState.boxIndex);
          setLocalGtBoxes(newBoxes);
          if (dragState.potentialSelect !== undefined) {
            setSelectedBoxIdx(dragState.potentialSelect);
          } else {
            setSelectedBoxIdx(null);
          }
        } else {
          onUpdateGt?.(item.name, localGtBoxes);
        }
      } else if (dragState.mode === 'move' || dragState.mode === 'resize') {
        const currentBox = localGtBoxes[dragState.boxIndex];
        const initial = dragState.initialBox;
        if (currentBox && initial) {
          if (currentBox.x !== initial.x || currentBox.y !== initial.y || currentBox.w !== initial.w || currentBox.h !== initial.h) {
            onUpdateGt?.(item.name, localGtBoxes);
          }
        } else {
          onUpdateGt?.(item.name, localGtBoxes);
        }
      }
      setDragState(null);
    }
  };

  useEffect(() => {
    if (!dragState) return;

    const handleGlobalMouseMove = (e: MouseEvent) => {
      handleMouseMove(e as unknown as React.MouseEvent<HTMLCanvasElement>);
    };

    const handleGlobalMouseUp = (e: MouseEvent) => {
      handleMouseUp(e as unknown as React.MouseEvent);
    };

    window.addEventListener('mousemove', handleGlobalMouseMove);
    window.addEventListener('mouseup', handleGlobalMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [dragState, handleMouseMove, handleMouseUp]);

  const handleMouseLeaveCanvas = () => {
    setHoveredStat(null);
    setHoverCoords(null);
  };

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const srcCanvas = canvasRef.current;
      if (!srcCanvas) return;

      const imgWidth = srcCanvas.width;
      const imgHeight = srcCanvas.height;

      const headerHeight = Math.max(30, Math.floor(imgHeight * 0.05));
      const canvas = document.createElement('canvas');
      canvas.width = imgWidth;
      canvas.height = imgHeight + headerHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width, headerHeight);

      ctx.drawImage(srcCanvas, 0, headerHeight);

      const textPadding = Math.max(10, canvas.width * 0.01);
      ctx.fillStyle = '#cbd5e1';
      ctx.font = `bold ${Math.max(14, headerHeight * 0.4)}px monospace`;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      ctx.fillText(item.name, textPadding, headerHeight / 2);

      const statFontSize = Math.max(12, headerHeight * 0.4);
      ctx.font = `bold ${statFontSize}px monospace`;

      const fpText = `FP:${computedStats.fp}`;
      const fnText = `FN:${computedStats.fn}`;
      const tpText = `TP:${computedStats.tp}`;

      let xOffset = canvas.width - textPadding;
      ctx.textAlign = 'right';

      ctx.fillStyle = config.styles.fp.color;
      ctx.fillText(fpText, xOffset, headerHeight / 2);
      xOffset -= ctx.measureText(fpText).width + statFontSize;

      ctx.fillStyle = config.styles.fn.color;
      ctx.fillText(fnText, xOffset, headerHeight / 2);
      xOffset -= ctx.measureText(fnText).width + statFontSize;

      ctx.fillStyle = config.styles.tpPred.color;
      ctx.fillText(tpText, xOffset, headerHeight / 2);

      const link = document.createElement('a');
      link.download = `vis_${item.name}`;
      link.href = canvas.toDataURL('image/jpeg', 0.95);
      link.click();
    } catch (e) {
      console.error("Download failed", e);
    } finally {
      setDownloading(false);
      setContextMenu(null);
    }
  };

  const openContextMenuInEditMode = (clientX: number, clientY: number) => {
    if (!canvasRef.current || !cachedData) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;

    const cx = (clientX - rect.left) * scaleX;
    const cy = (clientY - rect.top) * scaleY;

    const nx = cx / canvasRef.current.width;
    const ny = cy / canvasRef.current.height;

    let hitGt = -1;
    let hitGtArea = Infinity;

    let hitPredBox: BoundingBox | null = null;
    let hitPredArea = Infinity;

    for (let i = localGtBoxes.length - 1; i >= 0; i--) {
      const b = localGtBoxes[i];
      const bx1 = b.x - b.w / 2;
      const bx2 = b.x + b.w / 2;
      const by1 = b.y - b.h / 2;
      const by2 = b.y + b.h / 2;

      if (nx >= bx1 && nx <= bx2 && ny >= by1 && ny <= by2) {
        const area = b.w * b.h;
        if (area < hitGtArea) {
          hitGtArea = area;
          hitGt = i;
        }
      }
    }

    if (config.showPredictions !== false && cachedData) {
      for (const b of cachedData.boxes) {
        if (b.type === BoxType.TP_PRED || b.type === BoxType.FP) {
          const bx1 = b.x - b.w / 2;
          const bx2 = b.x + b.w / 2;
          const by1 = b.y - b.h / 2;
          const by2 = b.y + b.h / 2;
          if (nx >= bx1 && nx <= bx2 && ny >= by1 && ny <= by2) {
            const area = b.w * b.h;
            if (area < hitPredArea) {
              hitPredArea = area;
              hitPredBox = b;
            }
          }
        }
      }
    }

    if (hitGt !== -1 || hitPredBox) {
      setSelectedBoxIdx(hitGt !== -1 ? hitGt : null);
      setContextPredBox(hitPredBox);
    } else {
      setSelectedBoxIdx(null);
      setContextPredBox(null);
    }

    const x = Math.min(clientX, window.innerWidth - 200);
    const y = Math.min(clientY, window.innerHeight - 80);
    setContextMenu({ x, y });
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();

    if (isEditMode) {
      return;
    }

    const x = Math.min(e.clientX, window.innerWidth - 200);
    const y = Math.min(e.clientY, window.innerHeight - 80);
    setContextMenu({ x, y });
  };



  const playAudioForBox = async (box: BoundingBox, fullBand: boolean = false, hideBorder: boolean = false) => {
    if (!audioPlayer || !audioFiles) return;

    const audioName = getAudioFilename(item.name, audioFiles);
    if (!audioName) {
      console.warn("Audio file not found for image:", item.name);
      alert(`Audio file not found for: ${item.name}`);
      return;
    }
    const fileHandle = audioFiles[audioName];

    await audioPlayer.loadAudioFile(fileHandle);

    const imageStartTime = extractStartTimeFromFilename(item.name);
    const clipSec = config.audio?.clipSec ?? 6.0;
    const IMAGE_DURATION_MS = clipSec * 1000;

    const startX = box.x - box.w / 2;
    const startTime = imageStartTime + (startX * IMAGE_DURATION_MS);
    const duration = box.w * IMAGE_DURATION_MS;
    const playbackSpeed = config.audio?.playbackSpeed ?? 1.0;

    const minF = config.audio?.minFreq ?? 500;
    const maxF = config.audio?.maxFreq ?? 12000;

    let freqTop = maxF;
    let freqBottom = minF;

    if (!fullBand) {
      const boxTopY = box.y - box.h / 2;
      const boxBottomY = box.y + box.h / 2;
      freqTop = maxF - (boxTopY * (maxF - minF));
      freqBottom = maxF - (boxBottomY * (maxF - minF));
    }

    const clampFreq = (f: number) => Math.max(minF, Math.min(maxF, f));
    const finalMinFreq = clampFreq(Math.min(freqTop, freqBottom));
    const finalMaxFreq = clampFreq(Math.max(freqTop, freqBottom));

    const actualBox = fullBand ? { ...box, y: 0.5, h: 1.0 } : box;

    setPlayingBox(actualBox);

    setHidePlayingBorder(hideBorder || false);
    setPlayhead(0);
    setTempAudioBox(null);

    playbackIdRef.current += 1;
    const currentPlaybackId = playbackIdRef.current;

    const finishPlayback = () => {
      if (playbackIdRef.current !== currentPlaybackId) return;
      playbackIdRef.current += 1;
      setPlayhead(null);
      setPlayingBox(null);
      if (activePlaybackRef.current?.id === currentPlaybackId) {
        onSetGlobalPlayback?.(null);
      }
    };

    localPlaybackIdRef.current = currentPlaybackId;
    onSetGlobalPlayback?.({ id: currentPlaybackId, fileName: item.name });

    await audioPlayer.playSubRegion({
      startTimeMs: startTime,
      durationMs: duration,
      minFreq: finalMinFreq,
      maxFreq: finalMaxFreq,
      playbackSpeed: playbackSpeed,
      onFinish: () => {
        finishPlayback();
      }
    });

    let lastTime = performance.now();
    let accumulatedTime = 0;

    const animate = (now: number) => {
      if (playbackIdRef.current !== currentPlaybackId) return;

      const dt = now - lastTime;
      lastTime = now;

      if (!audioPlayer?.isPaused?.()) {
        accumulatedTime += dt;
      }

      const progress = Math.min(accumulatedTime / (duration / playbackSpeed), 1);

      setPlayhead(progress);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    requestAnimationFrame(animate);

    setContextMenu(null);
  };

  const handlePlayAudio = async () => {
    let box: BoundingBox | undefined;
    if (selectedBoxIdx !== null) {
      box = localGtBoxes[selectedBoxIdx];
    } else if (cachedData && contextMenu) {
      if (canvasRef.current) {
        const rect = canvasRef.current.getBoundingClientRect();
        const scaleX = canvasRef.current.width / rect.width;
        const scaleY = canvasRef.current.height / rect.height;
        const mx = (contextMenu.x - rect.left) * scaleX;
        const my = (contextMenu.y - rect.top) * scaleY;
        const nx = mx / canvasRef.current.width;
        const ny = my / canvasRef.current.height;

        let bestMatch = null;
        let minArea = Infinity;

        for (const b of cachedData.boxes) {
          const bx1 = b.x - b.w / 2;
          const bx2 = b.x + b.w / 2;
          const by1 = b.y - b.h / 2;
          const by2 = b.y + b.h / 2;

          if (nx >= bx1 && nx <= bx2 && ny >= by1 && ny <= by2) {
            const area = b.w * b.h;
            if (area < minArea) {
              minArea = area;
              bestMatch = b;
            }
          }
        }
        if (bestMatch) box = bestMatch;
      }
    }

    if (!box) return;
    await playAudioForBox(box);
  };

  const handleDeleteSelected = () => {
    if (selectedBoxIdx !== null && isEditMode) {
      const newBoxes = localGtBoxes.filter((_, i) => i !== selectedBoxIdx);
      setLocalGtBoxes(newBoxes);
      setSelectedBoxIdx(null);
      onUpdateGt?.(item.name, newBoxes);
      setContextMenu(null);
    }
  };

  let aspectClass = "aspect-square";
  if (config.aspectRatio === '16:9') aspectClass = "aspect-video";
  else if (config.aspectRatio === '4:3') aspectClass = "aspect-[4/3]";
  else if (config.aspectRatio === '1:1') aspectClass = "aspect-square";
  else if (config.aspectRatio === 'auto') aspectClass = config.gridSize === 1 ? "aspect-auto min-h-[500px] h-auto" : "aspect-auto h-auto";

  const isLocalModified = JSON.stringify(localGtBoxes) !== JSON.stringify(item.gtData || []);
  const showModifiedBadge = isLocalModified || item.isModified;

  return (
    <div
      className={`relative bg-slate-900 rounded-lg overflow-hidden border ${isFocused ? 'border-primary ring-2 ring-primary/50' : 'border-slate-700'} ${aspectClass} flex flex-col w-full focus:outline-none transition-all cursor-pointer`}
      onContextMenu={handleContextMenu}
      onClick={onSetFocus}
      onDoubleClick={onFocusToggle}
      tabIndex={isEditMode || isFocused ? 0 : undefined}
      onKeyDown={(e) => {
        if (isEditMode && selectedBoxIdx !== null && (e.key === 'Backspace' || e.key === 'Delete')) {
          e.preventDefault();
          e.stopPropagation();
          handleDeleteSelected();
        }
      }}
    >
      <div className="flex justify-between items-center w-full bg-slate-800/80 px-2 py-1 shrink-0 border-b border-slate-700 z-20 h-7 transition-all">
        <div className="text-[10px] font-mono font-bold text-slate-300 truncate max-w-[60%] flex items-center gap-2">
          <span className="truncate">{item.name}</span>
          {showModifiedBadge && (
            <span className="shrink-0 text-amber-500 border border-amber-500/40 px-1.5 py-0.5 rounded-[2px] text-[8px] leading-none bg-amber-500/5">MOD</span>
          )}
          {item.isSaved && !showModifiedBadge && (
            <span className="shrink-0 text-emerald-500 border border-emerald-500/40 px-1.5 py-0.5 rounded-[2px] text-[8px] leading-none bg-emerald-500/5 flex items-center gap-0.5">
              <Check className="w-2 h-2" /> SAVED
            </span>
          )}
        </div>
        <div className="flex gap-2 text-[10px] font-mono font-bold">
          <span
            onMouseEnter={() => !lockedStat && setHoveredStat(BoxType.TP_PRED)}
            onMouseLeave={() => !lockedStat && setHoveredStat(null)}
            onClick={(e) => {
              e.stopPropagation();
              if (lockedStat === BoxType.TP_PRED) {
                setLockedStat(null);
                setHoveredStat(null);
              } else {
                setLockedStat(BoxType.TP_PRED);
                setHoveredStat(BoxType.TP_PRED);
              }
            }}
            className={`cursor-pointer px-1.5 py-0.5 rounded transition-all select-none ${lockedStat === BoxType.TP_PRED ? 'bg-white/20 ring-1 ring-white/50' : 'hover:bg-white/10'}`}
            style={{ color: config.styles.tpGt.color }}
          >
            TP:{computedStats.tp}
          </span>
          <span
            onMouseEnter={() => !lockedStat && setHoveredStat(BoxType.FN)}
            onMouseLeave={() => !lockedStat && setHoveredStat(null)}
            onClick={(e) => {
              e.stopPropagation();
              if (lockedStat === BoxType.FN) {
                setLockedStat(null);
                setHoveredStat(null);
              } else {
                setLockedStat(BoxType.FN);
                setHoveredStat(BoxType.FN);
              }
            }}
            className={`cursor-pointer px-1.5 py-0.5 rounded transition-all select-none ${lockedStat === BoxType.FN ? 'bg-white/20 ring-1 ring-white/50' : 'hover:bg-white/10'}`}
            style={{ color: config.styles.fn.color }}
          >
            FN:{computedStats.fn}
          </span>
          <span
            onMouseEnter={() => !lockedStat && setHoveredStat(BoxType.FP)}
            onMouseLeave={() => !lockedStat && setHoveredStat(null)}
            onClick={(e) => {
              e.stopPropagation();
              if (lockedStat === BoxType.FP) {
                setLockedStat(null);
                setHoveredStat(null);
              } else {
                setLockedStat(BoxType.FP);
                setHoveredStat(BoxType.FP);
              }
            }}
            className={`cursor-pointer px-1.5 py-0.5 rounded transition-all select-none ${lockedStat === BoxType.FP ? 'bg-white/20 ring-1 ring-white/50' : 'hover:bg-white/10'}`}
            style={{ color: config.styles.fp.color }}
          >
            FP:{computedStats.fp}
          </span>
        </div>
      </div>

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900 z-10">
          <Loader2 className="w-8 h-8 text-primary animate-spin" />
        </div>
      )}


      <div className="relative flex-1 flex items-center justify-center min-h-0 min-w-0" ref={containerRef}>
        <canvas
          ref={canvasRef}
          className={`max-w-full max-h-full object-contain ${isEditMode ? 'cursor-default' : 'cursor-crosshair'}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeaveCanvas}
        />

        {hoverCoords && !isEditMode && (
          <>
            <div className="absolute top-0 bottom-0 border-l border-white/40 border-dashed pointer-events-none z-10" style={{ left: hoverCoords.x }} />
            <div className="absolute left-0 right-0 border-t border-white/40 border-dashed pointer-events-none z-10" style={{ top: hoverCoords.y }} />

            <div
              className="absolute left-1 bg-slate-800 text-white text-[10px] px-1.5 py-0.5 rounded shadow pointer-events-none z-20 border border-slate-600 font-mono -translate-y-1/2"
              style={{ top: hoverCoords.y }}
            >
              {Math.round(hoverCoords.freqHz)}Hz
            </div>

            <div
              className="absolute bottom-1 bg-slate-800 text-white text-[10px] px-1.5 py-0.5 rounded shadow pointer-events-none z-20 border border-slate-600 font-mono -translate-x-1/2"
              style={{ left: hoverCoords.x }}
            >
              {hoverCoords.timePx.toFixed(2)}s
            </div>
          </>
        )}
      </div>


      {contextMenu && (
        <div
          className="fixed z-50 bg-slate-800 border border-slate-700 rounded shadow-xl py-1 min-w-[160px] flex flex-col"
          style={{ top: contextMenu.y, left: contextMenu.x }}
          onClick={(e) => e.stopPropagation()}
        >
          {isEditMode ? (
            <>
              {selectedBoxIdx !== null && (
                <>
                  <button
                    onClick={handleDeleteSelected}
                    className="text-left px-4 py-2 text-xs text-red-400 hover:bg-slate-700 hover:text-red-300 flex items-center gap-2 transition-colors w-full border-b border-slate-700"
                  >
                    <Trash2 className="w-3 h-3" />
                    Delete Box
                  </button>
                  {audioFiles && Object.keys(audioFiles).length > 0 && (
                    <button
                      onClick={() => playAudioForBox(localGtBoxes[selectedBoxIdx])}
                      className="text-left px-4 py-2 text-xs text-indigo-300 hover:bg-slate-700 hover:text-indigo-200 flex items-center gap-2 transition-colors w-full"
                    >
                      <Download className="w-3 h-3 rotate-90" />
                      Play Audio Region
                    </button>
                  )}
                </>
              )}
              {contextPredBox && (
                <button
                  onClick={() => {
                    const { classId, x, y, w, h } = contextPredBox;
                    const newBox: BoundingBox = { classId, x, y, w, h, confidence: 1.0 };
                    const newBoxes = [...localGtBoxes, newBox];
                    setLocalGtBoxes(newBoxes);
                    onUpdateGt?.(item.name, newBoxes);
                    setContextMenu(null);
                    setContextPredBox(null);
                  }}
                  className="text-left px-4 py-2 text-xs text-emerald-400 hover:bg-slate-700 hover:text-emerald-300 flex items-center gap-2 transition-colors w-full"
                >
                  <Check className="w-3 h-3" />
                  Accept Prediction
                </button>
              )}
              <button
                onClick={() => {
                  navigator.clipboard.writeText(item.name);
                  setContextMenu(null);
                }}
                className="text-left px-4 py-2 text-xs text-slate-300 hover:bg-slate-700 hover:text-white flex items-center gap-2 transition-colors w-full border-t border-slate-700 mt-1 pt-2"
              >
                <Copy className="w-3 h-3" />
                Copy Filename
              </button>
              <button
                onClick={() => {
                  setLocalGtBoxes([]);
                  onUpdateGt?.(item.name, []);
                  setContextMenu(null);
                }}
                className="text-left px-4 py-2 text-xs text-red-500 hover:bg-slate-700 hover:text-red-400 flex items-center gap-2 transition-colors w-full border-t border-slate-700 mt-1 pt-2"
              >
                <Trash2 className="w-3 h-3" />
                Delete All GTs
              </button>
              {onRecoverOriginalGt && (
                <button
                  onClick={() => {
                    onRecoverOriginalGt(item.name);
                    setContextMenu(null);
                  }}
                  className="text-left px-4 py-2 text-xs text-blue-400 hover:bg-slate-700 hover:text-blue-300 flex items-center gap-2 transition-colors w-full"
                >
                  <Copy className="w-3 h-3" />
                  Recover original GTs
                </button>
              )}
            </>
          ) : (
            <>
              {audioFiles && Object.keys(audioFiles).length > 0 && (
                <button
                  onClick={handlePlayAudio}
                  className="text-left px-4 py-2 text-xs text-indigo-300 hover:bg-slate-700 hover:text-indigo-200 flex items-center gap-2 transition-colors w-full border-b border-slate-700"
                >
                  Play Audio Region
                </button>
              )}
              <button
                onClick={() => {
                  navigator.clipboard.writeText(item.name);
                  setContextMenu(null);
                }}
                className="text-left px-4 py-2 text-xs text-slate-300 hover:bg-slate-700 hover:text-white flex items-center gap-2 transition-colors w-full"
              >
                <Copy className="w-3 h-3" />
                Copy Filename
              </button>

              <button
                onClick={handleDownload}
                disabled={downloading}
                className="text-left px-4 py-2 text-xs text-slate-200 hover:bg-slate-700 hover:text-white flex items-center gap-2 transition-colors w-full"
              >
                {downloading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
                Save Visualization
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageViewer;