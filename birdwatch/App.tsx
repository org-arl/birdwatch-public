import React, { useState, useMemo, useRef, useEffect } from 'react';
import ControlPanel from './components/ControlPanel';
import ImageViewer from './components/ImageViewer';
import PRGraph from './components/PRGraph';
import { VisualizationConfig, ImageItem, FileMap, Project, FileCollection, BoxType, LabelMap, BoundingBox, PredictionSource } from './types';
import { ChevronLeft, ChevronRight, Inbox, Download, Loader2, ZoomIn, ZoomOut, Shuffle, PanelRight } from 'lucide-react';
import { drawVisualization } from './utils/render';
import { parseYoloFile, calculateMatches, applyNmsIou, preloadLabels } from './utils/yolo';
import { exportLabels, exportLabelsAsZip } from './utils/export';
import { AudioPlayer } from './utils/audio';
import { saveLabelToDB, getAllSavedLabels, deleteLabelFromDB } from './utils/db';
import { pickFlatDirectory, pickDatasetDirectory, isAudioFileName } from './utils/files';

const audioPlayer = new AudioPlayer();

const DEFAULT_CONFIG: VisualizationConfig = {
  matchOverlapMetric: 'iou',
  matchOverlapThreshold: 0.5,
  nmsIouThreshold: 0.7,
  confThreshold: 0.25,
  styles: {
    tpPred: { color: '#000000', dashed: false },
    tpGt: { color: '#FFFFFF', dashed: true },
    fn: { color: '#FFFF00', dashed: true },
    fp: { color: '#000000', dashed: false },
  },
  lineWidth: 4,
  labelFontSize: 23,
  gridSize: 9,
  aspectRatio: '1:1',
  zoomLevel: 1.0,
  audio: {
    minFreq: 500,
    maxFreq: 12000,
    highlightColor: '#4ade80',
    clipSec: 6.0,
    playbackSpeed: 1.0
  },
  editHighlightColor: '#fbbf24',
  showLabels: false,
  showPredictions: true
};

const generateId = () => {
  return Date.now().toString(36) + Math.random().toString(36).substring(2);
};

const createProject = (name: string): Project => ({
  id: generateId(),
  name,
  config: { ...DEFAULT_CONFIG },
  imageCollectionId: null,
  gtCollectionId: null,
  predictionSources: [],
});

const App: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>(() => [createProject('Default Project')]);
  const [activeProjectId, setActiveProjectId] = useState<string>(() => projects[0].id);
  const [collections, setCollections] = useState<FileCollection[]>([]);
  const [currentPage, setCurrentPage] = useState(0);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<{ current: number, total: number } | null>(null);
  const [activePlayback, setActivePlayback] = useState<{ id: number; fileName: string } | null>(null);
  const [isEditMode, setIsEditMode] = useState(false);
  const [modifiedFiles, setModifiedFiles] = useState<Set<string>>(new Set());
  const [savedFiles, setSavedFiles] = useState<Set<string>>(new Set());
  const [allSessionModifiedFiles, setAllSessionModifiedFiles] = useState<Set<string>>(new Set());

  const [audioFileMap, setAudioFileMap] = useState<FileMap>({});

  const [pageStats, setPageStats] = useState({ tp: 0, fp: 0, fn: 0 });
  const [globalHighlight, setGlobalHighlight] = useState<BoxType | null>(null);
  const [lockedHighlight, setLockedHighlight] = useState<BoxType | null>(null);

  const [jumpPageInput, setJumpPageInput] = useState("1");

  const [focusedItemIndex, setFocusedItemIndex] = useState(0);

  const [isolatedPredictionIds, setIsolatedPredictionIds] = useState<string[]>([]);
  const [sidebarWidth, setSidebarWidth] = useState(250);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const isDragging = useRef(false);
  const prevGridSizeRef = useRef(DEFAULT_CONFIG.gridSize);

  const activeProject = projects.find(p => p.id === activeProjectId) || projects[0];
  const { config } = activeProject;

  const imageFiles = useMemo(() =>
    collections.find(c => c.id === activeProject.imageCollectionId)?.files || {},
    [collections, activeProject.imageCollectionId]);

  const gtLabels = useMemo(() =>
    collections.find(c => c.id === activeProject.gtCollectionId)?.labels || {},
    [collections, activeProject.gtCollectionId]);

  const predictionSources = activeProject.predictionSources || [];

  const updateProject = (updates: Partial<Project>) => {
    setProjects(ps => ps.map(p => p.id === activeProjectId ? { ...p, ...updates } : p));
  };


  const setConfig = (newConfig: VisualizationConfig) => updateProject({ config: newConfig });

  const mergeIndexedDBLabels = async (targetGtId?: string) => {
    try {
      const savedInDb = await getAllSavedLabels();
      if (!savedInDb || savedInDb.length === 0) return;

      const targetId = targetGtId || activeProject.gtCollectionId;
      if (!targetId) return;

      const filenames = savedInDb.map((r: any) => r.filename);
      setModifiedFiles(prev => {
        const next = new Set(prev);
        filenames.forEach(f => next.add(f));
        return next;
      });
      setAllSessionModifiedFiles(prev => {
        const next = new Set(prev);
        filenames.forEach(f => next.add(f));
        return next;
      });
      setSavedFiles(prev => {
        const next = new Set(prev);
        filenames.forEach(f => next.delete(f));
        return next;
      });

      setCollections(prev => prev.map(c => {
        if (c.id === targetId) {
          const updatedLabels = { ...c.labels };
          savedInDb.forEach((record: any) => {
            updatedLabels[record.filename] = record.boxes;
          });
          return { ...c, labels: updatedLabels };
        }
        return c;
      }));
    } catch (e) {
      console.error("Failed to merge IndexedDB labels", e);
    }
  };

  const handleLoadImages = async () => {
    try {
      const { name, fileMap } = await pickFlatDirectory();

      const collId = generateId();
      setCollections(prev => [...prev, { id: collId, name, type: 'images', files: fileMap, count: Object.keys(fileMap).length }]);
      updateProject({ imageCollectionId: collId, imagePath: name });
      setCurrentPage(0);
      await mergeIndexedDBLabels();
    } catch (err) {
      console.error("Failed to load images", err);
    }
  };

  const handleLoadGT = async () => {
    try {
      const { name, fileMap } = await pickFlatDirectory();
      const labelMap = await preloadLabels(fileMap);

      const collId = generateId();
      setCollections(prev => [...prev, { id: collId, name, type: 'labels', files: fileMap, labels: labelMap, count: Object.keys(fileMap).length }]);
      updateProject({ gtCollectionId: collId, gtPath: name });
      await mergeIndexedDBLabels(collId);
    } catch (err) {
      console.error("Failed to load GT", err);
    }
  };

  const handleLoadPred = async () => {
    try {
      const { name, fileMap } = await pickFlatDirectory();
      const labelMap = await preloadLabels(fileMap);

      const color = '#000000';

      const newSource: PredictionSource = {
        id: generateId(),
        name,
        path: name,
        color: color,
        visible: true,
        labels: labelMap
      };

      updateProject({
        predictionSources: [...(activeProject.predictionSources || []), newSource]
      });
    } catch (err) {
      console.error("Failed to load predictions", err);
    }
  };

  const handleTogglePredictionVisibility = (id: string, visible: boolean) => {
    updateProject({
      predictionSources: (activeProject.predictionSources || []).map((s: PredictionSource) => s.id === id ? { ...s, visible } : s)
    });
  };

  const handleUpdatePredictionColor = (id: string, color: string) => {
    updateProject({
      predictionSources: (activeProject.predictionSources || []).map((s: PredictionSource) => s.id === id ? { ...s, color } : s)
    });
  };

  const handleDeletePrediction = (id: string) => {
    updateProject({
      predictionSources: (activeProject.predictionSources || []).filter((s: PredictionSource) => s.id !== id)
    });
    setIsolatedPredictionIds(prev => prev.filter(pId => pId !== id));
  };

  const handleReorderPredictions = (newSources: PredictionSource[]) => {
    updateProject({ predictionSources: newSources });
  };

  const handleToggleGroupVisibility = (groupId: string, visible: boolean) => {
    updateProject({
      predictionSources: (activeProject.predictionSources || []).map((s: PredictionSource) =>
        s.groupId === groupId ? { ...s, visible } : s
      )
    });
  };

  const handleToggleAllPredictionsVisibility = (visible: boolean) => {
    updateProject({
      predictionSources: (activeProject.predictionSources || []).map((s: PredictionSource) => ({ ...s, visible }))
    });
  };

  const ensureGtCollection = (): string | null => {
    if (!activeProject.imageCollectionId) return null;
    if (activeProject.gtCollectionId) return activeProject.gtCollectionId;

    const collId = generateId();
    const name = activeProject.imagePath ? `${activeProject.imagePath}/labels` : 'New Labels';
    setCollections(prev => [...prev, { id: collId, name, type: 'labels', files: {}, labels: {}, count: 0 }]);
    updateProject({ gtCollectionId: collId, gtPath: name });
    return collId;
  };

  const handleToggleEditMode = async (enabled: boolean) => {
    if (enabled) {
      const gtId = ensureGtCollection();
      if (!gtId) return;
      await mergeIndexedDBLabels(gtId);
    }
    setIsEditMode(enabled);
  };

  const handleUpdateLabels = (fileName: string, newBoxes: BoundingBox[]) => {
    const gtId = ensureGtCollection();
    if (!gtId) return;

    const baseName = fileName.substring(0, fileName.lastIndexOf('.'));
    const txtName = `${baseName}.txt`;

    setModifiedFiles(prev => new Set(prev).add(txtName));
    setAllSessionModifiedFiles(prev => new Set(prev).add(txtName));
    setSavedFiles(prev => {
      const next = new Set(prev);
      next.delete(txtName);
      return next;
    });

    saveLabelToDB(txtName, newBoxes).catch(e => console.error("Failed to save to IndexedDB", e));

    setCollections(prev => prev.map(c => {
      if (c.id === gtId) {
        return {
          ...c,
          labels: {
            ...c.labels,
            [txtName]: newBoxes
          }
        };
      }
      return c;
    }));
  };

  const handleRecoverOriginalGt = async (fileName: string) => {
    if (!activeProject.gtCollectionId) return;
    const baseName = fileName.substring(0, fileName.lastIndexOf('.'));
    const txtName = `${baseName}.txt`;
    const collection = collections.find(c => c.id === activeProject.gtCollectionId);
    if (!collection || !collection.files[txtName]) return;

    try {
      const file = collection.files[txtName];
      const originalBoxes = await parseYoloFile(file);
      handleUpdateLabels(fileName, originalBoxes);

      setModifiedFiles(prev => {
        const next = new Set(prev);
        next.delete(txtName);
        return next;
      });
    } catch (e) {
      console.error("Failed to recover original GT", e);
    }
  };

  const handleExportLabels = async (asZip: boolean = false) => {
    const filesToExport = Array.from(modifiedFiles).filter(f => !savedFiles.has(f));
    const zipExportFiles = Array.from(allSessionModifiedFiles);

    if (!activeProject.gtCollectionId || (asZip ? zipExportFiles.length === 0 : filesToExport.length === 0)) {
      alert("No new modifications to export.");
      return;
    }
    const collection = collections.find(c => c.id === activeProject.gtCollectionId);
    if (!collection || !collection.labels) return;

    setIsExporting(true);
    const targetFiles = asZip ? zipExportFiles : filesToExport;
    setExportProgress({ current: 0, total: targetFiles.length });

    try {
      const filteredLabels: LabelMap = {};
      targetFiles.forEach((name: string) => {
        if (collection.labels && collection.labels[name]) {
          filteredLabels[name] = collection.labels[name];
        }
      });

      if (asZip) {
        await exportLabelsAsZip(filteredLabels, (current, total) => setExportProgress({ current, total }));
      } else {
        await exportLabels(filteredLabels, (current, total) => setExportProgress({ current, total }));

        await Promise.all(filesToExport.map(f => deleteLabelFromDB(f as string)));

        setSavedFiles(prev => {
          const next = new Set(prev);
          filesToExport.forEach(f => next.add(f));
          return next;
        });
        setModifiedFiles(prev => {
          const next = new Set(prev);
          filesToExport.forEach(f => next.delete(f));
          return next;
        });
      }
    } catch (err) {
      console.error("Export failed", err);
      if (!asZip) {
        alert("Export failed or was cancelled.");
      }
    } finally {
      setIsExporting(false);
      setExportProgress(null);
    }
  };

  const handleLoadAudio = async () => {
    try {
      const { name, fileMap } = await pickFlatDirectory(isAudioFileName);
      setAudioFileMap(fileMap);
      updateProject({ audioPath: name });
    } catch (err) {
      console.error("Failed to load audio folder", err);
    }
  };

  const handleImportFolder = async () => {
    try {
      const {
        name,
        imgMap,
        gtMap,
        predMap,
        predSubfolders,
        audioMap,
        hasImages,
        hasGt,
        hasPred,
        hasAudio,
      } = await pickDatasetDirectory();

      if (hasImages) {
        const collId = generateId();
        setCollections(prev => [...prev, { id: collId, name: 'Images', type: 'images', files: imgMap, count: Object.keys(imgMap).length }]);
        updateProject({ imageCollectionId: collId, imagePath: `${name}/images` });
        setCurrentPage(0);
      }

      let gtCollId: string | null = null;
      if (hasGt) {
        const labelMap = await preloadLabels(gtMap);
        gtCollId = generateId();
        setCollections(prev => [...prev, { id: gtCollId, name: 'Labels', type: 'labels', files: gtMap, labels: labelMap, count: Object.keys(gtMap).length }]);
        updateProject({ gtCollectionId: gtCollId, gtPath: `${name}/labels` });
      }

      if (hasPred) {
        const baseColors = ['#000000'];
        let colorIdx = activeProject.predictionSources?.length || 0;
        let newSources: PredictionSource[] = [];

        const getNextColor = () => baseColors[colorIdx++ % baseColors.length];

        const adjustColor = (hex: string, percent: number) => {
          const num = parseInt(hex.replace('#', ''), 16),
            amt = Math.round(2.55 * percent),
            R = (num >> 16) + amt,
            G = (num >> 8 & 0x00FF) + amt,
            B = (num & 0x0000FF) + amt;
          return '#' + (0x1000000 + (R < 255 ? R < 0 ? 0 : R : 255) * 0x10000 + (G < 255 ? G < 0 ? 0 : G : 255) * 0x100 + (B < 255 ? B < 0 ? 0 : B : 255)).toString(16).slice(1);
        };

        if (Object.keys(predMap).length > 0) {
          const labelMap = await preloadLabels(predMap);
          newSources.push({
            id: generateId(),
            name: 'Predictions',
            path: `${name}/predictions`,
            color: getNextColor(),
            visible: true,
            labels: labelMap
          });
        }

        for (const group of predSubfolders) {
          const groupColor = getNextColor();

          if (Object.keys(group.map).length > 0) {
            const labelMap = await preloadLabels(group.map);
            newSources.push({
              id: generateId(),
              name: group.name,
              path: `${name}/predictions/${group.name}`,
              color: groupColor,
              visible: true,
              labels: labelMap,
              groupId: group.subfolders && group.subfolders.length > 0 ? group.name : undefined
            });
          }

          if (group.subfolders) {
            let subIdx = 0;
            for (const sub of group.subfolders) {
              if (Object.keys(sub.map).length > 0) {
                const labelMap = await preloadLabels(sub.map);
                newSources.push({
                  id: generateId(),
                  name: sub.name,
                  path: `${name}/predictions/${group.name}/${sub.name}`,
                  color: adjustColor(groupColor, (subIdx + 1) * -15),
                  visible: true,
                  labels: labelMap,
                  groupId: group.name
                });
                subIdx++;
              }
            }
          }
        }

        if (newSources.length > 0) {
          updateProject({ predictionSources: [...(activeProject.predictionSources || []), ...newSources] });
        }
      }

      if (hasAudio) {
        setAudioFileMap(audioMap);
        updateProject({ audioPath: `${name}/audio` });
      }

      if (gtCollId) {
        await mergeIndexedDBLabels(gtCollId);
      } else {
        await mergeIndexedDBLabels();
      }
    } catch (err) {
      console.error("Failed to import folder", err);
    }
  };

  const items: ImageItem[] = useMemo(() => {
    const imageNames = Object.keys(imageFiles).sort();
    return imageNames.map(imgName => {
      const baseName = imgName.substring(0, imgName.lastIndexOf('.'));
      const txtName = `${baseName}.txt`;

      const predictions = (activeProject.predictionSources || [])
        .filter(src => {
          if (isolatedPredictionIds.length > 0) {
            return isolatedPredictionIds.includes(src.id);
          }
          return true;
        })
        .map((src: PredictionSource) => ({
          sourceId: src.id,
          boxes: src.labels[txtName] || [],
          color: src.color,
          visible: src.visible
        }));

      return {
        name: imgName,
        file: imageFiles[imgName],
        gtData: gtLabels[txtName],
        predictions: predictions,
        isModified: modifiedFiles.has(txtName),
        isSaved: savedFiles.has(txtName),
      };
    });
  }, [imageFiles, gtLabels, activeProject.predictionSources, isolatedPredictionIds, modifiedFiles, savedFiles]);

  const totalPages = Math.ceil(items.length / config.gridSize);
  const currentItems = useMemo(() => items.slice(
    currentPage * config.gridSize,
    (currentPage + 1) * config.gridSize
  ), [items, currentPage, config.gridSize]);

  useEffect(() => {
    setJumpPageInput((currentPage + 1).toString());
  }, [currentPage]);

  // Keep the same image in view when grid size changes.
  useEffect(() => {
    const prevGridSize = prevGridSizeRef.current;
    if (prevGridSize !== config.gridSize) {
      let absoluteIndex = 0;
      if (prevGridSize === 1) {
        absoluteIndex = currentPage;
      } else {
        absoluteIndex = currentPage * prevGridSize + focusedItemIndex;
      }

      let newPage = 0;
      if (config.gridSize === 1) {
        newPage = absoluteIndex;
        setFocusedItemIndex(0);
      } else {
        newPage = Math.floor(absoluteIndex / config.gridSize);
        setFocusedItemIndex(absoluteIndex % config.gridSize);
      }

      setCurrentPage(newPage);
      prevGridSizeRef.current = config.gridSize;
    }
  }, [config.gridSize, currentPage, focusedItemIndex]);

  const toggleFocusMode = (indexInPage: number) => {
    if (config.gridSize === 1) {
      const revertGrid = prevGridSizeRef.current === 1 ? 9 : prevGridSizeRef.current;
      setConfig({ ...config, gridSize: revertGrid });
    } else {
      if (indexInPage !== focusedItemIndex) {
        setFocusedItemIndex(indexInPage);
      }
      setConfig({ ...config, gridSize: 1 });
    }
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') return;
      if (currentItems.length === 0) return;

      const cols = config.gridSize === 9 ? 3 : (config.gridSize === 16 ? 4 : 1);

      if (e.key === 'ArrowRight') {
        setFocusedItemIndex(p => Math.min(p + 1, currentItems.length - 1));
      } else if (e.key === 'ArrowLeft') {
        setFocusedItemIndex(p => Math.max(p - 1, 0));
      } else if (e.key === 'ArrowDown') {
        setFocusedItemIndex(p => Math.min(p + cols, currentItems.length - 1));
      } else if (e.key === 'ArrowUp') {
        setFocusedItemIndex(p => Math.max(p - cols, 0));
      } else if (e.key === 'd' || e.key === 'D') {
        setCurrentPage(p => Math.min(p + 1, totalPages - 1));
      } else if (e.key === 'a' || e.key === 'A') {
        setCurrentPage(p => Math.max(p - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        toggleFocusMode(focusedItemIndex);
      } else if (e.key === 'e' || e.key === 'E') {
        e.preventDefault();
        handleToggleEditMode(!isEditMode);
      }

      if (['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'].includes(e.key)) {
        setTimeout(() => {
          document.querySelector('[data-focused="true"]')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 50);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [config.gridSize, currentItems.length, focusedItemIndex, currentPage, isEditMode]);

  useEffect(() => {
    if (modifiedFiles.size === 0) return;
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [modifiedFiles.size]);

  useEffect(() => {
    if (totalPages > 0 && currentPage >= totalPages) {
      setCurrentPage(totalPages - 1);
    } else if (totalPages === 0 && currentPage !== 0) {
      setCurrentPage(0);
    }
  }, [totalPages, currentPage]);

  useEffect(() => {
    if (currentItems.length === 0) {
      setPageStats({ tp: 0, fp: 0, fn: 0 });
      return;
    }

    const totals = { tp: 0, fp: 0, fn: 0 };
    currentItems.forEach((item) => {
      const gtBoxes = item.gtData || [];
      const visibleSources = item.predictions?.filter(p => p.visible) || [];
      const primaryPred = visibleSources.length > 0 ? visibleSources[0].boxes : [];
      const result = calculateMatches(gtBoxes, primaryPred, config);

      result.forEach(b => {
        if (b.type === BoxType.TP_PRED) totals.tp++;
        else if (b.type === BoxType.FP) totals.fp++;
        else if (b.type === BoxType.FN) totals.fn++;
      });
    });

    setPageStats(totals);
  }, [currentItems, config.matchOverlapMetric, config.matchOverlapThreshold, config.confThreshold, config.nmsIouThreshold]);


  const nextPage = () => setCurrentPage(p => Math.min(p + 1, totalPages - 1));
  const prevPage = () => setCurrentPage(p => Math.max(p - 1, 0));

  const randomPage = () => {
    if (totalPages <= 1) return;
    const rnd = Math.floor(Math.random() * totalPages);
    setCurrentPage(rnd);
  };

  const handlePageJump = (e: React.FormEvent) => {
    e.preventDefault();
    const pageNum = parseInt(jumpPageInput, 10);
    if (!isNaN(pageNum) && pageNum >= 1 && pageNum <= totalPages) {
      setCurrentPage(pageNum - 1);
    } else {
      setJumpPageInput((currentPage + 1).toString());
    }
  };

  const handleDownloadPage = async () => {
    if (currentItems.length === 0) return;
    setIsDownloading(true);

    try {
      const loadedData = await Promise.all(currentItems.map(async (item) => {
        const file = item.file;
        const url = file instanceof File ? URL.createObjectURL(file) : URL.createObjectURL(await (file as FileSystemFileHandle).getFile());
        const img = new Image();
        img.src = url;
        await new Promise((resolve) => {
          img.onload = resolve;
          img.onerror = resolve;
        });
        return { item, img, url };
      }));

      const cols = config.gridSize === 9 ? 3 : config.gridSize === 16 ? 4 : 1;
      const isSingleImageMode = config.gridSize === 1;
      const targetCellWidth = 1600;
      const gap = isSingleImageMode ? 0 : 80;

      const rows: typeof loadedData[] = [];
      let currentRow: typeof loadedData = [];

      for (const data of loadedData) {
        currentRow.push(data);
        if (currentRow.length === cols) {
          rows.push(currentRow);
          currentRow = [];
        }
      }
      if (currentRow.length > 0) rows.push(currentRow);

      let totalHeight = 0;
      const rowConfigs = rows.map(row => {
        const processedItems = row.map(data => {
          if (isSingleImageMode) {
            const width = data.img.naturalWidth || targetCellWidth;
            const height = data.img.naturalHeight || Math.round(width / 1.77);
            return { ...data, width, height };
          }

          const aspect = data.img.naturalWidth ? (data.img.naturalWidth / data.img.naturalHeight) : 1.77;
          const height = Math.round(targetCellWidth / aspect);
          return { ...data, width: targetCellWidth, height };
        });
        const rowHeight = Math.max(...processedItems.map(i => i.height));
        const y = totalHeight;
        totalHeight += rowHeight + gap;
        return { items: processedItems, rowHeight, y };
      });
      if (rowConfigs.length > 0) totalHeight -= gap;

      const totalWidth = rowConfigs.length > 0
        ? Math.max(...rowConfigs.map(row =>
          row.items.reduce((acc, item, idx) => acc + item.width + (idx > 0 ? gap : 0), 0)
        ))
        : 0;

      const canvas = document.createElement('canvas');
      canvas.width = totalWidth;
      canvas.height = totalHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Could not create canvas context');

      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (const rowConfig of rowConfigs) {
        for (let i = 0; i < rowConfig.items.length; i++) {
          const { item, img, width, height } = rowConfig.items[i];
          const x = i * (targetCellWidth + gap);
          const y = rowConfig.y + (rowConfig.rowHeight - height) / 2;

          ctx.save();
          ctx.translate(x, y);
          ctx.beginPath();
          ctx.rect(0, 0, width, height);
          ctx.clip();

          const scaleFactor = width / (img.naturalWidth || width);
          const currentGt = item.gtData || [];
          const visibleSources = item.predictions?.filter(p => p.visible) || [];
          const shouldShowPreds = config.showPredictions !== false;
          let renderBoxes = [];

          if (shouldShowPreds && visibleSources.length > 0) {
            const primaryMatches = calculateMatches(currentGt, visibleSources[0].boxes, config);
            primaryMatches.forEach(m => {
              if (m.type === BoxType.TP_PRED || m.type === BoxType.FP) {
                m.color = visibleSources[0].color;
              }
            });
            renderBoxes.push(...primaryMatches);

            for (let srcIdx = 1; srcIdx < visibleSources.length; srcIdx++) {
              const src = visibleSources[srcIdx];
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

          await drawVisualization(ctx, item, config, width, height, img, {
            fontSize: Math.round(config.labelFontSize * scaleFactor),
            forceLineWidth: Math.max(1, Math.round(config.lineWidth * scaleFactor)),
            preCalculatedBoxes: renderBoxes
          });

          ctx.restore();
        }
      }

      const link = document.createElement('a');
      link.download = `page_${currentPage + 1}_${activeProject.name}.jpg`;
      link.href = canvas.toDataURL('image/jpeg', 0.92);
      link.click();

      loadedData.forEach(d => URL.revokeObjectURL(d.url));

    } catch (e) {
      console.error("Failed to generate download", e);
      alert("Failed to generate download image");
    } finally {
      setIsDownloading(false);
    }
  };

  const gridClass = config.gridSize === 1
    ? "grid-cols-1 justify-items-center max-w-5xl mx-auto"
    : config.gridSize === 9
      ? "grid-cols-1 md:grid-cols-2 xl:grid-cols-3"
      : "grid-cols-2 md:grid-cols-3 xl:grid-cols-4";

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const newWidth = document.body.clientWidth - e.clientX;
      setSidebarWidth(Math.max(300, Math.min(newWidth, 800)));
    };

    const handleMouseUp = () => {
      isDragging.current = false;
      document.body.style.cursor = 'default';
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      <ControlPanel
        projects={projects}
        activeProjectId={activeProjectId}
        onImportImages={handleLoadImages}
        onImportGT={handleLoadGT}
        onLoadPred={handleLoadPred}
        predictionSources={activeProject.predictionSources || []}
        onTogglePredictionVisibility={handleTogglePredictionVisibility}
        onUpdatePredictionColor={handleUpdatePredictionColor}
        onDeletePrediction={handleDeletePrediction}
        onReorderPredictions={handleReorderPredictions}
        onSetIsolatedPredictions={setIsolatedPredictionIds}
        onToggleGroupVisibility={handleToggleGroupVisibility}
        onToggleAllPredictionsVisibility={handleToggleAllPredictionsVisibility}
        config={config}
        onConfigChange={setConfig}
        stats={{
          totalImages: items.length,
          hasGt: Object.keys(gtLabels).length > 0,
          canAnnotate: items.length > 0,
          imagePath: activeProject.imagePath,
          gtPath: activeProject.gtPath,
          audioPath: activeProject.audioPath
        }}
        isEditMode={isEditMode}
        onToggleEditMode={handleToggleEditMode}
        onExportLabels={handleExportLabels}
        onLoadAudio={handleLoadAudio}
        onImportFolder={handleImportFolder}
        hasAudio={Object.keys(audioFileMap).length > 0}
        isExporting={isExporting}
        exportProgress={exportProgress}
      />

      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <div className="h-16 border-b border-slate-700 flex items-center justify-between px-6 bg-surface shadow-sm z-10 flex-shrink-0">
          <div className="text-slate-300 text-sm flex items-center gap-4 flex-1 min-w-0">
            <form onSubmit={handlePageJump} className="flex items-center gap-2 text-slate-400 flex-shrink-0">
              <span>Page</span>
              <input
                type="number"
                value={jumpPageInput}
                onChange={(e) => setJumpPageInput(e.target.value)}
                onBlur={() => handlePageJump({ preventDefault: () => { } } as any)}
                className="w-16 bg-slate-800 border border-slate-700 rounded text-center text-white focus:outline-none focus:border-primary text-sm py-1"
              />
              <span>of <span className="text-white font-bold">{totalPages || 1}</span></span>
            </form>
          </div>

          <div className="flex items-center justify-center gap-6 flex-shrink-0 mx-4">
            {items.length > 0 && (
              <>
                <div className="flex items-center gap-4 bg-slate-800/80 px-5 py-2 rounded-full border border-slate-700 shadow-sm">
                  <div
                    onMouseEnter={() => !lockedHighlight && setGlobalHighlight(BoxType.TP_PRED)}
                    onMouseLeave={() => !lockedHighlight && setGlobalHighlight(null)}
                    onClick={() => {
                      if (lockedHighlight === BoxType.TP_PRED) {
                        setLockedHighlight(null);
                        setGlobalHighlight(null);
                      } else {
                        setLockedHighlight(BoxType.TP_PRED);
                        setGlobalHighlight(BoxType.TP_PRED);
                      }
                    }}
                    className={`cursor-pointer px-3 py-0.5 rounded transition-all select-none ${lockedHighlight === BoxType.TP_PRED ? 'bg-white/20 ring-1 ring-white/50' : 'hover:bg-white/10'}`}
                    style={{ color: config.styles.tpGt.color }}
                  >
                    <span className="font-bold mr-1">TP:</span>{pageStats.tp}
                  </div>
                  <div className="w-px h-4 bg-slate-600"></div>
                  <div
                    onMouseEnter={() => !lockedHighlight && setGlobalHighlight(BoxType.FN)}
                    onMouseLeave={() => !lockedHighlight && setGlobalHighlight(null)}
                    onClick={() => {
                      if (lockedHighlight === BoxType.FN) {
                        setLockedHighlight(null);
                        setGlobalHighlight(null);
                      } else {
                        setLockedHighlight(BoxType.FN);
                        setGlobalHighlight(BoxType.FN);
                      }
                    }}
                    className={`cursor-pointer px-3 py-0.5 rounded transition-all select-none ${lockedHighlight === BoxType.FN ? 'bg-white/20 ring-1 ring-white/50' : 'hover:bg-white/10'}`}
                    style={{ color: config.styles.fn.color }}
                  >
                    <span className="font-bold mr-1">FN:</span>{pageStats.fn}
                  </div>
                  <div className="w-px h-4 bg-slate-600"></div>
                  <div
                    onMouseEnter={() => !lockedHighlight && setGlobalHighlight(BoxType.FP)}
                    onMouseLeave={() => !lockedHighlight && setGlobalHighlight(null)}
                    onClick={() => {
                      if (lockedHighlight === BoxType.FP) {
                        setLockedHighlight(null);
                        setGlobalHighlight(null);
                      } else {
                        setLockedHighlight(BoxType.FP);
                        setGlobalHighlight(BoxType.FP);
                      }
                    }}
                    className={`cursor-pointer px-3 py-0.5 rounded transition-all select-none ${lockedHighlight === BoxType.FP ? 'bg-white/20 ring-1 ring-white/50' : 'hover:bg-white/10'}`}
                    style={{ color: config.styles.fp.color }}
                  >
                    <span className="font-bold mr-1">FP:</span>{pageStats.fp}
                  </div>
                </div>

                <div className="flex items-center gap-2 bg-slate-800/50 px-3 py-2 rounded-full border border-slate-700">
                  <button onClick={() => setConfig({ ...config, zoomLevel: Math.max(0.5, config.zoomLevel - 0.1) })} className="p-1 hover:bg-slate-700 rounded-full text-slate-400 hover:text-white transition-colors"><ZoomOut className="w-3.5 h-3.5" /></button>
                  <input
                    type="range"
                    min="0.5" max="3" step="0.1"
                    value={config.zoomLevel}
                    onChange={(e) => setConfig({ ...config, zoomLevel: parseFloat(e.target.value) })}
                    className="w-24 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary"
                  />
                  <button onClick={() => setConfig({ ...config, zoomLevel: Math.min(3, config.zoomLevel + 0.1) })} className="p-1 hover:bg-slate-700 rounded-full text-slate-400 hover:text-white transition-colors"><ZoomIn className="w-3.5 h-3.5" /></button>
                </div>

                <div className="flex items-center gap-2 bg-slate-800/50 px-3 py-1.5 rounded-full border border-slate-700">
                  <span className="text-xs text-slate-400 font-semibold mr-1">Speed</span>
                  <select
                    className="bg-transparent text-xs text-white outline-none cursor-pointer appearance-none font-bold select-none pr-1"
                    value={config.audio?.playbackSpeed ?? 1}
                    onChange={(e) => setConfig({ ...config, audio: { ...config.audio, playbackSpeed: parseFloat(e.target.value) } })}
                  >
                    <option value="0.5" className="bg-slate-800">0.5x</option>
                    <option value="0.75" className="bg-slate-800">0.75x</option>
                    <option value="1" className="bg-slate-800">1x</option>
                    <option value="1.25" className="bg-slate-800">1.25x</option>
                    <option value="1.5" className="bg-slate-800">1.5x</option>
                  </select>
                </div>
              </>
            )}
          </div>

          <div className="flex justify-end gap-4 flex-1 min-w-0">
            <>
              {items.length > 0 && (
                  <button
                    onClick={handleDownloadPage}
                    disabled={isDownloading}
                    className="bg-primary hover:bg-blue-600 text-white p-2 rounded flex items-center justify-center disabled:opacity-50 transition-colors shadow-lg shadow-blue-900/20"
                    title="Download View"
                  >
                    {isDownloading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5" />}
                  </button>
                )}

                <div className="flex gap-2">
                  <button
                    onClick={prevPage}
                    disabled={currentPage === 0}
                    className="p-2 rounded hover:bg-slate-700 disabled:opacity-30 disabled:hover:bg-transparent text-slate-200"
                    title="Previous Page"
                  >
                    <ChevronLeft className="w-5 h-5" />
                  </button>

                  <button
                    onClick={randomPage}
                    disabled={totalPages <= 1}
                    className="p-2 rounded hover:bg-slate-700 disabled:opacity-30 disabled:hover:bg-transparent text-slate-200"
                    title="Random Page"
                  >
                    <Shuffle className="w-5 h-5" />
                  </button>

                  <button
                    onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                    className={`p-2 rounded transition-colors ${isSidebarOpen ? 'bg-primary/20 text-primary' : 'hover:bg-slate-700 text-slate-200'}`}
                    title={isSidebarOpen ? "Hide Sidebar" : "Show Sidebar"}
                  >
                    <PanelRight className="w-5 h-5" />
                  </button>

                  <button
                    onClick={nextPage}
                    disabled={currentPage >= totalPages - 1}
                    className="p-2 rounded hover:bg-slate-700 disabled:opacity-30 disabled:hover:bg-transparent text-slate-200"
                    title="Next Page"
                  >
                    <ChevronRight className="w-5 h-5" />
                  </button>
                </div>
            </>
          </div>
        </div>

        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 overflow-hidden relative flex flex-col">
            <div className="w-full h-full overflow-auto custom-scrollbar p-6">
              {items.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-500">
                  <Inbox className="w-12 h-12 mb-4 opacity-50" />
                  <p>No Images Selected</p>
                </div>
              ) : (
                <div
                  className={`grid ${gridClass} gap-4 pb-10 origin-top-left transition-all duration-200 ease-out`}
                  style={{ width: `${config.zoomLevel * 100}%` }}
                >
                  {currentItems.map((item, idx) => (
                    <ImageViewer
                      key={item.name}
                      item={item}
                      config={config}
                      externalHighlight={globalHighlight}
                      isEditMode={isEditMode}
                      onUpdateGt={handleUpdateLabels}
                      audioPlayer={audioPlayer}
                      audioFiles={audioFileMap}
                      activePlayback={activePlayback}
                      onSetGlobalPlayback={setActivePlayback}
                      isFocused={focusedItemIndex === idx}
                      onFocusToggle={() => toggleFocusMode(idx)}
                      onSetFocus={() => setFocusedItemIndex(idx)}
                      onRecoverOriginalGt={handleRecoverOriginalGt}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>

          {isSidebarOpen && (
            <>
              <div
                className="w-1 bg-slate-800 hover:bg-primary cursor-col-resize z-20 flex-shrink-0 transition-colors border-l border-slate-700"
                onMouseDown={(e) => {
                  isDragging.current = true;
                  document.body.style.cursor = 'col-resize';
                  e.preventDefault();
                }}
              />
              <div
                style={{ width: sidebarWidth }}
                className="flex-shrink-0 bg-surface/30 border-l border-slate-700 overflow-hidden shadow-xl z-10 flex flex-col"
              >
                <PRGraph items={items} config={config} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;