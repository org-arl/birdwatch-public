import { LabelMap } from '../types';

import JSZip from 'jszip';

async function ensureWritePermission(dirHandle: FileSystemDirectoryHandle): Promise<void> {
    const opts: FileSystemHandlePermissionDescriptor = { mode: 'readwrite' };
    if (await dirHandle.queryPermission(opts) !== 'granted') {
        const result = await dirHandle.requestPermission(opts);
        if (result !== 'granted') {
            throw new DOMException('Write permission denied.', 'NotAllowedError');
        }
    }
}

function formatLabelContent(boxes: LabelMap[string]): string {
    return boxes.map(box =>
        `${box.classId} ${box.x.toFixed(6)} ${box.y.toFixed(6)} ${box.w.toFixed(6)} ${box.h.toFixed(6)}`
    ).join('\n');
}

/** Write modified labels to a user-chosen folder (YOLO format: class cx cy w h). */
export const exportLabels = async (labelMap: LabelMap, onProgress?: (current: number, total: number) => void) => {
    const dirHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
    await ensureWritePermission(dirHandle);

    const entries = Object.entries(labelMap);
    const total = entries.length;
    let current = 0;

    if (onProgress) onProgress(current, total);

    for (const [filename, boxes] of entries) {
        const fileHandle = await dirHandle.getFileHandle(filename, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(formatLabelContent(boxes));
        await writable.close();

        current++;
        if (onProgress) onProgress(current, total);
    }
};

export const exportLabelsAsZip = async (labelMap: LabelMap, onProgress?: (current: number, total: number) => void) => {
    try {
        const zip = new JSZip();
        const entries = Object.entries(labelMap);
        const total = entries.length;
        let current = 0;

        if (onProgress) onProgress(current, total);

        for (const [filename, boxes] of entries) {
            zip.file(filename, formatLabelContent(boxes));

            current++;
            if (onProgress) onProgress(current, total);
        }

        const blob = await zip.generateAsync({ type: "blob" });

        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `birdwatch_labels_export_${new Date().toISOString().replace(/[:.]/g, '-')}.zip`;
        document.body.appendChild(a);
        a.click();

        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);

    } catch (err) {
        console.error("Zip Export failed:", err);
        throw err;
    }
};
