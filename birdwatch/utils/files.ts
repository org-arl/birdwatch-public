import { FileMap } from '../types';

const AUDIO_EXTENSIONS = new Set(['wav', 'mp3', 'ogg', 'm4a']);

async function ensureReadPermission(dirHandle: FileSystemDirectoryHandle): Promise<void> {
  const opts: FileSystemHandlePermissionDescriptor = { mode: 'read' };
  if (await dirHandle.queryPermission(opts) !== 'granted') {
    const result = await dirHandle.requestPermission(opts);
    if (result !== 'granted') {
      throw new DOMException('Read permission denied.', 'NotAllowedError');
    }
  }
}

async function readFlatDirectory(
  dirHandle: FileSystemDirectoryHandle,
  filter?: (name: string) => boolean
): Promise<FileMap> {
  await ensureReadPermission(dirHandle);
  const fileMap: FileMap = {};

  for await (const entry of dirHandle.values()) {
    if (entry.kind === 'file' && !entry.name.startsWith('.')) {
      if (filter && !filter(entry.name)) continue;
      fileMap[entry.name] = await entry.getFile();
    }
  }

  return fileMap;
}

/** Pick a folder via the File System Access API and read its files. */
export async function pickFlatDirectory(
  filter?: (name: string) => boolean
): Promise<{ name: string; fileMap: FileMap }> {
  const dirHandle = await window.showDirectoryPicker();
  const fileMap = await readFlatDirectory(dirHandle, filter);
  return { name: dirHandle.name, fileMap };
}

export function isAudioFile(file: File): boolean {
  const ext = file.name.split('.').pop()?.toLowerCase();
  return !!ext && AUDIO_EXTENSIONS.has(ext);
}

export function isAudioFileName(name: string): boolean {
  const ext = name.split('.').pop()?.toLowerCase();
  return !!ext && AUDIO_EXTENSIONS.has(ext);
}

export interface DatasetImportMaps {
  imgMap: FileMap;
  gtMap: FileMap;
  predMap: FileMap;
  predSubfolders: { name: string; map: FileMap; subfolders?: { name: string; map: FileMap }[] }[];
  audioMap: FileMap;
  hasImages: boolean;
  hasGt: boolean;
  hasPred: boolean;
  hasAudio: boolean;
}

/** Pick a dataset root (images/, labels/, predictions/, audio/) via the File System Access API. */
export async function pickDatasetDirectory(): Promise<{ name: string } & DatasetImportMaps> {
  const dirHandle = await window.showDirectoryPicker();
  await ensureReadPermission(dirHandle);

  let imgMap: FileMap = {};
  let gtMap: FileMap = {};
  let predMap: FileMap = {};
  let predSubfolders: DatasetImportMaps['predSubfolders'] = [];
  let audioMap: FileMap = {};

  let hasImages = false;
  let hasGt = false;
  let hasPred = false;
  let hasAudio = false;

  for await (const entry of dirHandle.values()) {
    if (entry.kind !== 'directory') continue;

    if (entry.name === 'images') {
      hasImages = true;
      imgMap = await readFlatDirectory(entry);
    } else if (entry.name === 'labels') {
      hasGt = true;
      gtMap = await readFlatDirectory(entry);
    } else if (entry.name === 'predictions') {
      hasPred = true;
      for await (const subEntry of entry.values()) {
        if (subEntry.kind === 'directory') {
          const subMap: FileMap = {};
          const subfolders: { name: string; map: FileMap }[] = [];

          for await (const nestedEntry of subEntry.values()) {
            if (nestedEntry.kind === 'directory') {
              subfolders.push({
                name: nestedEntry.name,
                map: await readFlatDirectory(nestedEntry),
              });
            } else if (nestedEntry.kind === 'file' && !nestedEntry.name.startsWith('.')) {
              subMap[nestedEntry.name] = await nestedEntry.getFile();
            }
          }

          predSubfolders.push({ name: subEntry.name, map: subMap, subfolders });
        } else if (subEntry.kind === 'file' && !subEntry.name.startsWith('.')) {
          predMap[subEntry.name] = await subEntry.getFile();
        }
      }
    } else if (entry.name === 'audio') {
      hasAudio = true;
      audioMap = await readFlatDirectory(entry, isAudioFileName);
    }
  }

  return {
    name: dirHandle.name,
    imgMap,
    gtMap,
    predMap,
    predSubfolders,
    audioMap,
    hasImages,
    hasGt,
    hasPred,
    hasAudio,
  };
}
