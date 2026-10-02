"""
    BirdBox

Time-frequency localization of bird vocalizations in spectrograms using YOLO
or RF-DETR.

# Audio

* `load_audio(path; channel=1)` — load one channel from an audio file.
* `split_recording(samples, sr; ...)` — split a waveform into overlapping clips.

# Spectrogram pipeline

* `spectrogram(x, sr; ...)` — STFT log-magnitude spectrogram in `[0, 1]`.
* `spec2img(spec; ...)` — convert spectrogram to 3-channel RGB matrix via colormap.
* `write_spectrogram_images(recording; ...)` — split a recording into clips and save spectrogram images.

# Detection

* `detect(model_path; backend=:yolo, ...)` — run a detector on spectrogram images in `imgdir`.
* `detect(recording, model_path; backend=:yolo, ...)` — split a recording into clips, generate spectrogram images, then run the detector.

# Labels

* `read_yolo_labels(sourcedir)` — parse YOLO `.txt` label files into a `DataFrame`.
* `add_timefreq_columns(df, fmin, fmax, tstart, tend)` — append `(t0, t1, f0, f1)` columns to a label `DataFrame`.

# Training data prep / training

* `write_split_file(split, files; outdir)` — write a `<split>.txt` list of spectrogram image paths.
* `write_data_yaml(; ...)` — write a YOLO `data.yaml`.
* `write_padded_dataset(outdir; train, val, ...)` — pad spectrograms and rewrite labels for RF-DETR.
* `train(model, data; backend=:yolo, ...)` — train/fine-tune YOLO or RF-DETR.
"""
module BirdBox

export load_audio, split_recording
export spectrogram, spec2img, write_spectrogram_images
export detect
export read_yolo_labels, add_timefreq_columns
export write_split_file, write_data_yaml, write_padded_dataset, train
export FMIN, FMAX, DURATION, OVERLAP, PRED_DIR, TRAIN_DIR, IMGSIZE, NMS_IOU, MINCONF

include("config.jl")
include("audio.jl")
include("spectrogram.jl")
include("labels.jl")
include("detect.jl")
include("data.jl")
include("train.jl")

end # module
