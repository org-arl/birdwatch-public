"""
    BirdBox

Time-frequency localization of bird vocalizations in spectrograms using YOLO.

# Audio

* `load_audio(path; channel=1)` — load one channel from an audio file.
* `split_recording(samples, sr; ...)` — split a waveform into overlapping clips.

# Spectrogram pipeline

* `spectrogram(x, sr; ...)` — STFT log-magnitude spectrogram in `[0, 1]`.
* `spec2img(spec; ...)` — convert spectrogram to 3-channel RGB matrix via colormap.
* `write_spectrogram_images(recording; ...)` — split a recording into clips and save spectrogram images.

# Detection

* `detect(model_path; ...)` — run YOLO on spectrogram image in `imgdir`.
* `detect(recording, model_path; ...)` — split recording into clips, generate spectrogram images, then run YOLO model.

# Labels

* `read_yolo_labels(sourcedir)` — parse YOLO `.txt` label files into a `DataFrame`.
* `add_timefreq_columns(df, fmin, fmax, tstart, tend)` — append `(t0, t1, f0, f1)` columns to a label `DataFrame`.

# Training data prep / training

* `write_split_file(split, files; outdir)` — write a `<split>.txt` list of spectrogram image paths.
* `write_data_yaml(path; ...)` — write a YOLO `data.yaml`.
* `train(model_path, data_yaml; ...)` — train/fine-tune YOLO via Ultralytics.
"""
module BirdBox

export load_audio, split_recording
export spectrogram, spec2img, write_spectrogram_images
export detect
export read_yolo_labels, add_timefreq_columns
export write_split_file, write_data_yaml, train
export FMIN, FMAX, DURATION, OVERLAP, PRED_DIR, TRAIN_DIR, IMGSIZE, NMS_IOU, MINCONF

include("config.jl")
include("audio.jl")
include("spectrogram.jl")
include("labels.jl")
include("detect.jl")
include("data.jl")
include("train.jl")

end # module
