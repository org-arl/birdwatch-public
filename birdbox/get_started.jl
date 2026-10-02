## ── 0. First time setup ────────────────────────────────────────────────────────────
# First-time only: run `julia --project=. birdbox/setup.jl` from a terminal to create
# conda environments and install Python and Julia dependencies.


cd(@__DIR__)
using Pkg
Pkg.activate(".")
using BirdBox

device = "cpu" # set to "0" to use GPU if available
ENV["RFDETR_PYTHON"] = strip(read(`conda run -n birdbox-rfdetr which python`, String))

# Paths to example recording and pre-trained models
recording = joinpath(@__DIR__, "examples", "example.wav")
yolo_path = joinpath(@__DIR__, "models", "yolo11n.pt")
rfdetr_path = joinpath(@__DIR__, "models", "rfdetr-nano.pth")

## ── 1. Make predictions ────────────────────────────────────────────────────────────
df = detect(recording, yolo_path; device, backend = :yolo);
df = detect(recording, rfdetr_path; device, backend = :rfdetr);

## ── 2. Annotate or analyze recordings ───────────────────────────────────────────────────

# 2.1 Open https://org-arl.github.io/birdwatch-public/
# 2.2 Load spectrogram images, bounding box labels and recordings (optional)
# 2.3 Annotate new recordings, edit existing annotations or analyze detections

## ── 3. Train custom YOLO model ────────────────────────────────────────────────────────────

# 3.1 Create dummy ground truth data by running inference on the recording
yolo_dir = "data/yolo"
detect(recording, yolo_path; device, savedir = yolo_dir, save_conf = false); # exclude confidence scores for training data

# 3.2 Split data into train, val and test sets
files = readdir(joinpath(yolo_dir, "images"), join = true)
write_split_file("train", files[1:2]; outdir = yolo_dir)
write_split_file("val", files[3:4]; outdir = yolo_dir)
write_split_file("test", files[5:end]; outdir = yolo_dir)

# 3.3 Create data.yaml
data_yaml = write_data_yaml(; path = joinpath(yolo_dir, "data.yaml"), root = yolo_dir)

# 3.4 Train model
train(yolo_path, data_yaml;
    single_cls = true,
    lr0 = 0.01,
    weight_decay = 0.0005,
    batch = 2,
    epochs = 2,
    patience = 5,
    device = device,
    name = "example_run_yolo",
    amp = false, # set true for mixed precision to speed up training on GPUs
);

## ── 4. RF-DETR ─────────────────────────────────────────────────────────

# Train from pre-trained checkpoint (uses the dummy data from section 3).
rfdetr_dir = "data/rfdetr"
write_padded_dataset(rfdetr_dir; train = joinpath(yolo_dir, "train.txt"), val = joinpath(yolo_dir, "val.txt"))
train(rfdetr_path, rfdetr_dir;
    backend = :rfdetr,
    batch = 2,
    epochs = 2,
    patience = 5,
    device = device,
    name = "example_run_rfdetr",
);
