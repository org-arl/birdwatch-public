## ── 0. First time setup ────────────────────────────────────────────────────────────
# First-time only: run `julia --project=. birdbox/setup.jl` from a terminal to create the
# conda environment and install Python and Julia dependencies.


cd(@__DIR__)
using Pkg
Pkg.activate(".")

using BirdBox
device = "cpu" # set to "0" to use GPU if available

## ── 1. Make predictions ────────────────────────────────────────────────────────────

recording = joinpath(@__DIR__, "examples", "example.wav")
model_path = joinpath(@__DIR__, "models", "yolo11n.pt")

df = detect(recording, model_path; device);

## ── 2. Annotate or analyze recordings ───────────────────────────────────────────────────

# 2.1 Open https://yolo-vislab.vercel.app/
# 2.2 Load recording(s), spectrogram images and YOLO labels
# 2.3 Annotate new recordings, edit existing annotations or analyze detections

## ── 3. Train custom YOLO model ────────────────────────────────────────────────────────────

# 3.1 Create dummy ground truth data by running inference on the recording
savedir = joinpath(@__DIR__, "data", "example_data")
detect(recording, model_path; device, savedir, save_conf = false); # exclude confidence scores for training data

# 3.2 Split data into train, val and test sets
files = readdir(joinpath(savedir, "images"), join = true)
write_split_file("train", files[1:2])
write_split_file("val", files[3:4])
write_split_file("test", files[5:end])

# 3.3 Create data.yaml
data_yaml = write_data_yaml()

# 3.4 Train model
train(model_path, data_yaml;
    single_cls = true,
    lr0 = 0.01,
    weight_decay = 0.0005,
    batch = 2,
    epochs = 2,
    patience = 5,
    device = device,
    name = "example_run",
    amp = false, # set true for mixed precision to speed up training on GPUs
);