# First-time setup for BirdBox. Run as a script with `julia --project=. setup.jl`,
# or step through it line-by-line in a Julia REPL.

cd(@__DIR__)
using Pkg
using Downloads
Pkg.activate(".")

const MODEL_RELEASE = "https://github.com/org-arl/birdwatch-public/releases/download/models-v1"
const MODEL_FILES = ("yolo11n.pt", "yolo11l.pt", "rfdetr-nano.pth", "rfdetr-large.pth")

# Install Julia dependencies
Pkg.instantiate()

# ==============================================
# Set up environment for YOLO backend 
# ==============================================

# Create conda environment
run(`conda create -n birdbox-yolo python=3.11 -y`)
run(`conda run -n birdbox-yolo pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121`)
run(`conda run -n birdbox-yolo pip install -r requirements-yolo.txt --upgrade-strategy only-if-needed`)


# Point PyCall to the YOLO conda environment
if Sys.iswindows()
    ENV["PYTHON"] = read(`conda run -n birdbox-yolo where python`, String) |> x -> split(x, '\n')[1] |> strip
else
    ENV["PYTHON"] = read(`conda run -n birdbox-yolo which python`, String) |> x -> strip(x)
end

Pkg.build("PyCall")

# ==============================================
# Set up environment for RF-DETR backend 
# ==============================================

# Create conda environment
run(`conda create -n birdbox-rfdetr python=3.12 -y`)
run(`conda run -n birdbox-rfdetr pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121`)
run(`conda run -n birdbox-rfdetr pip install -r requirements-rfdetr.txt --upgrade-strategy only-if-needed`)

# ==============================================
# Download pretrained models 
# ==============================================

# Download into models/ directory if not already present
models_dir = joinpath(@__DIR__, "models")
isdir(models_dir) || mkpath(models_dir)
for name in MODEL_FILES
    dest = joinpath(models_dir, name)
    if isfile(dest)
        @info "Already present" dest
        continue
    end
    url = "$MODEL_RELEASE/$name"
    try
        @info "Downloading $name"
        Downloads.download(url, dest)
    catch err
        @warn "Could not download $name; place it in models/ manually" url exception = err
    end
end