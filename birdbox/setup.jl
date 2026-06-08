# First-time setup for BirdBox. Run as a script with `julia --project=. setup.jl`,
# or step through it line-by-line in a Julia REPL.


cd(@__DIR__)
using Pkg
Pkg.activate(".")

# Create conda environment
run(`conda create -n birdbox python=3.11 -y`)

# Install Python dependencies into that environment
run(`conda run -n birdbox pip install -r requirements.txt`)

# Install Julia dependencies
Pkg.instantiate()

# Point PyCall to the conda environment
if Sys.iswindows()
    ENV["PYTHON"] = read(`conda run -n birdbox where python`, String) |> x -> split(x, '\n')[1] |> strip
else
    ENV["PYTHON"] = read(`conda run -n birdbox which python`, String) |> strip
end

Pkg.build("PyCall")
