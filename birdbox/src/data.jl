using FileIO
using Printf
using Colors


"""
    write_split_file(split, files; outdir="data/") -> path

Write one image path per line to `<outdir>/<split>.txt`.
"""
function write_split_file(split::String, files::AbstractVector{<:AbstractString}; outdir = "data/")
    isdir(outdir) || mkpath(outdir)
    path = joinpath(outdir, "$split.txt")
    open(path, "w") do io
        for f in files
            println(io, f)
        end
    end
    @info "Saved $split.txt"
    return path
end

"""
    write_data_yaml(path; root="data/", train="train.txt", val="val.txt",
                    test="test.txt", names=["call"], imgsz=IMGSIZE) -> path

Write a YOLO `data.yaml` describing the dataset split files and class names.
"""
function write_data_yaml(;
        path::AbstractString = "data/data.yaml",
        root::AbstractString = "data/",
        train::AbstractString = "train.txt",
        val::AbstractString = "val.txt",
        test::AbstractString = "test.txt",
        names::AbstractVector{<:AbstractString} = ["bird_call"],
        imgsz::Int = IMGSIZE,
    )
    nc = length(names)
    names_block = join(("- $n" for n in names), "\n")
    write(path, """
            path: $root
            train: $train
            val: $val
            test: $test
            nc: $nc
            names:
            $names_block
            task: detect
            imgsz: $imgsz
        """)
    return path
end

function _pad(img::AbstractMatrix; target::Int = IMGSIZE)
    src = img isa AbstractMatrix{<:RGB} ? img : RGB.(img)
    h, w = size(src)
    (w > target || h > target) && error("image $(w)×$(h) exceeds target $target")
    w == target && h == target && return src
    T = eltype(src)
    canvas = fill(T(0, 0, 0), target, target)
    canvas[1:h, 1:w] .= src
    return canvas
end

_yolo_to_padded(xc, yc, w, h, orig_w, orig_h; target = IMGSIZE) =
    (xc * orig_w / target, yc * orig_h / target, w * orig_w / target, h * orig_h / target)

_label_path(image_path::AbstractString) =
    joinpath(dirname(dirname(image_path)), "labels", splitext(basename(image_path))[1] * ".txt")

function _read_split_list(path::AbstractString)
    return [strip(line) for line in eachline(path) if !isempty(strip(line))]
end

function _write_padded_split(name::AbstractString, image_paths, outdir::AbstractString, target::Int)
    img_dir = joinpath(outdir, name, "images")
    lab_dir = joinpath(outdir, name, "labels")
    mkpath(img_dir)
    mkpath(lab_dir)
    n_box = 0
    for src in image_paths
        isfile(src) || error("missing image: $src")
        label = _label_path(src)
        isfile(label) || error("missing label: $label")
        img = load(src)
        orig_h, orig_w = size(img)
        save(joinpath(img_dir, basename(src)), _pad(img; target))
        lines = String[]
        for line in eachline(label)
            parts = split(strip(line))
            length(parts) >= 5 || continue
            xc, yc, w, h = parse.(Float64, parts[2:5])
            xc_p, yc_p, w_p, h_p = _yolo_to_padded(xc, yc, w, h, orig_w, orig_h; target)
            extra = length(parts) > 5 ? " " * join(parts[6:end], " ") : ""
            push!(lines, Printf.@sprintf("%s %.6f %.6f %.6f %.6f%s",
                parts[1], xc_p, yc_p, w_p, h_p, extra))
        end
        write(joinpath(lab_dir, splitext(basename(src))[1] * ".txt"),
            isempty(lines) ? "" : join(lines, "\n") * "\n")
        n_box += length(lines)
    end
    @info "$name: $(length(image_paths)) images, $n_box boxes"
    return length(image_paths), n_box
end

"""
    write_padded_dataset(outdir; train, val, test=nothing, names=["call"],
                         target=IMGSIZE) -> outdir

Build an RF-DETR dataset from YOLO split lists: right-pad spectrogram PNGs to
`target × target` and rewrite labels into padded-image coordinates.

`train`, `val`, and optional `test` are paths to `.txt` files listing image
paths (as written by [`write_split_file`](@ref)). Labels are read from a sibling
`labels/` directory. Writes `train/`, `valid/` (and `test/` if given) plus
`data.yaml`.
"""
function write_padded_dataset(outdir::AbstractString;
        train::AbstractString,
        val::AbstractString,
        test::Union{AbstractString, Nothing} = nothing,
        names::AbstractVector{<:AbstractString} = ["call"],
        target::Int = IMGSIZE,
    )
    isdir(outdir) || mkpath(outdir)
    splits = ["train" => train, "valid" => val]
    test !== nothing && push!(splits, "test" => test)
    for (name, list_path) in splits
        _write_padded_split(name, _read_split_list(list_path), outdir, target)
    end
    names_block = join(("  - $n" for n in names), "\n")
    test_line = test === nothing ? "" : "test: test/images\n"
    write(joinpath(outdir, "data.yaml"), """
            names:
            $names_block
            nc: $(length(names))
            train: train/images
            val: valid/images
            $test_line""")
    return outdir
end
