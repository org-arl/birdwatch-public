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
