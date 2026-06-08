using DataFrames

"""
    read_yolo_labels(sourcedir) -> DataFrame

Read every YOLO-format `.txt` label file in `sourcedir` into a single `DataFrame`
with columns `source, class, xcenter, ycenter, width, height, confidence`. Rows
missing a confidence column are filled with `1.0`.
"""
function read_yolo_labels(sourcedir::String)
    source = String[]
    class = Int[]
    xcenter = Float64[]
    ycenter = Float64[]
    width = Float64[]
    height = Float64[]
    confidence = Float64[]
    for name in sort(readdir(sourcedir))
        endswith(name, ".txt") || continue
        stem = splitext(name)[1]
        for line in eachline(joinpath(sourcedir, name))
            parts = split(strip(line))
            length(parts) >= 5 || continue
            push!(source, stem)
            push!(class, round(Int, parse(Float64, parts[1])))
            xc, yc, w, h = parse.(Float64, parts[2:5])
            push!(xcenter, round(xc, digits=3))
            push!(ycenter, round(yc, digits=3))
            push!(width, round(w, digits=3))
            push!(height, round(h, digits=3))
            push!(confidence, round(length(parts) >= 6 ? parse(Float64, parts[6]) : 1.0, digits=3))
        end
    end
    return DataFrame(;
        source, class, xcenter, ycenter, width, height, confidence)
end

"""
    add_timefreq_columns(df, fmin, fmax, tstart, tend) -> DataFrame

Add bounding box coordinates in time (s) and frequency (Hz)
based on the existing YOLO-normalized coordinates.
"""
function add_timefreq_columns(df::DataFrame, fmin::Real, fmax::Real,
        tstart::AbstractVector{<:Real}, duration::Real)
    fspan = fmax - fmin
    t0 = Vector{Float64}(undef, nrow(df))
    t1 = Vector{Float64}(undef, nrow(df))
    f0 = Vector{Float64}(undef, nrow(df))
    f1 = Vector{Float64}(undef, nrow(df))
    for i in 1:nrow(df)
        xc, yc, w, h = df.xcenter[i], df.ycenter[i], df.width[i], df.height[i]
        x0 = clamp(xc - w / 2, 0.0, 1.0)
        x1 = clamp(xc + w / 2, 0.0, 1.0)
        y0 = clamp(yc - h / 2, 0.0, 1.0)
        y1 = clamp(yc + h / 2, 0.0, 1.0)
        t0[i] = tstart[i] + x0 * duration
        t1[i] = tstart[i] + x1 * duration
        f0[i] = fmin + (1 - y1) * fspan
        f1[i] = fmin + (1 - y0) * fspan
    end
    out = copy(df)
    out.t0 = t0
    out.t1 = t1
    out.f0 = f0
    out.f1 = f1
    return sort(out, :t0)
end
