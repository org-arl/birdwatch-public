using PyCall

_rfdetr_python() = get(ENV, "RFDETR_PYTHON", "python3")
_rfdetr_script(name::AbstractString) = joinpath(PKG_ROOT, "python", "rfdetr", name)

function _rfdetr_device(device)
    device === nothing && return nothing
    s = string(device)
    s == "cpu" && return "cpu"
    occursin(r"^\d+$", s) && return "cuda:$s"
    return s
end

function _detect_yolo(model_path::String;
        imgdir::String, savedir::String, imgsize::Int, minconf::Real,
        nms_iou::Real, device::String, kwargs...)
    yolo_model = pyimport("ultralytics").YOLO(model_path)
    @info "Detecting bird calls..."
    yolo_model.predict(;
        source = imgdir,
        imgsz = imgsize,
        device = device,
        conf = minconf,
        iou = nms_iou,
        save_txt = true,
        save_conf = true,
        save_dir = savedir,
        kwargs...,
    )
end

function _detect_rfdetr(model_path::String;
        imgdir::String, savedir::String, minconf::Real, device::String, kwargs...)
    labels = joinpath(savedir, "labels")
    mkpath(labels)
    args = String[
        _rfdetr_python(),
        _rfdetr_script("predict.py"),
        "--checkpoint", model_path,
        "--images", imgdir,
        "--out", labels,
        "--confidence", string(minconf),
    ]
    dev = _rfdetr_device(device)
    if dev !== nothing
        push!(args, "--device", dev)
    end
    @info "Detecting bird calls..."
    run(Cmd(args))
end

"""
    detect(model_path; backend=:yolo, imgdir=joinpath(PRED_DIR, "images"),
           savedir=PRED_DIR,
           imgsize=IMGSIZE, minconf=MINCONF, nms_iou=NMS_IOU, device="0") -> DataFrame

Run a detector on spectrogram PNGs in `imgdir` using weights at `model_path`
and save predicted label files under `joinpath(savedir, "labels")`.

`backend` is `:yolo` (Ultralytics, default) or `:rfdetr`. Extra `kwargs` are
forwarded to the YOLO predictor only.

Returns a `DataFrame` with predicted bounding boxes in both normalized YOLO format
and in absolute time (s) and frequency (Hz).
"""
function detect(model_path::String;
        backend::Symbol = :yolo,
        imgdir::String = joinpath(PRED_DIR, "images"),
        savedir::String = PRED_DIR,
        imgsize::Int = IMGSIZE,
        minconf::Real = MINCONF,
        nms_iou::Real = NMS_IOU,
        device::String = "0",
        kwargs...
    )
    if backend === :yolo
        _detect_yolo(model_path; imgdir, savedir, imgsize, minconf, nms_iou, device, kwargs...)
    elseif backend === :rfdetr
        _detect_rfdetr(model_path; imgdir, savedir, minconf, device, kwargs...)
    else
        error("backend must be :yolo or :rfdetr, got $(repr(backend))")
    end
    detections_df = read_yolo_labels(joinpath(savedir, "labels"))
    start_times = time_in_recording.(detections_df.source)
    return add_timefreq_columns(detections_df, FMIN, FMAX, start_times, DURATION)
end

function detect(recording::String, model_path::String;
        imgdir::String = joinpath(PRED_DIR, "images"),
        savedir::String = PRED_DIR,
        kwargs...
    )
    write_spectrogram_images(recording; outdir = joinpath(savedir, "images"))
    return detect(model_path; imgdir, savedir, kwargs...)
end
