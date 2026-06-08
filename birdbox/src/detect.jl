using PyCall

"""
    detect(model_path; imgdir=joinpath(PRED_DIR, "images"),
           savedir=PRED_DIR,
           imgsize=IMGSIZE, minconf=MINCONF, nms_iou=NMS_IOU, device="0") -> DataFrame

Run YOLO on spectrogram PNGs in `imgdir` using weights at `model_path`
and save predicted label files under `joinpath(savedir, "labels")`.

Returns a `DataFrame` with predicted bounding boxes in both normalized YOLO format
and in absolute time (s) and frequency (Hz).
"""
function detect(model_path::String;
        imgdir::String = joinpath(PRED_DIR, "images"),
        savedir::String = PRED_DIR,
        imgsize::Int = IMGSIZE,
        minconf::Real = MINCONF,
        nms_iou::Real = NMS_IOU,
        device::String = "0",
        kwargs...
    )
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
