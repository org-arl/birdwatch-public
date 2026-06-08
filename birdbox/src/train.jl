using PyCall

"""
    train(model_path, data_yaml; imgsz=IMGSIZE, project=TRAIN_DIR, kwargs...)

Train or fine-tune a YOLO model in-process via Ultralytics. Extra `kwargs` are
forwarded to `ultralytics.YOLO.train` (e.g. `epochs`, `batch`, `lr0`, `patience`,
`device`, `name`, `single_cls`, etc.).
"""
function train(model_path::String, data_yaml::String;
        imgsz = IMGSIZE,
        project = TRAIN_DIR,
        kwargs...,
    )
    pyimport("ultralytics").YOLO(model_path).train(;
        data = data_yaml,
        imgsz,
        project,
        kwargs...,
    )
end
