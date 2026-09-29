using PyCall

"""
    train(model, data; backend=:yolo, imgsz=IMGSIZE, project=TRAIN_DIR, kwargs...)

Train or fine-tune a detector.

- `backend=:yolo` (default) — `model` is a `.pt` path, `data` is a YOLO
  `data.yaml`. Extra `kwargs` are forwarded to `ultralytics.YOLO.train`.
- `backend=:rfdetr` — `model` is `:nano`, `:large`, or a `.pth` path; `data`
  is a padded dataset directory from [`write_padded_dataset`](@ref).
"""
function train(model, data;
        backend::Symbol = :yolo,
        imgsz = IMGSIZE,
        project = TRAIN_DIR,
        kwargs...,
    )
    if backend === :yolo
        return pyimport("ultralytics").YOLO(string(model)).train(;
            data = string(data),
            imgsz,
            project,
            kwargs...,
        )
    elseif backend === :rfdetr
        return _train_rfdetr(model, data; project, kwargs...)
    else
        error("backend must be :yolo or :rfdetr, got $(repr(backend))")
    end
end

function _train_rfdetr(model, dataset_dir;
        project = TRAIN_DIR,
        name = "rfdetr",
        epochs = 100,
        patience = 10,
        batch = 16,
        device = "0",
        skip_best_epochs = 3,
    )
    model_arg = model isa Symbol ? string(model) : string(model)
    out = joinpath(string(project), string(name))
    args = String[
        _rfdetr_python(),
        _rfdetr_script("train.py"),
        "--model", model_arg,
        "--dataset-dir", string(dataset_dir),
        "--output-dir", out,
        "--epochs", string(epochs),
        "--patience", string(patience),
        "--batch", string(batch),
        "--skip-best-epochs", string(skip_best_epochs),
    ]
    dev = _rfdetr_device(device)
    if dev !== nothing
        push!(args, "--device", dev)
    end
    run(Cmd(args))
end
