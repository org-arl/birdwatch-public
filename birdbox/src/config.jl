const PKG_ROOT = dirname(@__DIR__)
const PRED_DIR = joinpath(PKG_ROOT, "detect")
const TRAIN_DIR = joinpath(PKG_ROOT, "train")

const FMIN     = 500.0
const FMAX     = 12_000.0
const DURATION = 6.0
const OVERLAP  = 1.0
const CHANNEL  = 1

const IMGSIZE  = 1024
const NMS_IOU  = 0.7
const MINCONF  = 0.15
