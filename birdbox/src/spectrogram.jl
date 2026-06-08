using ColorSchemes
using Colors
using DSP
using FileIO
using Printf
using Statistics
using FixedPointNumbers: N0f8

const MAGNITUDE_CLIP = (1.0, 99.8) # (lower, upper) percentile of spectrogram values to clip
const GAMMA          = 0.85
const FREQ_RES       = 44100.0 / 4096 # STFT bin spacing in Hz; FFT size is scaled with sample rate to keep this constant.

_hop(sr, duration, time_bins) = max(1, Int(floor((sr * duration) / time_bins)) + 1)
_nfft(sr, freq_res) = max(2, Int(round(sr / freq_res)))

function _linear_resample!(dest::AbstractVector{Float64}, src::AbstractVector{<:Real})
    m, n = length(dest), length(src)
    m == n && (dest .= Float64.(src); return dest)
    n == 1 && (fill!(dest, Float64(src[1])); return dest)
    for (k, xq) in enumerate(LinRange(1.0, Float64(n), m))
        i0 = clamp(floor(Int, xq), 1, n - 1)
        α = xq - i0
        @inbounds dest[k] = muladd(α, Float64(src[i0 + 1]), (1 - α) * Float64(src[i0]))
    end
    return dest
end

function _resample_frequency(M::AbstractMatrix{<:Real}, target::Int)
    Hi, Wi = size(M)
    Hi == target && return Matrix{Float64}(M)
    out = zeros(Float64, target, Wi)
    @views for j in 1:Wi
        _linear_resample!(out[:, j], M[:, j])
    end
    return out
end

function _enhance_contrast(M::AbstractMatrix{Float64}, clip::Tuple{<:Real, <:Real}, gamma::Real)
    isempty(M) && error("Empty spectrogram matrix")
    ql = quantile(vec(M), clip[1] / 100)
    qh = quantile(vec(M), clip[2] / 100)
    qh <= ql && (qh = ql + eps(Float64))
    return clamp.((clamp.(M, ql, qh) .- ql) ./ (qh - ql), 0.0, 1.0) .^ gamma
end

"""
    spectrogram(x, sr; duration=DURATION, fmin=FMIN, fmax=FMAX, fbins=1024,
                clip=(1.0, 99.8), gamma=0.85, freq_res=44100/4096, center=true)

STFT-based log-magnitude spectrogram band-limited to `[fmin, fmax]` (Hz).

Magnitudes are converted to decibels, floored at `dynamic_range` dB,
and lastly percentile-clipped and gamma-compressed to `[0, 1]`.

FFT size and hop length are chosen so that the output spectrogram is roughly `fbins × fbins`
regardless of sample rate and duration. The frequency axis is then resampled to exactly `fbins` rows
while the time axis is left unchanged.

"""
function spectrogram(x::AbstractVector{<:Real}, sr::Real;
        duration::Real = DURATION,
        fmin::Real     = FMIN,
        fmax::Real     = FMAX,
        fbins::Int    = IMGSIZE,
        clip::Tuple{<:Real, <:Real} = MAGNITUDE_CLIP,
        gamma::Real    = GAMMA,
        freq_res::Real = FREQ_RES,
        center::Bool   = true,
        dynamic_range::Real = 80.0, #dB 
    )
    sr = Float64(sr)
    nfft = _nfft(sr, freq_res)
    noverlap = nfft - _hop(sr, duration, 1024)
    xw = center ? vcat(zeros(nfft ÷ 2), x, zeros(nfft ÷ 2)) : x
    Sg = DSP.spectrogram(xw, nfft, noverlap; fs = sr, window = DSP.hanning)
    P = Matrix{Float64}(Sg.power)
    P = max.(DSP.pow2db.(P ./ max(maximum(P), eps(Float64))), -dynamic_range)
    f = vec(Sg.freq)
    P = P[(f .>= fmin) .& (f .<= fmax), :]
    P = _resample_frequency(P, fbins)
    return _enhance_contrast(P, clip, gamma)
end

"""
    spec2img(spec; colormap=ColorSchemes.magma) -> Matrix{RGB{N0f8}}

Convert spectrogram to RGB image using the given `colormap`.
"""
function spec2img(spec::AbstractMatrix{<:Real}; colormap = ColorSchemes.magma)
    H, W = size(spec)
    img = Matrix{RGB{N0f8}}(undef, H, W)
    @inbounds for j in 1:W, i in 1:H
        v = spec[i, j]
        img[i, j] = get(colormap, isnan(v) ? 0.0 : clamp(v, 0.0, 1.0))
    end
    return img[H:-1:1, :]
end

"""
    time_in_recording(filename) -> Float64

Extract the start time from the recording given the filename.
If filename does not ends with `_t<ms>`, return 0.
"""
function time_in_recording(filename::AbstractString)
    m = match(r"_t(\d+)$", filename)
    return isnothing(m) ? 0.0 : parse(Int, m.captures[1]) / 1000
end

"""
    write_spectrogram_images(recording; outdir=joinpath(PRED_DIR, "images"),
                             clip_duration=DURATION, overlap=OVERLAP,
                             fmin=FMIN, fmax=FMAX, channel=CHANNEL) -> outdir

Split `recording` into overlapping clips, compute spectrograms, convert to
RGB images and save to disk.
"""
function write_spectrogram_images(recording::String;
        outdir = joinpath(PRED_DIR, "images"),
        clip_duration = DURATION,
        overlap = OVERLAP,
        fmin = FMIN,
        fmax = FMAX,
        channel = CHANNEL,
    )
    samples, sr = load_audio(recording; channel)
    clips = split_recording(samples, sr; clip_duration, overlap)
    stride = clip_duration - overlap

    isdir(outdir) || mkpath(outdir)
    stem = splitext(basename(recording))[1]

    @info "Generating spectrogram images..."
    for (i, clip) in enumerate(clips)
        t0 = (i - 1) * stride
        spec = spectrogram(clip, sr; duration = clip_duration, fmin, fmax)
        img = spec2img(spec)
        path = joinpath(outdir, Printf.@sprintf("%s_ch%d_w%04d_t%d.png", stem, channel, i - 1, round(Int, t0 * 1000)))
        save(path, img)
    end
    return outdir
end
