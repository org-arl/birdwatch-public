using FileIO

"""
    load_audio(path; channel=1) -> (samples::Vector{Float64}, sr::Float64)

Load one channel of an audio file. Supports any format handled by `FileIO`/`LibSndFile`
(WAV, OGG, FLAC, ...).
"""
function load_audio(path::AbstractString; channel::Integer = 1)
    s = FileIO.load(String(path))
    if s isa Tuple
        raw, sr_raw = s[1], s[2]
        data = ndims(raw) == 1 ? reshape(raw, :, 1) : raw
        sr = Float64(sr_raw)
    else
        data = ndims(s.data) == 1 ? reshape(s.data, :, 1) : s.data
        sr = Float64(s.samplerate)
    end
    1 <= channel <= size(data, 2) ||
        error("$(path) has $(size(data, 2)) channel(s); requested channel=$channel")
    return Float64.(vec(data[:, channel])), sr
end

"""
    split_recording(samples, sr; clip_duration=DURATION, overlap=OVERLAP) -> clips

Split `samples` into overlapping clips of length `clip_duration` seconds.
The last clip is zero-padded when the recording ends mid-clip.
"""
function split_recording(samples::AbstractVector{<:Real}, sr::Real; clip_duration = DURATION, overlap = OVERLAP)
    stride = clip_duration - overlap
    stride > 0 || error("overlap ($overlap) must be smaller than clip_duration ($clip_duration)")
    duration = length(samples) / sr
    starts = collect(0.0:stride:max(duration - 1e-9, 0.0))
    clips = Vector{Vector{eltype(samples)}}(undef, length(starts))
    for (i, t0) in enumerate(starts)
        s0 = round(Int, t0 * sr) + 1
        s1 = round(Int, (t0 + clip_duration) * sr)
        s0 = clamp(s0, 1, length(samples) + 1)
        if s1 <= length(samples)
            clips[i] = vec(samples[s0:s1])
        else
            clips[i] = vcat(samples[s0:end], zeros(eltype(samples), s1 - length(samples)))
        end
    end
    return clips
end
