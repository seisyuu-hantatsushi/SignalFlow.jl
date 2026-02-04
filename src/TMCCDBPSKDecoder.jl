module TMCCDBPSKDecoder

import ..SignalFlowBlock
import ..input!
import ..RingBuffers: RingFrameBuffer

mutable struct TMCCDBPSKDecoderContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    nfft::Int
    bins::Vector{Int}
    label::String
    frame_symbols::Int
    symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}}
    skip_ref_symbol::Bool
    ref_symbol_index::Int
    log_interval::Float64
    hist_len::Int
    last_log_time::Float64
    symbol_counter::Int
    prev_symbols::Vector{ComplexF32}
    has_prev::Vector{Bool}
    bit_history::Vector{Int}
    conf_history::Vector{Float64}
    carrier_bit_histories::Vector{Vector{Int}}
    sync_words::Vector{Vector{Int}}
    sync_acq_threshold::Float64
    sync_track_threshold::Float64
    sync_unlock_threshold::Float64
    sync_lock_confirm::Int
    sync_lock_count::Int
    sync_streak_best::Float64
    sync_locked::Bool
    sync_fine_locked::Bool
    sync_fine_threshold::Float64
    sync_fine_confirm::Int
    sync_fine_count::Int
    sync_fine_fail_count::Int
    sync_fine_fail_tolerate::Int
    sync_phase_guard::Int
    sync_phase::Int
    sync_carrier::Int
    sync_expect_word::Int
    sync_phase_votes::Vector{Int}
    sync_pair_votes::Matrix{Int}
    sync_last_score::Float64
    sync_last_word::Int
    sync_last_carrier::Int
    outbuf::Vector{ComplexF32}
    ringbuffer::RingFrameBuffer{ComplexF32}
    holdbuf::Union{Nothing, Int}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

const LOG_LOCK = ReentrantLock()

@inline function phase_distance(a::Int, b::Int, period::Int)
    d = abs(a - b)
    return min(d, period - d)
end

@inline function atomic_log_line(msg::AbstractString)
    lock(LOG_LOCK) do
        print(stdout, msg, '\n')
    end
end

function CreateTMCCDBPSKDecoder(; nfft::Int = 8192,
                                bins::Vector{Int},
                                label::AbstractString = "TMCC",
                                frame_symbols::Int = 204,
                                symbol_index_ref::Union{Nothing,Base.Threads.Atomic{Int}} = nothing,
                                skip_ref_symbol::Bool = true,
                                ref_symbol_index::Int = 1,
                                log_interval::Real = 0.5,
                                hist_len::Int = 64,
                                sync_word::Union{Nothing,Vector{Int}} = nothing,
                                sync_acq_threshold::Real = 0.75,
                                sync_track_threshold::Real = 0.75,
                                sync_unlock_threshold::Real = 0.625,
                                sync_lock_confirm::Int = 3,
                                sync_fine_threshold::Real = 0.688,
                                sync_fine_confirm::Int = 4,
                                sync_fine_fail_tolerate::Int = 3,
                                sync_phase_guard::Int = 1,
                                poolsize::Int = 8)
    nfft < 32 && error("TMCCDBPSKDecoder: nfft must be >= 32.")
    isempty(bins) && error("TMCCDBPSKDecoder: bins must not be empty.")
    log_interval <= 0 && error("TMCCDBPSKDecoder: log_interval must be > 0.")
    hist_len < 1 && error("TMCCDBPSKDecoder: hist_len must be >= 1.")
    frame_symbols < 1 && error("TMCCDBPSKDecoder: frame_symbols must be >= 1.")
    ref_symbol_index < 1 && error("TMCCDBPSKDecoder: ref_symbol_index must be >= 1.")
    ref_symbol_index > frame_symbols && error("TMCCDBPSKDecoder: ref_symbol_index must be <= frame_symbols.")
    (0.0 <= sync_acq_threshold <= 1.0) || error("TMCCDBPSKDecoder: sync_acq_threshold must be in [0, 1].")
    (0.0 <= sync_track_threshold <= 1.0) || error("TMCCDBPSKDecoder: sync_track_threshold must be in [0, 1].")
    (0.0 <= sync_unlock_threshold <= 1.0) || error("TMCCDBPSKDecoder: sync_unlock_threshold must be in [0, 1].")
    (0.0 <= sync_fine_threshold <= 1.0) || error("TMCCDBPSKDecoder: sync_fine_threshold must be in [0, 1].")
    sync_track_threshold < sync_acq_threshold && error("TMCCDBPSKDecoder: sync_track_threshold must be >= sync_acq_threshold.")
    sync_unlock_threshold > sync_acq_threshold && error("TMCCDBPSKDecoder: sync_unlock_threshold must be <= sync_acq_threshold.")
    sync_lock_confirm < 1 && error("TMCCDBPSKDecoder: sync_lock_confirm must be >= 1.")
    sync_fine_confirm < 1 && error("TMCCDBPSKDecoder: sync_fine_confirm must be >= 1.")
    sync_fine_fail_tolerate < 1 && error("TMCCDBPSKDecoder: sync_fine_fail_tolerate must be >= 1.")
    sync_phase_guard < 0 && error("TMCCDBPSKDecoder: sync_phase_guard must be >= 0.")
    poolsize < 1 && error("TMCCDBPSKDecoder: poolsize must be at least 1.")
    @inbounds for b in bins
        (b < 1 || b > nfft) && error("TMCCDBPSKDecoder: bin out of range: $b")
    end
    sync_words = Vector{Vector{Int}}()
    if sync_word === nothing || isempty(sync_word)
        # ARIB STD-B31 3.15.4 / Table 3-20: w0/w1 toggled every frame.
        push!(sync_words, [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0])
        push!(sync_words, [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    else
        @inbounds for b in sync_word
            (b == 0 || b == 1) || error("TMCCDBPSKDecoder: sync_word must contain only 0/1.")
        end
        push!(sync_words, copy(sync_word))
    end
    if !isempty(sync_words)
        wlen = length(sync_words[1])
        @inbounds for w in sync_words
            length(w) == wlen || error("TMCCDBPSKDecoder: sync words must have same length.")
        end
    end

    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    ctx = TMCCDBPSKDecoderContext(Base.Threads.Atomic{Bool}(true),
                                  nfft,
                                  copy(bins),
                                  String(label),
                                  frame_symbols,
                                  symbol_index_ref,
                                  skip_ref_symbol,
                                  ref_symbol_index,
                                  Float64(log_interval),
                                  Int(hist_len),
                                  0.0,
                                  0,
                                  fill(ComplexF32(0, 0), length(bins)),
                                  fill(false, length(bins)),
                                  Int[],
                                  Float64[],
                                  [Int[] for _ in 1:length(bins)],
                                  sync_words,
                                  Float64(sync_acq_threshold),
                                  Float64(sync_track_threshold),
                                  Float64(sync_unlock_threshold),
                                  Int(sync_lock_confirm),
                                  0,
                                  0.0,
                                  false,
                                  false,
                                  Float64(sync_fine_threshold),
                                  Int(sync_fine_confirm),
                                  0,
                                  0,
                                  Int(sync_fine_fail_tolerate),
                                  Int(sync_phase_guard),
                                  1,
                                  0,
                                  1,
                                  zeros(Int, frame_symbols),
                                  zeros(Int, frame_symbols, length(bins)),
                                  0.0,
                                  0,
                                  0,
                                  Vector{ComplexF32}(undef, nfft),
                                  RingFrameBuffer(ComplexF32, nfft, poolsize),
                                  nothing,
                                  nothing,
                                  new_sinks,
                                  sinks)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function task!(context::TMCCDBPSKDecoderContext)
    try
        while context.running[]
            if isready(context.ringbuffer.fullQ)
                rd_index = take!(context.ringbuffer.fullQ)
                rd_buffer = context.ringbuffer.bufs[rd_index]
                if rd_buffer.store_size == context.nfft
                    copyto!(context.outbuf, 1, rd_buffer.buf, 1, context.nfft)
                    if context.symbol_index_ref !== nothing
                        context.symbol_counter = context.symbol_index_ref[]
                    end
                    is_ref_symbol = false
                    if context.skip_ref_symbol && context.sync_locked
                        # B1..B16 start at sync_phase, so B0 is one symbol before that.
                        ref_phase = context.sync_phase == 1 ? context.frame_symbols : (context.sync_phase - 1)
                        is_ref_symbol = (mod(context.symbol_counter - ref_phase, context.frame_symbols) == 0)
                    end

                    bit_acc = 0.0
                    valid = 0
                    bits = Vector{Int}(undef, length(context.bins))
                    conf = 0.0
                    @inbounds for i in 1:length(context.bins)
                        idx = context.bins[i]
                        v = rd_buffer.buf[idx]
                        bits[i] = -1
                        if context.has_prev[i]
                            d = v * conj(context.prev_symbols[i])
                            mag = abs(d)
                            if mag > 0
                                m = real(d) / mag
                                bits[i] = m < 0 ? 1 : 0
                                bit_acc += m
                                conf += abs(m)
                                valid += 1
                            end
                        end
                        context.prev_symbols[i] = v
                        context.has_prev[i] = true
                    end

                    if valid > 0 && !is_ref_symbol
                        bit = bit_acc < 0 ? 1 : 0
                        conf = conf / valid
                        push!(context.bit_history, bit)
                        push!(context.conf_history, conf)
                        if length(context.bit_history) > context.hist_len
                            deleteat!(context.bit_history, 1)
                            deleteat!(context.conf_history, 1)
                        end
                        @inbounds for i in 1:length(bits)
                            if bits[i] >= 0
                                push!(context.carrier_bit_histories[i], bits[i])
                                if length(context.carrier_bit_histories[i]) > context.hist_len
                                    deleteat!(context.carrier_bit_histories[i], 1)
                                end
                            end
                        end

                        if !isempty(context.sync_words)
                            swlen = length(context.sync_words[1])
                            best_car = 0
                            best_w = 0
                            best_m = -1
                            for ci in 1:length(context.carrier_bit_histories)
                                h = context.carrier_bit_histories[ci]
                                if length(h) < swlen
                                    continue
                                end
                                base = length(h) - swlen
                                cand_m = -1
                                cand_w = 0
                                for wi in 1:length(context.sync_words)
                                    w = context.sync_words[wi]
                                    m = 0
                                    @inbounds for i in 1:swlen
                                        m += (h[base + i] == w[i]) ? 1 : 0
                                    end
                                    if m > cand_m
                                        cand_m = m
                                        cand_w = wi
                                    end
                                end
                                if cand_m > best_m
                                    best_m = cand_m
                                    best_w = cand_w
                                    best_car = ci
                                end
                            end
                            if best_car != 0
                                score = best_m / swlen
                                context.sync_last_score = score
                                context.sync_last_word = best_w
                                context.sync_last_carrier = best_car
                                if score >= context.sync_acq_threshold
                                    head_symbol = context.symbol_counter - swlen + 1
                                    phase = mod(head_symbol, context.frame_symbols)
                                    phase == 0 && (phase = context.frame_symbols)
                                    if !context.sync_locked
                                        context.sync_phase_votes[phase] += 1
                                        context.sync_pair_votes[phase, best_car] += 1
                                        phase_votes = context.sync_phase_votes[phase]
                                        if phase_votes >= context.sync_lock_confirm &&
                                           score >= context.sync_track_threshold
                                            context.sync_locked = true
                                            context.sync_fine_locked = false
                                            context.sync_fine_count = 0
                                            context.sync_phase = phase
                                            context.sync_carrier = best_car
                                            context.sync_lock_count = phase_votes
                                            context.sync_streak_best = score
                                            context.sync_expect_word = if length(context.sync_words) > 1
                                                best_w == 1 ? 2 : 1
                                            else
                                                best_w
                                            end
                                        end
                                    else
                                        ph_ok = phase_distance(phase, context.sync_phase, context.frame_symbols) <= 1
                                        word_ok = (length(context.sync_words) <= 1) ||
                                                  (best_w == context.sync_expect_word)
                                        if ph_ok && score >= context.sync_track_threshold
                                            if score >= context.sync_track_threshold
                                                context.sync_lock_count = min(context.sync_lock_count + 1,
                                                                              context.sync_lock_confirm * 3)
                                                # Guard phase updates to avoid occasional large phase jumps.
                                                if phase_distance(phase, context.sync_phase, context.frame_symbols) <= context.sync_phase_guard
                                                    context.sync_phase = phase
                                                end
                                                context.sync_carrier = best_car
                                                if length(context.sync_words) > 1
                                                    if word_ok
                                                        context.sync_expect_word = (context.sync_expect_word == 1) ? 2 : 1
                                                    end
                                                end
                                            end
                                        else
                                            context.sync_lock_count -= word_ok ? 1 : 2
                                            if context.sync_lock_count <= 0
                                                context.sync_locked = false
                                                context.sync_fine_locked = false
                                                context.sync_fine_count = 0
                                                context.sync_lock_count = 0
                                                context.sync_streak_best = 0.0
                                                context.sync_carrier = 0
                                                fill!(context.sync_pair_votes, 0)
                                            end
                                        end
                                    end
                                elseif context.sync_locked && score < context.sync_unlock_threshold
                                    context.sync_lock_count -= 1
                                    if context.sync_lock_count <= 0
                                        context.sync_locked = false
                                        context.sync_fine_locked = false
                                        context.sync_fine_count = 0
                                        context.sync_lock_count = 0
                                        context.sync_streak_best = 0.0
                                        context.sync_carrier = 0
                                        fill!(context.sync_pair_votes, 0)
                                    end
                                end
                                if context.sync_locked
                                    head_symbol = context.symbol_counter - swlen + 1
                                    phase_now = mod(head_symbol, context.frame_symbols)
                                    phase_now == 0 && (phase_now = context.frame_symbols)
                                    phase_ok = phase_distance(phase_now, context.sync_phase, context.frame_symbols) <= 2
                                    word_ok = (length(context.sync_words) <= 1) || (best_w == context.sync_expect_word)
                                    carrier_ok = best_car == context.sync_carrier
                                    if phase_ok && score >= context.sync_fine_threshold
                                        delta = 1
                                        word_ok && (delta += 1)
                                        carrier_ok && (delta += 1)
                                        context.sync_fine_count = min(context.sync_fine_count + delta,
                                                                      context.sync_fine_confirm * 4)
                                        context.sync_fine_fail_count = 0
                                    else
                                        # Decay only after consecutive misses, improving fine-lock hold.
                                        context.sync_fine_fail_count += 1
                                        if context.sync_fine_fail_count >= context.sync_fine_fail_tolerate
                                            context.sync_fine_count = max(context.sync_fine_count - 1, 0)
                                            context.sync_fine_fail_count = 0
                                        end
                                    end
                                    context.sync_fine_locked = context.sync_fine_count >= context.sync_fine_confirm
                                else
                                    context.sync_fine_locked = false
                                    context.sync_fine_count = 0
                                    context.sync_fine_fail_count = 0
                                end
                            end
                        end
                        now = time()
                        if now - context.last_log_time >= context.log_interval
                            bitstr = join((b < 0 ? "x" : string(b) for b in bits), "")
                            histstr = join(string.(context.bit_history), "")
                            msg = "TMCCDBPSK[" * context.label *
                                  "]: bit=" * string(bit) *
                                  " metric=" * string(round(bit_acc / valid, digits = 3)) *
                                  " conf=" * string(round(conf, digits = 3)) *
                                  " bins=" * bitstr *
                                  " hist=" * histstr *
                                  " sync_locked=" * string(context.sync_locked) *
                                  " sync_fine_locked=" * string(context.sync_fine_locked) *
                                  " sync_phase=" * string(context.sync_phase) *
                                  " sync_score=" * string(round(context.sync_last_score, digits = 3)) *
                                  " sync_word=" * string(context.sync_last_word) *
                                  " sync_carrier=" * string(context.sync_last_carrier) *
                                  " lock_carrier=" * string(context.sync_carrier) *
                                  " lock_votes=" * string(context.sync_lock_count) *
                                  " fine_votes=" * string(context.sync_fine_count)
                            atomic_log_line(msg)
                            context.last_log_time = now
                        end
                    end
                    context.symbol_counter += 1

                    while isready(context.new_sinks)
                        push!(context.sinks, take!(context.new_sinks))
                    end
                    for sink in context.sinks
                        input!(sink, context.outbuf, context.nfft)
                    end
                end
                rd_buffer.store_size = 0
                put!(context.ringbuffer.freeQ, rd_index)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("TMCCDBPSKDecoder error: ", e)
        end
    end
    return nothing
end

function input!(context::TMCCDBPSKDecoderContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size != context.nfft
        return -1
    end

    if isready(context.ringbuffer.freeQ)
        idx = take!(context.ringbuffer.freeQ)
        buf = context.ringbuffer.bufs[idx]
        copyto!(buf.buf, 1, samples, 1, actual_size)
        buf.store_size = actual_size
        put!(context.ringbuffer.fullQ, idx)
    else
        return -1
    end

    return samples_size
end

function stop!(context::TMCCDBPSKDecoderContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
