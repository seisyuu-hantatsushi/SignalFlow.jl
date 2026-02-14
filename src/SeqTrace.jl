module SeqTrace

import ..AsyncLogger

const ENABLED = Base.Threads.Atomic{Bool}(false)
const NEXT_SEQ = Base.Threads.Atomic{UInt64}(0)
const LOG_LOCK = ReentrantLock()
const BUFFER_SEQ = IdDict{Any, UInt64}()
const LAST_IN = IdDict{Any, UInt64}()
const LAST_OUT = IdDict{Any, UInt64}()
const EVENT_COUNT = Base.Threads.Atomic{UInt64}(0)
const LOG_INTERVAL = Base.Threads.Atomic{UInt64}(200)
const LOG_OK = Base.Threads.Atomic{Bool}(false)
const STAGE_FILTER = Base.RefValue{Union{Nothing, String}}(nothing)

function configure!(; enabled::Bool = false, log_interval::Int = 200, log_ok::Bool = false,
                     stage_filter::Union{Nothing, AbstractString} = nothing)
    log_interval < 1 && error("SeqTrace: log_interval must be >= 1.")
    ENABLED[] = enabled
    LOG_INTERVAL[] = UInt64(log_interval)
    LOG_OK[] = log_ok
    STAGE_FILTER[] = stage_filter === nothing ? nothing : String(stage_filter)
    return nothing
end

@inline is_enabled() = ENABLED[]

@inline function next_seq!()
    # atomic_add! already returns the incremented value for Atomic{UInt64}.
    return Threads.atomic_add!(NEXT_SEQ, UInt64(1))
end

function set_seq!(buf, seq::UInt64)
    lock(LOG_LOCK) do
        if seq == 0
            if haskey(BUFFER_SEQ, buf)
                delete!(BUFFER_SEQ, buf)
            end
        else
            BUFFER_SEQ[buf] = seq
        end
    end
    return seq == 0 ? UInt64(0) : seq
end

function get_seq(buf)
    lock(LOG_LOCK) do
        return get(BUFFER_SEQ, buf, UInt64(0))
    end
end

function inherit_seq!(src_buf, dst_buf)
    seq = get_seq(src_buf)
    set_seq!(dst_buf, seq)
    return seq
end

@inline function stage_allowed(stage::AbstractString)
    filter = STAGE_FILTER[]
    return filter === nothing || stage == filter
end

@inline function maybe_log_ok(stage::AbstractString, direction::AbstractString, seq::UInt64)
    if !LOG_OK[] || !stage_allowed(stage)
        return nothing
    end
    n = Threads.atomic_add!(EVENT_COUNT, UInt64(1)) + UInt64(1)
    if (n % LOG_INTERVAL[]) == 0
        AsyncLogger.log_async("SeqTrace[", stage, "] ", direction, " seq=", Int64(seq))
    end
    return nothing
end

function log_in!(stage::AbstractString, block, seq::UInt64; strict::Bool = true)
    if !is_enabled() || seq == 0 || !stage_allowed(stage)
        return nothing
    end
    lock(LOG_LOCK) do
        prev = get(LAST_IN, block, UInt64(0))
        if strict && prev != 0 && seq != prev + 1
            AsyncLogger.log_async("SeqTrace[", stage, "] in_mismatch expected=", Int64(prev + 1),
                                  " actual=", Int64(seq),
                                  " delta=", Int64(seq) - Int64(prev))
        else
            maybe_log_ok(stage, "in", seq)
        end
        LAST_IN[block] = seq
    end
    return nothing
end

function log_out!(stage::AbstractString, block, seq::UInt64; strict::Bool = true)
    if !is_enabled() || seq == 0 || !stage_allowed(stage)
        return nothing
    end
    lock(LOG_LOCK) do
        prev = get(LAST_OUT, block, UInt64(0))
        if strict && prev != 0 && seq != prev + 1
            AsyncLogger.log_async("SeqTrace[", stage, "] out_mismatch expected=", Int64(prev + 1),
                                  " actual=", Int64(seq),
                                  " delta=", Int64(seq) - Int64(prev))
        else
            maybe_log_ok(stage, "out", seq)
        end
        LAST_OUT[block] = seq
    end
    return nothing
end

end
