module SeqCheckMonitor

import ..SignalFlowBlock
import ..input!
import ..SeqTrace
import ..AsyncLogger

mutable struct SeqCheckMonitorContext <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    frame_size::Int
    label::String
    log_interval::Float64
    last_log_time::Float64
    last_seq::UInt64
    log_ok::Bool
    seq_q::Channel{UInt64}
    state_lock::ReentrantLock
    worker::Union{Nothing,Task}
end

function CreateSeqCheckMonitor(; frame_size::Int,
                               label::AbstractString = "SeqCheck",
                               log_interval::Real = 1.0,
                               log_ok::Bool = false,
                               forward::Bool = true,
                               poolsize::Int = 64)
    frame_size < 1 && error("SeqCheckMonitor: frame_size must be >= 1.")
    log_interval <= 0 && error("SeqCheckMonitor: log_interval must be > 0.")
    poolsize < 1 && error("SeqCheckMonitor: poolsize must be >= 1.")
    forward && error("SeqCheckMonitor: forward=true is no longer supported (monitor-only mode).")

    ctx = SeqCheckMonitorContext(Base.Threads.Atomic{Bool}(true),
                                 frame_size,
                                 String(label),
                                 Float64(log_interval),
                                 0.0,
                                 UInt64(0),
                                 log_ok,
                                 Channel{UInt64}(poolsize),
                                 ReentrantLock(),
                                 nothing)
    ctx.worker = Threads.@spawn task!(ctx)
    return ctx
end

function handle_seq!(context::SeqCheckMonitorContext, seq::UInt64)
    lock(context.state_lock) do
        if seq == 0
            now = time()
            if now - context.last_log_time >= context.log_interval
                AsyncLogger.log_async("SeqCheck[", context.label, "] missing seq")
                context.last_log_time = now
            end
        else
            if context.last_seq != 0 && seq != context.last_seq + 1
                AsyncLogger.log_async("SeqCheck[", context.label, "] jump expected=",
                                      Int64(context.last_seq + 1),
                                      " actual=", Int64(seq),
                                      " delta=", Int64(seq) - Int64(context.last_seq))
            elseif context.log_ok
                now = time()
                if now - context.last_log_time >= context.log_interval
                    AsyncLogger.log_async("SeqCheck[", context.label, "] ok seq=", Int64(seq))
                    context.last_log_time = now
                end
            end
            context.last_seq = seq
        end
    end
    return nothing
end

function task!(context::SeqCheckMonitorContext)
    try
        while context.running[] || isready(context.seq_q)
            if isready(context.seq_q)
                seq = take!(context.seq_q)
                handle_seq!(context, seq)
            else
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("SeqCheckMonitor error: ", e)
        end
    end
    return nothing
end

function input!(context::SeqCheckMonitorContext, samples::AbstractVector{ComplexF32}, samples_size::Int)
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size != context.frame_size
        return -1
    end

    if !SeqTrace.is_enabled()
        return true
    end

    seq = SeqTrace.get_seq(samples)
    put!(context.seq_q, seq)
    return true
end

function stop!(context::SeqCheckMonitorContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

end
