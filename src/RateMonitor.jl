module RateMonitor

import ..SignalFlowBlock
import ..input!

mutable struct RateMonitorContext{T} <: SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    label::String
    report_interval::Float64
    last_time::Float64
    last_samples::Int
    total_samples::Int
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function CreateRateMonitor(::Type{T};
                           label::AbstractString = "RateMonitor",
                           report_interval::Real = 1.0) where {T}
    report_interval <= 0 && error("RateMonitor: report_interval must be positive.")
    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    now = time()
    ctx = RateMonitorContext{T}(Base.Threads.Atomic{Bool}(true),
                                String(label),
                                Float64(report_interval),
                                now,
                                0,
                                0,
                                new_sinks,
                                sinks)
    return ctx
end

function input!(context::RateMonitorContext{T}, samples::AbstractVector{T}, samples_size::Int) where {T}
    if !context.running[] || samples_size <= 0
        return false
    end

    actual_size = min(samples_size, length(samples))
    if actual_size <= 0
        return 0
    end

    context.total_samples += actual_size
    now = time()
    dt = now - context.last_time
    if dt >= context.report_interval
        delta = context.total_samples - context.last_samples
        rate = delta / dt
        println("RateMonitor[", context.label, "] rate=", round(rate; digits = 1), " S/sec")
        context.last_samples = context.total_samples
        context.last_time = now
    end

    while isready(context.new_sinks)
        push!(context.sinks, take!(context.new_sinks))
    end
    for sink in context.sinks
        input!(sink, samples, actual_size)
    end

    return samples_size
end

function stop!(context::RateMonitorContext)
    context.running[] = false
    return nothing
end

end
