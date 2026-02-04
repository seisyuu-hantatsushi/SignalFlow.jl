module AsyncLogger

mutable struct LoggerContext
    running::Base.Threads.Atomic{Bool}
    queue::Channel{Any}
    worker::Union{Nothing, Task}
    drop_count::Base.Threads.Atomic{Int}
    drop_log_interval::Int
    flush_interval::Int
    line_count::Base.Threads.Atomic{Int}
end

const DEFAULT_LOGGER = Ref{Union{Nothing, LoggerContext}}(nothing)
const INIT_LOCK = ReentrantLock()

function create_logger(; capacity::Int = 4096, drop_log_interval::Int = 1000, flush_interval::Int = 64)
    capacity < 1 && error("AsyncLogger: capacity must be >= 1.")
    drop_log_interval < 1 && error("AsyncLogger: drop_log_interval must be >= 1.")
    flush_interval < 1 && error("AsyncLogger: flush_interval must be >= 1.")
    q = Channel{Any}(capacity)
    ctx = LoggerContext(Base.Threads.Atomic{Bool}(true),
                        q,
                        nothing,
                        Base.Threads.Atomic{Int}(0),
                        drop_log_interval,
                        flush_interval,
                        Base.Threads.Atomic{Int}(0))
    ctx.worker = Threads.@spawn worker_task!(ctx)
    return ctx
end

function worker_task!(context::LoggerContext)
    try
        for msg in context.queue
            if msg isa Tuple
                print(stdout, msg..., '\n')
            else
                print(stdout, msg, '\n')
            end
            n = Threads.atomic_add!(context.line_count, 1) + 1
            if (n % context.flush_interval) == 0
                flush(stdout)
            end
        end
        flush(stdout)
    catch e
        if !(e isa InterruptException || e isa InvalidStateException)
            print(stderr, "AsyncLogger worker error: ", e, '\n')
        end
    end
    return nothing
end

function default_logger()
    ctx = DEFAULT_LOGGER[]
    if ctx !== nothing
        return ctx::LoggerContext
    end
    lock(INIT_LOCK) do
        if DEFAULT_LOGGER[] === nothing
            DEFAULT_LOGGER[] = create_logger()
            atexit(stop_default_logger!)
        end
        return DEFAULT_LOGGER[]::LoggerContext
    end
end

@inline function queue_tryput!(q::Channel{Any}, item)
    # Julia 1.12 has no Base.tryput! for Channel; emulate best-effort enqueue.
    # If queue appears full, drop immediately instead of blocking producers.
    if Base.n_avail(q) >= getfield(q, :sz_max)
        return false
    end
    put!(q, item)
    return true
end

@inline function log_async(msg::AbstractString; logger::Union{Nothing,LoggerContext} = nothing)
    ctx = logger === nothing ? default_logger() : logger
    if !ctx.running[]
        return false
    end
    ok = false
    try
        ok = queue_tryput!(ctx.queue, String(msg))
    catch e
        if e isa InvalidStateException
            return false
        end
        rethrow()
    end
    if !ok
        n = Threads.atomic_add!(ctx.drop_count, 1) + 1
        if (n % ctx.drop_log_interval) == 0
            print(stderr, "AsyncLogger: dropped_messages=", n, '\n')
        end
    end
    return ok
end

@inline function log_async(args...; logger::Union{Nothing,LoggerContext} = nothing)
    ctx = logger === nothing ? default_logger() : logger
    if !ctx.running[]
        return false
    end
    ok = false
    try
        ok = queue_tryput!(ctx.queue, args)
    catch e
        if e isa InvalidStateException
            return false
        end
        rethrow()
    end
    if !ok
        n = Threads.atomic_add!(ctx.drop_count, 1) + 1
        if (n % ctx.drop_log_interval) == 0
            print(stderr, "AsyncLogger: dropped_messages=", n, '\n')
        end
    end
    return ok
end

function stop!(context::LoggerContext)
    context.running[] = false
    close(context.queue)
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

function stop_default_logger!()
    lock(INIT_LOCK) do
        if DEFAULT_LOGGER[] !== nothing
            stop!(DEFAULT_LOGGER[]::LoggerContext)
            DEFAULT_LOGGER[] = nothing
        end
    end
    return nothing
end

end
