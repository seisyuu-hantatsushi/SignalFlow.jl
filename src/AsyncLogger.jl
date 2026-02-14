module AsyncLogger

mutable struct LoggerContext
    running::Base.Threads.Atomic{Bool}
    queue::Channel{Any}
    queue_lock::ReentrantLock
    worker::Union{Nothing, Task}
    drop_count::Base.Threads.Atomic{Int}
    drop_log_interval::Int
    flush_interval::Int
    line_count::Base.Threads.Atomic{Int}
end

const DEFAULT_LOGGER = Ref{Union{Nothing, LoggerContext}}(nothing)
const INIT_LOCK = ReentrantLock()
const MAX_ARG_LEN = 512
const MAX_LINE_LEN = 4096

@inline function _safe_str(x)
    if x isa AbstractString
        s = x
    elseif x isa Number || x isa Symbol || x isa Bool || x isa Char
        s = string(x)
    elseif x isa Exception
        s = sprint(showerror, x)
    else
        s = summary(x)
    end
    if ncodeunits(s) > MAX_ARG_LEN
        return s[1:MAX_ARG_LEN] * "...(truncated)"
    end
    return s
end

@inline function _format_line(args::Tuple)
    io = IOBuffer()
    for a in args
        print(io, _safe_str(a))
    end
    s = String(take!(io))
    if ncodeunits(s) > MAX_LINE_LEN
        return s[1:MAX_LINE_LEN] * "...(truncated)"
    end
    return s
end

function create_logger(; capacity::Int = 4096, drop_log_interval::Int = 1000, flush_interval::Int = 64)
    capacity < 1 && error("AsyncLogger: capacity must be >= 1.")
    drop_log_interval < 1 && error("AsyncLogger: drop_log_interval must be >= 1.")
    flush_interval < 1 && error("AsyncLogger: flush_interval must be >= 1.")
    q = Channel{Any}(capacity)
    ctx = LoggerContext(Base.Threads.Atomic{Bool}(true),
                        q,
                        ReentrantLock(),
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
                print(stdout, _format_line(msg), '\n')
            else
                print(stdout, _safe_str(msg), '\n')
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

@inline function queue_tryput!(ctx::LoggerContext, item)
    # Emulate non-blocking put for Channel.
    # Guard check+put with a lock so producers cannot race into blocking put!.
    lock(ctx.queue_lock) do
        if isfull(ctx.queue)
            return false
        end
        put!(ctx.queue, item)
        return true
    end
end

@inline function log_async(msg::AbstractString; logger::Union{Nothing,LoggerContext} = nothing)
    ctx = logger === nothing ? default_logger() : logger
    if !ctx.running[]
        return false
    end
    ok = false
    try
        ok = queue_tryput!(ctx, String(msg))
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
        ok = queue_tryput!(ctx, args)
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
