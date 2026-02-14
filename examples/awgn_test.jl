import SignalFlow
import SignalFlow.AWGNInjector
import SignalFlow.BandSNREstimator

mutable struct SyntheticIQSourceContext <: SignalFlow.SignalFlowBlock
    running::Base.Threads.Atomic{Bool}
    samplerate::Float64
    frame_size::Int
    tone_freqs_hz::Vector{Float64}
    tone_amp::Float32
    noise_sigma::Float32
    phase::Vector{Float64}
    phase_step::Vector{Float64}
    outbuf::Vector{ComplexF32}
    worker::Union{Nothing,Task}
    new_sinks::Channel{SignalFlow.SignalFlowBlock}
    sinks::Vector{SignalFlow.SignalFlowBlock}
end

function CreateSyntheticIQSource(; samplerate::Real = 1_000_000,
                                 frame_size::Int = 4096,
                                 tone_freqs_hz::Vector{Float64} = [-100_000.0, -40_000.0, 40_000.0, 100_000.0],
                                 tone_amp::Real = 0.5,
                                 noise_sigma::Real = 0.0)
    samplerate <= 0 && error("SyntheticIQSource: samplerate must be > 0.")
    frame_size < 1 && error("SyntheticIQSource: frame_size must be >= 1.")
    isempty(tone_freqs_hz) && error("SyntheticIQSource: tone_freqs_hz must not be empty.")
    tone_amp <= 0 && error("SyntheticIQSource: tone_amp must be > 0.")
    noise_sigma < 0 && error("SyntheticIQSource: noise_sigma must be >= 0.")

    phase = zeros(Float64, length(tone_freqs_hz))
    phase_step = [2pi * f / Float64(samplerate) for f in tone_freqs_hz]
    ctx = SyntheticIQSourceContext(Base.Threads.Atomic{Bool}(true),
                                   Float64(samplerate),
                                   frame_size,
                                   tone_freqs_hz,
                                   Float32(tone_amp),
                                   Float32(noise_sigma),
                                   phase,
                                   phase_step,
                                   Vector{ComplexF32}(undef, frame_size),
                                   nothing,
                                   Channel{SignalFlow.SignalFlowBlock}(64),
                                   SignalFlow.SignalFlowBlock[])
    ctx.worker = Threads.@spawn source_task!(ctx)
    return ctx
end

function source_task!(context::SyntheticIQSourceContext)
    frame_sec = context.frame_size / context.samplerate
    next_deadline = time()
    try
        while context.running[]
            @inbounds for n in 1:context.frame_size
                v = ComplexF32(0, 0)
                for i in eachindex(context.phase)
                    ph = context.phase[i]
                    v += ComplexF32(cos(ph), sin(ph))
                    ph += context.phase_step[i]
                    if ph > pi
                        ph -= 2pi
                    elseif ph < -pi
                        ph += 2pi
                    end
                    context.phase[i] = ph
                end
                v *= context.tone_amp / length(context.phase)
                if context.noise_sigma > 0f0
                    v += ComplexF32(context.noise_sigma * randn(Float32),
                                    context.noise_sigma * randn(Float32))
                end
                context.outbuf[n] = v
            end

            while isready(context.new_sinks)
                push!(context.sinks, take!(context.new_sinks))
            end
            for sink in context.sinks
                SignalFlow.input!(sink, context.outbuf, context.frame_size)
            end

            next_deadline += frame_sec
            wait_s = next_deadline - time()
            if wait_s > 0
                sleep(wait_s)
            else
                next_deadline = time()
                yield()
            end
        end
    catch e
        if !(e isa InterruptException)
            println("SyntheticIQSource error: ", e)
        end
    end
    return nothing
end

function stop!(context::SyntheticIQSourceContext)
    context.running[] = false
    if context.worker !== nothing
        wait(context.worker)
    end
    return nothing
end

function parse_args(args::Vector{String})
    snr_db = 10.0
    samplerate = 1_000_000.0
    frame_size = 4096
    signal_bw_hz = 150_000.0
    noise_inner_hz = 200_000.0
    noise_outer_hz = 300_000.0
    src_noise_sigma = 0.0
    log_interval = 1.0

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--snr-db"
            i += 1
            i > length(args) && error("Missing value for --snr-db")
            snr_db = parse(Float64, args[i])
        elseif a == "--samplerate"
            i += 1
            i > length(args) && error("Missing value for --samplerate")
            samplerate = parse(Float64, args[i])
        elseif a == "--frame-size"
            i += 1
            i > length(args) && error("Missing value for --frame-size")
            frame_size = parse(Int, args[i])
        elseif a == "--signal-bw-hz"
            i += 1
            i > length(args) && error("Missing value for --signal-bw-hz")
            signal_bw_hz = parse(Float64, args[i])
        elseif a == "--noise-inner-hz"
            i += 1
            i > length(args) && error("Missing value for --noise-inner-hz")
            noise_inner_hz = parse(Float64, args[i])
        elseif a == "--noise-outer-hz"
            i += 1
            i > length(args) && error("Missing value for --noise-outer-hz")
            noise_outer_hz = parse(Float64, args[i])
        elseif a == "--src-noise-sigma"
            i += 1
            i > length(args) && error("Missing value for --src-noise-sigma")
            src_noise_sigma = parse(Float64, args[i])
        elseif a == "--log-interval"
            i += 1
            i > length(args) && error("Missing value for --log-interval")
            log_interval = parse(Float64, args[i])
        else
            error("Unknown argument: $a")
        end
        i += 1
    end

    frame_size < 1 && error("--frame-size must be >= 1")
    samplerate <= 0 && error("--samplerate must be > 0")
    signal_bw_hz <= 0 && error("--signal-bw-hz must be > 0")
    noise_inner_hz <= signal_bw_hz && error("--noise-inner-hz must be > --signal-bw-hz")
    noise_outer_hz <= noise_inner_hz && error("--noise-outer-hz must be > --noise-inner-hz")
    src_noise_sigma < 0 && error("--src-noise-sigma must be >= 0")
    log_interval <= 0 && error("--log-interval must be > 0")

    return snr_db, samplerate, frame_size, signal_bw_hz, noise_inner_hz, noise_outer_hz, src_noise_sigma, log_interval
end

function connect_blocks!(src, sink)
    src === nothing && return
    sink === nothing && return
    SignalFlow.append_block!(src, sink)
    return nothing
end

function main()
    prev_exit_on_sigint = Base.exit_on_sigint(false)
    restore_exit_on_sigint = prev_exit_on_sigint isa Bool
    snr_db, samplerate, frame_size, signal_bw_hz, noise_inner_hz, noise_outer_hz, src_noise_sigma, log_interval = parse_args(ARGS)

    println("AWGN test config: snr_db=", round(snr_db, digits = 2),
            " samplerate=", Int(round(samplerate)),
            " frame_size=", frame_size,
            " signal_bw_hz=", round(signal_bw_hz, digits = 1),
            " noise_band=(", round(noise_inner_hz, digits = 1), ", ", round(noise_outer_hz, digits = 1), ")")

    src = CreateSyntheticIQSource(; samplerate = samplerate,
                                  frame_size = frame_size,
                                  noise_sigma = src_noise_sigma)
    awgn = AWGNInjector.CreateAWGNInjector(ComplexF32; snr_db = snr_db,
                                           frame_size = frame_size,
                                           log_stats = true,
                                           log_interval = log_interval,
                                           poolsize = 16)
    snr_before = BandSNREstimator.CreateBandSNREstimator(; samplerate = samplerate,
                                                         fft_size = frame_size,
                                                         signal_band = (-signal_bw_hz, signal_bw_hz),
                                                         noise_bands = [(-noise_outer_hz, -noise_inner_hz),
                                                                        (noise_inner_hz, noise_outer_hz)],
                                                         label = "before_awgn",
                                                         window = BandSNREstimator.Hann,
                                                         enable_stats = false,
                                                         stats_interval = log_interval,
                                                         log_stats = true,
                                                         log_interval = log_interval,
                                                         poolsize = 16)
    snr_after = BandSNREstimator.CreateBandSNREstimator(; samplerate = samplerate,
                                                        fft_size = frame_size,
                                                        signal_band = (-signal_bw_hz, signal_bw_hz),
                                                        noise_bands = [(-noise_outer_hz, -noise_inner_hz),
                                                                       (noise_inner_hz, noise_outer_hz)],
                                                        label = "after_awgn",
                                                        window = BandSNREstimator.Hann,
                                                        enable_stats = false,
                                                        stats_interval = log_interval,
                                                        log_stats = true,
                                                        log_interval = log_interval,
                                                        poolsize = 16)

    SignalFlow.reset_flow_graph!()
    connect_blocks!(src, snr_before)
    connect_blocks!(src, awgn)
    connect_blocks!(awgn, snr_after)

    println("Press Ctrl-C to stop.")
    interrupted = false
    try
        while true
            sleep(1.0)
        end
    catch e
        if e isa InterruptException
            interrupted = true
            println("Interrupt received. Stopping blocks...")
        else
            rethrow()
        end
    finally
        shutdown_once = function ()
            try
                SignalFlow.stop_flow_graph!(timeout_sec = 0.5, clear_graph = true)
            catch e
                if !(e isa InterruptException)
                    println("shutdown warning: ", e)
                end
            end
            try
                SignalFlow.AsyncLogger.stop_default_logger!()
            catch e
                if !(e isa InterruptException)
                    println("shutdown warning (async logger): ", e)
                end
            end
        end

        shutdown_done = false
        for _ in 1:2
            try
                Base.disable_sigint() do
                    shutdown_once()
                end
                shutdown_done = true
                break
            catch e
                if !(e isa InterruptException)
                    rethrow()
                end
            end
        end
        if !shutdown_done
            try
                shutdown_once()
            catch
            end
        end
        if restore_exit_on_sigint
            Base.exit_on_sigint(prev_exit_on_sigint)
        end
        if interrupted
            println("Shutdown complete.")
            exit(130)
        end
    end
end

main()
