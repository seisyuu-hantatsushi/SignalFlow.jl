import SignalFlow
import SignalFlow.ADFMCOMMS2Src
import SignalFlow.FFTBlock
import SignalFlow.FFTView
import SignalFlow.ISDBTSymbolSync
import SignalFlow.SeqCheckMonitor
import SignalFlow.SignalStatsMonitor

const ADC_SamplingRate = 8_000_000  # 8 Msps
const BandWidth = 7_000_000         # 7 MHz
const GI_Ratio = 1 // 8
const OFDM_NFFT = 8064
const OFDM_NCP = Int(round(OFDM_NFFT * Float64(GI_Ratio)))
const BlockPool = 64
const SyncFrameSize = 131072

function parse_si_hz(s::String)
    m = match(r"^([+-]?[0-9]*\.?[0-9]+)([kKmMgGtTpP]?)$", s)
    m === nothing && error("Invalid frequency: $s")
    value = parse(Float64, m.captures[1])
    suffix = m.captures[2]
    scale = suffix == "" ? 1.0 :
        suffix in ("k", "K") ? 1e3 :
        suffix in ("M", "m") ? 1e6 :
        suffix in ("G", "g") ? 1e9 :
        suffix in ("T", "t") ? 1e12 :
        suffix in ("P", "p") ? 1e15 : 1.0
    return value * scale
end

function parse_args(args)
    carrier = nothing
    uri = nothing
    diag = false
    show_fft = false
    src_poolsize = 768
    src_dispatch_burst = 32
    src_drop_backpressure = true
    fft_perf_interval = 0
    seq_trace = false
    seq_trace_log_interval = 200
    seq_trace_stage = nothing
    no_seqcheck = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "-c" || a == "--carrierFreq"
            i += 1
            i > length(args) && error("Missing value for $a")
            carrier = parse_si_hz(args[i])
        elseif a == "-i" || a == "--uri"
            i += 1
            i > length(args) && error("Missing value for $a")
            uri = args[i]
        elseif a == "--diag"
            diag = true
        elseif a == "--show-fft"
            show_fft = true
        elseif a == "--src-poolsize"
            i += 1
            i > length(args) && error("Missing value for $a")
            src_poolsize = parse(Int, args[i])
        elseif a == "--src-dispatch-burst"
            i += 1
            i > length(args) && error("Missing value for $a")
            src_dispatch_burst = parse(Int, args[i])
        elseif a == "--src-no-drop-backpressure"
            src_drop_backpressure = false
        elseif a == "--fft-perf-interval"
            i += 1
            i > length(args) && error("Missing value for $a")
            fft_perf_interval = parse(Int, args[i])
        elseif a == "--seq-trace"
            seq_trace = true
        elseif a == "--seq-trace-log-interval"
            i += 1
            i > length(args) && error("Missing value for $a")
            seq_trace_log_interval = parse(Int, args[i])
        elseif a == "--seq-trace-stage"
            i += 1
            i > length(args) && error("Missing value for $a")
            seq_trace_stage = args[i]
        elseif a == "--no-seqcheck"
            no_seqcheck = true
        else
            error("Unknown argument: $a")
        end
        i += 1
    end
    carrier === nothing && error("Carrier frequency is required. Use -c/--carrierFreq (e.g. 473.142857M).")
    src_poolsize < 1 && error("--src-poolsize must be >= 1")
    src_dispatch_burst < 1 && error("--src-dispatch-burst must be >= 1")
    fft_perf_interval < 0 && error("--fft-perf-interval must be >= 0")
    seq_trace_log_interval < 1 && error("--seq-trace-log-interval must be >= 1")
    return carrier, uri, diag, show_fft, src_poolsize, src_dispatch_burst, src_drop_backpressure,
           fft_perf_interval, seq_trace, seq_trace_log_interval, seq_trace_stage, no_seqcheck
end

using ADFMCOMMS2

function connect_blocks!(src, sink)
    src === nothing && return
    sink === nothing && return
    SignalFlow.append_block!(src, sink)
    return
end

function main()
    prev_exit_on_sigint = Base.exit_on_sigint(false)
    restore_exit_on_sigint = prev_exit_on_sigint isa Bool
    carrier, uri, diag, show_fft, src_poolsize, src_dispatch_burst, src_drop_backpressure,
        fft_perf_interval, seq_trace, seq_trace_log_interval, seq_trace_stage, no_seqcheck = parse_args(ARGS)

    # Miss-only logging to avoid adding load from OK logs.
    SignalFlow.SeqTrace.configure!(enabled = seq_trace,
                                   log_interval = seq_trace_log_interval,
                                   log_ok = false,
                                   stage_filter = seq_trace_stage)
    seq_trace && println("SeqTrace enabled: log_interval=", seq_trace_log_interval,
                         " log_ok=false",
                         " stage=", seq_trace_stage === nothing ? "all" : seq_trace_stage)
    if fft_perf_interval > 0
        println("FFTBlock perf enabled: interval=", fft_perf_interval)
    end

    if uri === nothing
        uris = ADFMCOMMS2.scan("ip")
        if isempty(uris)
            error("No SDR URI found via scan(\"ip\"). Use -i/--uri (e.g. ip:192.168.10.90).")
        end
        uri = uris[1]
    end

    rfsrc = ADFMCOMMS2Src.open(ComplexF32,
                               uri,
                               UInt64(round(carrier)),
                               UInt32(ADC_SamplingRate),
                               UInt32(BandWidth);
                               poolsize = src_poolsize,
                               dispatch_burst = src_dispatch_burst,
                               drop_on_backpressure = src_drop_backpressure,
                               backpressure_log_interval = diag ? 800 : 200)

    sync = ISDBTSymbolSync.CreateISDBTSymbolSync(; mode = 3,
                                                 gi = GI_Ratio,
                                                 samplerate = ADC_SamplingRate,
                                                 nfft_override = OFDM_NFFT,
                                                 search_symbols = 4,
                                                 cp_step = 1,
                                                 search_window = 8,
                                                 full_search_interval = 4000,
                                                 offset_penalty = 0.012,
                                                 max_drift = 4,
                                                 lock_metric_thresh = 0.0034,
                                                 unlock_metric_thresh = 0.0007,
                                                 lock_confirm = 4,
                                                 unlock_confirm = 20,
                                                 stats_update_interval = 10,
                                                 cfo_enabled = true,
                                                 enable_stats = false,
                                                 log_stats = true,
                                                 poolsize = BlockPool,
                                                 frame_size = SyncFrameSize)

    fft = FFTBlock.CreateFFTBlock(ComplexF32, OFDM_NFFT;
                                  window = FFTBlock.Rectangular,
                                  scale = FFTBlock.FFTScaleSqrt,
                                  perf_log_interval = fft_perf_interval,
                                  poolsize = BlockPool)
    if no_seqcheck
        seq_check = nothing
    else
        seq_check = SeqCheckMonitor.CreateSeqCheckMonitor(; frame_size = OFDM_NFFT,
                                                          label = "FFT",
                                                          log_interval = 1.0,
                                                          log_ok = false,
                                                          forward = false)
    end

    if diag
        stats_sync = SignalStatsMonitor.CreateSignalStatsMonitor(; frame_size = OFDM_NFFT,
                                                                 label = "SymbolSync out",
                                                                 log_interval = 30.0)
        stats_fft = SignalStatsMonitor.CreateSignalStatsMonitor(; frame_size = OFDM_NFFT,
                                                                label = "FFT out",
                                                                log_interval = 30.0)
    else
        stats_sync = nothing
        stats_fft = nothing
    end

    if show_fft
        fft_view = FFTView.CreateView(ComplexF32,
                                      UInt64(ADC_SamplingRate),
                                      UInt64(4096),
                                      FFTView.Hann;
                                      frame_size = 4096,
                                      title = "ISDB-T FFT",
                                      update_interval = diag ? 60 : 5)
    else
        fft_view = nothing
    end

    SignalFlow.reset_flow_graph!()
    connect_blocks!(rfsrc, sync)
    if stats_sync !== nothing
        connect_blocks!(sync, stats_sync)
    end
    connect_blocks!(sync, fft)
    connect_blocks!(fft, seq_check)
    if stats_fft !== nothing
        connect_blocks!(fft, stats_fft)
    end
    if fft_view !== nothing
        connect_blocks!(fft, fft_view)
    end

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
                redirect_stderr(devnull) do
                    SignalFlow.stop_flow_graph!(timeout_sec = 0.5, clear_graph = true)
                end
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
