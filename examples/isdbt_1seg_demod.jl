import SignalFlow
import SignalFlow.ADFMCOMMS2Src
import SignalFlow.BandSNREstimator
import SignalFlow.ConstellationView
import SignalFlow.FFTBlock
import SignalFlow.FFTView
import SignalFlow.ISDBTPilotEqualizer
import SignalFlow.ISDBT1SegSymbolSync

const DEFAULT_SAMPLERATE = 8_000_000  # 8 Msps
const DEFAULT_BANDWIDTH  =   800_000  # 800 kHz (1seg focus)

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
    samplerate = DEFAULT_SAMPLERATE
    bandwidth = DEFAULT_BANDWIDTH
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
        elseif a == "-s" || a == "--samplerate"
            i += 1
            i > length(args) && error("Missing value for $a")
            samplerate = parse_si_hz(args[i])
        elseif a == "-b" || a == "--bandwidth"
            i += 1
            i > length(args) && error("Missing value for $a")
            bandwidth = parse_si_hz(args[i])
        else
            error("Unknown argument: $a")
        end
        i += 1
    end
    carrier === nothing && error("Carrier frequency is required. Use -c/--carrierFreq (e.g. 473.142857M).")
    return carrier, uri, Int(round(samplerate)), Int(round(bandwidth))
end

using ADFMCOMMS2

function main()
    carrier, uri, samplerate, bandwidth = parse_args(ARGS)
    if uri === nothing
        uri = ADFMCOMMS2.scan("ip")[1]
        if isempty(uri)
            error("No SDR URI found via scan(\"ip\"). Use -i/--uri (e.g. ip:192.168.10.90).")
        end
    end

    rfsrc = ADFMCOMMS2Src.open(ComplexF32,
                               uri,
                               UInt64(round(carrier)),
                               UInt32(samplerate),
                               UInt32(bandwidth))

    view = FFTView.CreateView(ComplexF32,
                              UInt64(samplerate),
                              UInt64(4096),
                              FFTView.Hann;
                              title = "ISDB-T 1seg Spectrum",
                              fmin = -bandwidth / 2,
                              fmax = bandwidth / 2,
                              avg_enabled = true,
                              avg_time_s = 0.5,
                              tick_step = 100_000.0,
                              label_step = 100_000.0)
    snr = BandSNREstimator.CreateBandSNREstimator(; samplerate = samplerate,
                                                  fft_size = 4096,
                                                  signal_band = (-200_000, 200_000),
                                                  noise_bands = [(-500_000.0, -350_000.0), (350_000.0, 500_000.0)],
                                                  enable_stats = true,
                                                  stats_avg_len = 20,
                                                  log_stats = true,
                                                  log_interval = 1.0)
    sync = ISDBT1SegSymbolSync.CreateISDBT1SegSymbolSync(; input_samplerate = samplerate,
                                                         sync_samplerate = 8_000_000,
                                                         mode = 3,
                                                         gi = 1//8,
                                                         search_symbols = 10,
                                                         cfo_enabled = true,
                                                         cfo_alpha = 0.01,
                                                         enable_stats = true,
                                                         stats_avg_len = 20,
                                                         log_stats = true,
                                                         log_interval = 1.0)
    fft = FFTBlock.CreateFFTBlock(ComplexF32, 8192;
                                  window = FFTBlock.Rectangular,
                                  scale = FFTBlock.FFTScaleNone)
    pilot_eq = ISDBTPilotEqualizer.CreateISDBTPilotEqualizer(; nfft = 8192,
                                                             pilot_spacing = 12,
                                                             pilot_offset0 = 3,
                                                             pilot_offset_step = 3,
                                                             output_mode = 2)
    constellation = ConstellationView.CreateView(UInt64(samplerate);
                                                 frame_size = 2048,
                                                 axis_limit = 1.2,
                                                 title = "ISDB-T 1seg Constellation")

    SignalFlow.append_block!(rfsrc, view)
    SignalFlow.append_block!(rfsrc, snr)
    SignalFlow.append_block!(rfsrc, sync)
    SignalFlow.append_block!(sync, fft)
    SignalFlow.append_block!(fft, pilot_eq)
    SignalFlow.append_block!(pilot_eq, constellation)

    println("Press Ctrl-C to stop.")
    try
        wait(Condition())
    catch e
        if !(e isa InterruptException)
            rethrow()
        end
    end
end

main()
