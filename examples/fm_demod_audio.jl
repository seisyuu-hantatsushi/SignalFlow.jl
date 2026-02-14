import SignalFlow
import SignalFlow.ADFMCOMMS2Src
import SignalFlow.LPF
import SignalFlow.WBFM
import SignalFlow.WBFMStereoDemod
import SignalFlow.AlsaSink

const ADC_SamplingRate   = 1_200_000 # A/D Sampling Rate 1.2MHz
const BandWidth          =   800_000 # Band width 800KHz
const Audio_SamplingRate =    48_000 # Audio Sampling rate

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
    device = "default"

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
        elseif a == "-d" || a == "--device"
            i += 1
            i > length(args) && error("Missing value for $a")
            device = args[i]
        else
            error("Unknown argument: $a")
        end
        i += 1
    end
    carrier === nothing && error("Carrier frequency is required. Use -c/--carrierFreq (e.g. 77.8M).")
    return carrier, uri, device
end

using ADFMCOMMS2

function main()
    carrier, uri, device = parse_args(ARGS)

    if uri === nothing
        uri = ADFMCOMMS2.scan("ip")[1]
        if isempty(uri)
            error("No SDR URI found via scan(\"ip\"). Use -i/--uri (e.g. ip:192.168.10.90).")
        end
    end

    rfsrc = ADFMCOMMS2Src.open(ComplexF32,
                               uri,
                               UInt64(round(carrier)),
                               UInt32(ADC_SamplingRate),
                               UInt32(BandWidth))

    lpf_decimation = 2
    wbfm_decim = 2

    lpf = LPF.CreateLPF(ComplexF32, Float64(ADC_SamplingRate), 240e3;
                        filter_type = LPF.FIR,
                        decimation = lpf_decimation,
                        frame_size = 16384,
                        poolsize = 32)

    wbfm = WBFM.CreateWBFM(ComplexF32, Float64(ADC_SamplingRate / lpf_decimation);
                           deviation = 75e3,
                           decimation = wbfm_decim,
                           frame_size = 16384,
                           poolsize = 32)

    stereo = WBFMStereoDemod.CreateWBFMStereoDemod(Float32, Float64(ADC_SamplingRate / (lpf_decimation * wbfm_decim));
                                                   output_rate = Audio_SamplingRate,
                                                   output_format = WBFMStereoDemod.Int16PCM,
                                                   deemphasis_us = 75.0,
                                                   frame_size = 16384,
                                                   poolsize = 32)

    sink = AlsaSink.CreateAlsaSink(device;
                                   ch = 2,
                                   sampleRate = Audio_SamplingRate,
                                   bufferTime = 500_000,
                                   periodTime = 100_000,
                                   enable_clock_correction = true,
                                   target_delay_frames = 0,
                                   max_ppm = 200.0,
                                   frame_size = 16384,
                                   poolsize = 32,
                                   enable_stats_view = true)

    SignalFlow.append_block!(rfsrc, lpf)
    SignalFlow.append_block!(lpf, wbfm)
    SignalFlow.append_block!(wbfm, stereo)
    SignalFlow.append_block!(stereo, sink)

    println("Press Ctrl-C to stop.")
    try
        wait(Condition())
    catch e
        if !(e isa InterruptException)
            rethrow()
        end
    end

    AlsaSink.stop!(sink)
end

main()
