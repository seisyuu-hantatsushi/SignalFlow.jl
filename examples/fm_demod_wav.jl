import SignalFlow
import SignalFlow.ADFMCOMMS2Src
import SignalFlow.LPF
import SignalFlow.WBFM
import SignalFlow.WBFMStereoDemod
import SignalFlow.WavSink

const ADC_SamplingRate   = 1_200_000 # A/D Sampling Rate 1.2MHz
const BandWidth          =   800_000 # Band width 800KHz

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
    outfile = "fm.wav"

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
        elseif a == "-o" || a == "--output"
            i += 1
            i > length(args) && error("Missing value for $a")
            outfile = args[i]
        else
            error("Unknown argument: $a")
        end
        i += 1
    end
    carrier === nothing && error("Carrier frequency is required. Use -c/--carrierFreq (e.g. 77.8M).")
    return carrier, uri, outfile
end

using ADFMCOMMS2

function main()
    carrier, uri, outfile = parse_args(ARGS)

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

    lpf = LPF.CreateLPF(ComplexF32, Float64(ADC_SamplingRate), 240e3;
                        filter_type = LPF.FIR,
                        frame_size = 16384,
                        poolsize = 32)
    wbfm_decim = 2
    wbfm = WBFM.CreateWBFM(ComplexF32, Float64(ADC_SamplingRate);
                           deviation = 75e3,
                           frame_size = 16384,
                           poolsize = 32,
                           decimation = wbfm_decim)
    stereo = WBFMStereoDemod.CreateWBFMStereoDemod(Float32, Float64(ADC_SamplingRate) / wbfm_decim;
                                                   output_rate = 48_000,
                                                   output_format = WBFMStereoDemod.Int16PCM,
                                                   frame_size = 16384,
                                                   poolsize = 32)
    sink = WavSink.CreateWavSink(Int16, outfile;
                                 samplerate = 48_000,
                                 channels = 2,
                                 format = WavSink.Int16PCM,
                                 frame_size = 16384,
                                 poolsize = 32)

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

    WavSink.stop!(sink)
end

main()
