
import SignalFlow
import SignalFlow.ADFMCOMMS2Src
import SignalFlow.BandSNREstimator
import SignalFlow.FFTView
import SignalFlow.LPF
import SignalFlow.WBFM
import SignalFlow.WBFMStereoDemod
import SignalFlow.WaveformView

const ADC_SamplingRate   = 1_200_000 # A/D Sampling Rate 1.2MHz
const BandWidth          =   800_000 # Band width 800KHz
const Audio_SamplingRate =    48_000 # Demodulated Audio Sampling rate

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
        else
            error("Unknown argument: $a")
        end
        i += 1
    end
    carrier === nothing && error("Carrier frequency is required. Use -c/--carrierFreq (e.g. 77.8M).")
    return carrier, uri
end

using ADFMCOMMS2

function main()

    lpf_decimation = 2
    carrier, uri = parse_args(ARGS)

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

    snr = BandSNREstimator.CreateBandSNREstimator(; samplerate = ADC_SamplingRate,
                                                  fft_size = 4096,
                                                  signal_band = (-80_000.0, 80_000.0),
                                                  noise_bands = [(-400_000.0, -300_000.0), (300_000.0, 400_000.0)],
                                                  enable_stats = true,
                                                  stats_avg_len = 20,
                                                  log_stats = true,
                                                  log_interval = 1.0)

    view1 = FFTView.CreateView(ComplexF32,
                               UInt64(ADC_SamplingRate),
                               UInt64(8196),
                               FFTView.Hann;
                               title = "Before LPF",
                               fmin = -BandWidth / 2,
                               fmax = BandWidth / 2)

    view2 = FFTView.CreateView(ComplexF32,
                               UInt64(ADC_SamplingRate),
                               UInt64(8196),
                               FFTView.Hann;
                               title = "After LPF",
                               fmin = -BandWidth / 2,
                               fmax = BandWidth / 2)

    view3 = FFTView.CreateView(Float32,
                               UInt64(ADC_SamplingRate/lpf_decimation),
                               UInt64(8196),
                               FFTView.Hann;
                               title = "WBFM Demod (FFT)",
                               fmin = -100_000.0,
                               fmax = 100_000.0)

    lpf = LPF.CreateLPF(ComplexF32, Float64(ADC_SamplingRate), 240e3;
                        filter_type = LPF.FIR,
                        decimation = lpf_decimation,
                        frame_size = 16384,
                        poolsize = 32) # 1.2MS/sec -> 600KS/sec
    wbfm = WBFM.CreateWBFM(ComplexF32, Float64(ADC_SamplingRate/lpf_decimation);
                           deviation = 75e3,
                           frame_size = 16384,
                           poolsize = 32)
    stereo = WBFMStereoDemod.CreateWBFMStereoDemod(Float32, Float64(ADC_SamplingRate/lpf_decimation);
                                                   output_rate = Audio_SamplingRate,
                                                   output_format = WBFMStereoDemod.Float32PCM,
                                                   deemphasis_us = 75.0,
                                                   frame_size = 16384,
                                                   poolsize = 32)
    waveform = WaveformView.CreateView(Float32, UInt64(Audio_SamplingRate);
                                        frame_size = 16384,
                                        channels = 2,
                                        title = "Audio Waveform")

    SignalFlow.append_block!(rfsrc, lpf)
    SignalFlow.append_block!(rfsrc, snr)
    SignalFlow.append_block!(rfsrc, view1)
    SignalFlow.append_block!(lpf, view2)
    SignalFlow.append_block!(lpf, wbfm)
    SignalFlow.append_block!(wbfm, view3)
    SignalFlow.append_block!(wbfm, stereo)
    SignalFlow.append_block!(stereo, waveform)
    
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
