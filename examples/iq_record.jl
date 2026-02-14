import SignalFlow
import SignalFlow.ADFMCOMMS2Src
import SignalFlow.IQFileSink

function parse_si_hz(s::String)
    m = match(r"^([+-]?[0-9]*\\.?[0-9]+)([kKmMgGtTpP]?)$", s)
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
    lo = nothing
    bw = nothing
    sr = nothing
    uri = nothing
    outfile = nothing
    duration = 0.0

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "-l" || a == "--lo"
            i += 1
            i > length(args) && error("Missing value for $a")
            lo = parse_si_hz(args[i])
        elseif a == "-b" || a == "--bandwidth"
            i += 1
            i > length(args) && error("Missing value for $a")
            bw = parse_si_hz(args[i])
        elseif a == "-s" || a == "--samplerate"
            i += 1
            i > length(args) && error("Missing value for $a")
            sr = parse_si_hz(args[i])
        elseif a == "-i" || a == "--uri"
            i += 1
            i > length(args) && error("Missing value for $a")
            uri = args[i]
        elseif a == "-o" || a == "--output"
            i += 1
            i > length(args) && error("Missing value for $a")
            outfile = args[i]
        elseif a == "-d" || a == "--duration"
            i += 1
            i > length(args) && error("Missing value for $a")
            duration = parse(Float64, args[i])
        else
            error("Unknown argument: $a")
        end
        i += 1
    end

    lo === nothing && error("LO is required. Use -l/--lo (e.g. 100M).")
    bw === nothing && error("Bandwidth is required. Use -b/--bandwidth (e.g. 2M).")
    sr === nothing && error("Sampling rate is required. Use -s/--samplerate (e.g. 2.4M).")
    outfile === nothing && error("Output file is required. Use -o/--output (e.g. iq.raw).")
    duration < 0 && error("Duration must be >= 0 seconds.")

    return lo, bw, sr, uri, outfile, duration
end

using ADFMCOMMS2

function main()
    lo, bw, sr, uri, outfile, duration = parse_args(ARGS)

    if uri === nothing
        uri = ADFMCOMMS2.scan("ip")[1]
        if isempty(uri)
            error("No SDR URI found via scan(\"ip\"). Use -i/--uri (e.g. ip:192.168.10.90).")
        end
    end

    rfsrc = ADFMCOMMS2Src.open(ComplexF32,
                               uri,
                               UInt64(round(lo)),
                               UInt32(round(sr)),
                               UInt32(round(bw)))

    sink = IQFileSink.CreateIQFileSink(ComplexF32, outfile;
                                       frame_size = 16384,
                                       poolsize = 32,
                                       flush_interval_frames = 32)

    SignalFlow.append_block!(rfsrc, sink)

    println("Recording IQ to: ", outfile)
    if duration > 0
        println("Duration: ", duration, " sec")
    else
        println("Press Ctrl-C to stop.")
    end

    stopping = Base.Threads.Atomic{Bool}(false)
    stop_all = function()
        if !stopping[]
            stopping[] = true
            IQFileSink.stop!(sink)
            ADFMCOMMS2Src.close!(rfsrc)
        end
        return nothing
    end

    timer_task = nothing
    if duration > 0
        timer_task = @async begin
            sleep(duration)
            stop_all()
        end
    end

    try
        wait(Condition())
    catch e
        if !(e isa InterruptException)
            rethrow()
        end
    end

    stop_all()
    timer_task !== nothing && wait(timer_task)
end

main()
