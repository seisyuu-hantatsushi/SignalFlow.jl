module SignalFlow

abstract type SignalFlowBlock end
function input! end

include("ADFMCOMMS2Src.jl")
include("RingBuffers.jl")
include("ISDBTSymbolSync.jl")
include("ISDBT1SegSymbolSync.jl")
include("ISDBTPilotEqualizer.jl")
include("FFTBlock.jl")
include("WBFMStereoDemod.jl")
include("AlsaSink.jl")
include("WavSink.jl")
include("WaveformView.jl")
include("ConstellationView.jl")
include("RateMonitor.jl")
include("BandSNREstimator.jl")
include("FFTView.jl")
include("LPF.jl")
include("WBFM.jl")

function append_block!(src::SignalFlowBlock, sink::SignalFlowBlock)
    put!(src.new_sinks, sink)
end

end
