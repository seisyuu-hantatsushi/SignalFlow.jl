module SignalFlow

abstract type SignalFlowBlock end
function input! end

include("RingBuffers.jl")
include("ADFMCOMMS2Src.jl")
include("ISDBTPRBS.jl")
include("ISDBTSymbolSync.jl")
include("ISDBT1SegSymbolSync.jl")
include("ISDBTPilotEqualizer.jl")
include("ISDBTDataCarrierExtractor.jl")
include("ISDBTCPECorrector.jl")
include("ISDBTPhaseSlopeCorrector.jl")
include("ISDBTFrameSync.jl")
include("FFTBlock.jl")
include("BinPowerMonitor.jl")
include("PilotCorrelationMonitor.jl")
include("SignalStatsMonitor.jl")
include("GainBlock.jl")
include("ISDBTPilotExtractor.jl")
include("TMCCDBPSKDecoder.jl")
include("WBFMStereoDemod.jl")
include("AlsaSink.jl")
include("WavSink.jl")
include("IQFileSink.jl")
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
