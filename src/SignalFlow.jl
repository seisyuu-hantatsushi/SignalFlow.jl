module SignalFlow

abstract type SignalFlowBlock end
function input! end

const FLOW_GRAPH_LOCK = ReentrantLock()
const FLOW_GRAPH_NODES = IdDict{SignalFlowBlock,Bool}()
const FLOW_GRAPH_EDGES = IdDict{SignalFlowBlock,Vector{SignalFlowBlock}}()

function reset_flow_graph!()
    lock(FLOW_GRAPH_LOCK) do
        empty!(FLOW_GRAPH_NODES)
        empty!(FLOW_GRAPH_EDGES)
    end
    return nothing
end

function flow_graph_snapshot()
    lock(FLOW_GRAPH_LOCK) do
        nodes = collect(keys(FLOW_GRAPH_NODES))
        edges = IdDict{SignalFlowBlock,Vector{SignalFlowBlock}}()
        for (src, dsts) in FLOW_GRAPH_EDGES
            edges[src] = copy(dsts)
        end
        return nodes, edges
    end
end

function register_flow_node!(blk::SignalFlowBlock)
    lock(FLOW_GRAPH_LOCK) do
        FLOW_GRAPH_NODES[blk] = true
    end
    return nothing
end

include("RingBuffers.jl")
include("AsyncLogger.jl")
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
    lock(FLOW_GRAPH_LOCK) do
        FLOW_GRAPH_NODES[src] = true
        FLOW_GRAPH_NODES[sink] = true
        push!(get!(FLOW_GRAPH_EDGES, src, SignalFlowBlock[]), sink)
    end
    put!(src.new_sinks, sink)
end

end
