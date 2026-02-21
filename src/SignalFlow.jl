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
include("SeqTrace.jl")
include("SeqCheckMonitor.jl")
include("ADFMCOMMS2Src.jl")
include("AWGNInjector.jl")
include("CFOPhaseInjector.jl")
include("OFDMSymbolImpairInjector.jl")
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
include("ISDBTEVMMonitor.jl")
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

function stop_block_gracefully!(blk::SignalFlowBlock; timeout_sec::Real = 0.5)
    # Phase-1: ask task loops to stop without blocking.
    if hasproperty(blk, :running)
        try
            blk.running[] = false
        catch
        end
    end

    mod = parentmodule(typeof(blk))
    stop_task = Threads.@spawn begin
        try
            if isdefined(mod, :stop!)
                f = getfield(mod, :stop!)
                if applicable(f, blk)
                    f(blk)
                    return
                end
            end
            if isdefined(mod, :close!)
                f = getfield(mod, :close!)
                if applicable(f, blk)
                    f(blk)
                    return
                end
            end
        catch
        end
    end

    status = timedwait(() -> istaskdone(stop_task), timeout_sec)
    return status != :timed_out
end

function stop_flow_graph!(; timeout_sec::Real = 0.5, clear_graph::Bool = true)
    nodes, edges = flow_graph_snapshot()
    indeg = IdDict{SignalFlowBlock, Int}()
    for node in nodes
        indeg[node] = 0
    end
    for (_, dsts) in edges
        for dst in dsts
            indeg[dst] = get(indeg, dst, 0) + 1
        end
    end

    queue = SignalFlowBlock[node for node in nodes if get(indeg, node, 0) == 0]
    order = SignalFlowBlock[]
    while !isempty(queue)
        node = popfirst!(queue)
        push!(order, node)
        for dst in get(edges, node, SignalFlowBlock[])
            d = get(indeg, dst, 0) - 1
            indeg[dst] = d
            if d == 0
                push!(queue, dst)
            end
        end
    end
    if length(order) != length(nodes)
        # Fallback for cycles/incomplete graph: deterministic stop order.
        order = copy(nodes)
    end

    # Phase-1 broadcast: every running flag goes false first.
    for blk in nodes
        if hasproperty(blk, :running)
            try
                blk.running[] = false
            catch
            end
        end
    end

    # Phase-2: stop downstream -> upstream.
    for blk in reverse(order)
        local ok = false
        try
            ok = stop_block_gracefully!(blk; timeout_sec = timeout_sec)
        catch e
            if !(e isa InterruptException)
                @warn "shutdown warning" block = typeof(blk) error = e
            end
        end
        if !ok
            @warn "shutdown timeout" block = typeof(blk)
        end
    end

    if clear_graph
        reset_flow_graph!()
    end
    return nothing
end

end
