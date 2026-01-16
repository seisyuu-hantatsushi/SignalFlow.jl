module SignalFlow

abstract type SignalFlowBlock end
function input! end

include("ADFMCOMMS2Src.jl")
include("RingBuffers.jl")
include("FFTView.jl")
include("LPF.jl")
include("WBFM.jl")

function append_block!(src::SignalFlowBlock, sink::SignalFlowBlock)
    put!(src.new_sinks, sink)
end

end
