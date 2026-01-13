module SignalFlow

abstract type SignalFlowBlock end
function input! end

include("ADFMCOMMS2Src.jl")
include("FFTView.jl")

function append_block!(src::SignalFlowBlock, sink::SignalFlowBlock)
    put!(src.new_sinks, sink)
end

end
