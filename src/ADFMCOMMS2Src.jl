module ADFMCOMMS2Src

using ADFMCOMMS2

import ..SignalFlowBlock
import ..input!

mutable struct ADFMCOMMS2Src{T} <: SignalFlowBlock 
    running::Base.Threads.Atomic{Bool}
    adapter::ADFMCOMMS2.SDR_RxAdapter{T}
    task::Union{Nothing, Task}
    new_sinks::Channel{SignalFlowBlock}
    sinks::Vector{SignalFlowBlock}
end

function open(::Type{T}, uri::String, frequency::UInt64, samplerate::UInt32, bandwidth::UInt32) where {T}

    adapter = ADFMCOMMS2.SDR_RxAdapter(uri,
                                       frequency,
                                       samplerate,
                                       bandwidth,
                                       T)
    
    new_sinks = Channel{SignalFlowBlock}(4)
    sinks = Vector{SignalFlowBlock}()
    src = ADFMCOMMS2Src(Base.Threads.Atomic{Bool}(true),
                        adapter,
                        nothing,
                        new_sinks,
                        sinks)
    src.task = Threads.@spawn task!(src)
    return src
end

function close!(context::ADFMCOMMS2Src)

    if !context.running[]
        return nothing
    end

    context.running[] = false
    if context.task !== nothing
        Base.disable_sigint() do
            try
                wait(context.task)
            catch e
                if !(e isa InterruptException)
                    rethrow()
                end
            end
        end
    end

end

function task!(context::ADFMCOMMS2Src)
    total_recv_samples::UInt64 = 0
    prev_total_recv_samples::UInt64 = 0
    ADFMCOMMS2.start!(context.adapter)
    try
        recv_buffer = Vector{ComplexF32}(undef, ADFMCOMMS2.SamplingFrameSize(context.adapter))
        prev_time = now_time = time_ns()
        while context.running[]
            now_time = time_ns()

            recv_size = ADFMCOMMS2.recv!(context.adapter, recv_buffer)
            if recv_size < 0
                error("RF Receive Error")
            end
            total_recv_samples += recv_size

            if isready(context.new_sinks)
                push!(context.sinks, take!(context.new_sinks))
            end

            for sink in context.sinks
                input!(sink, recv_buffer, recv_size)
            end
            
            if now_time - prev_time >= 1_000_000_000
                diff_time = Float32(now_time - prev_time)/1000_000_000
                diff_samples = total_recv_samples - prev_total_recv_samples
                println("recv rate: ",Float32(diff_samples)/diff_time, "S/sec")
                prev_time = now_time
                prev_total_recv_samples = total_recv_samples
            end
        end
        
    catch e
        println(e)
    end
    ADFMCOMMS2.stop!(context.adapter)
end

end
