module RingBuffers

mutable struct FrameBuffer{T}
    store_size::Int
    buf::Vector{T}
end

function Base.show(io::IO, fb::FrameBuffer{T}) where {T}
    print(io, "FrameBuffer{", T, "}(store_size=", fb.store_size,
          ", len=", length(fb.buf), ")")
end

function FrameBuffer(::Type{T}, frame_size::Int) where {T}
    return FrameBuffer(0, Vector{T}(undef, frame_size))
end

mutable struct RingFrameBuffer{T}
    frame_size::Int
    bufs::Vector{FrameBuffer{T}}
    freeQ::Channel{Int}
    fullQ::Channel{Int}
end

function Base.show(io::IO, rb::RingFrameBuffer{T}) where {T}
    print(io, "RingFrameBuffer{", T, "}(frame_size=", rb.frame_size,
          ", poolsize=", length(rb.bufs), ")")
end

function RingFrameBuffer(::Type{T}, frame_size::Int, poolsize::Int) where {T}
    freeQ = Channel{Int}(poolsize)
    fullQ = Channel{Int}(poolsize)
    for i in 1:poolsize
        put!(freeQ, i)
    end
    return RingFrameBuffer(frame_size, [FrameBuffer(T, frame_size) for _ in 1:poolsize], freeQ, fullQ)
end

end
