module ISDBTPRBS

const SEG0_FLIP = Ref(false)

function set_seg0_flip!(enabled::Bool)
    SEG0_FLIP[] = enabled
    return nothing
end

function _bits_from_string(s::String)
    return [c == '1' ? 1 : 0 for c in collect(s)]
end

function _prbs_bits(init_bits::Vector{Int}, count::Int)
    n = length(init_bits)
    n == 11 || error("ISDBTPRBS: PRBS length must be 11.")
    bits = copy(init_bits)
    out = Vector{Int}(undef, count)
    @inbounds for i in 1:count
        out[i] = bits[end]
        fb = xor(bits[end], bits[end - 2])
        for k in n:-1:2
            bits[k] = bits[k - 1]
        end
        bits[1] = fb
    end
    return out
end

function mode3_initial_value(segment::Int)
    order = [11, 9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 10, 12]
    values = [
        "11111111111",
        "11011100101",
        "10010100000",
        "01110001001",
        "00100011001",
        "11100110110",
        "00100001011",
        "11100111101",
        "01101010011",
        "10111010010",
        "01100010010",
        "11110100101",
        "00010011100",
    ]
    for (i, seg) in enumerate(order)
        if seg == segment
            return _bits_from_string(values[i])
        end
    end
    error("ISDBTPRBS: unsupported segment: $segment")
end

function mode3_segment_prbs(segment::Int; carriers::Int = 432)
    return _prbs_bits(mode3_initial_value(segment), carriers)
end

function pilot_value_from_bit(bit::Int)
    # Table 3-17: Wi=1 => (-4/3,0), Wi=0 => (+4/3,0)
    return bit == 1 ? ComplexF32(-4/3, 0) : ComplexF32(4/3, 0)
end

function pilot_value_unit_from_bit(bit::Int)
    # Use unit-magnitude reference for post-equalized pilots.
    return bit == 1 ? ComplexF32(-1, 0) : ComplexF32(1, 0)
end

function seg0_carrier_to_bin(nfft::Int, carrier::Int, segment_carriers::Int = 432)
    half = segment_carriers ÷ 2
    carrier < 0 && return 0
    carrier >= segment_carriers && return 0
    c = SEG0_FLIP[] ? (segment_carriers - 1 - carrier) : carrier
    # Carrier numbering follows low-to-high frequency within the segment; no DC carrier.
    # Map 0..half-1 to negative bins just below DC (tail), half..end to positive bins above DC.
    if c < half
        return nfft - (half - 1 - c)
    end
    return 2 + (c - half)
end

function seg0_bin_to_carrier(nfft::Int, bin::Int, segment_carriers::Int = 432)
    half = segment_carriers ÷ 2
    pos_start = 2
    pos_end = 1 + half
    neg_start = nfft - (half - 1)
    neg_end = nfft
    if neg_start <= bin <= neg_end
        c = bin - neg_start
        return SEG0_FLIP[] ? (segment_carriers - 1 - c) : c
    end
    if pos_start <= bin <= pos_end
        c = half + (bin - pos_start)
        return SEG0_FLIP[] ? (segment_carriers - 1 - c) : c
    end
    return -1
end

end
