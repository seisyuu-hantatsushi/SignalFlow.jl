import SignalFlow
import SignalFlow.ADFMCOMMS2Src
import SignalFlow.BandSNREstimator
import SignalFlow.AWGNInjector
import SignalFlow.ConstellationView
import SignalFlow.BinPowerMonitor
import SignalFlow.PilotCorrelationMonitor
import SignalFlow.SignalStatsMonitor
import SignalFlow.ISDBTEVMMonitor
import SignalFlow.FFTBlock
import SignalFlow.FFTView
import SignalFlow.GainBlock
import SignalFlow.TMCCDBPSKDecoder
import SignalFlow.ISDBTPilotExtractor
import SignalFlow.ISDBTDataCarrierExtractor
import SignalFlow.ISDBTCPECorrector
import SignalFlow.ISDBTPhaseSlopeCorrector
import SignalFlow.ISDBTFrameSync
import SignalFlow.ISDBTPilotEqualizer
import SignalFlow.ISDBTSymbolSync
import SignalFlow.RateMonitor
import SignalFlow.WaveformView

const ADC_SamplingRate = 8_000_000  # 8 Msps
const BandWidth = 7_000_000         # 7 MHz
const DataBandWidth = 5_700_000     # 5.7 MHz
const SegmentBandwidth = DataBandWidth / 13
const SegmentCarriers = 432
const GI_Ratio = 1 // 8
const OFDM_NFFT = 8064
const OFDM_NCP = Int(round(OFDM_NFFT * Float64(GI_Ratio)))
const FrameSymbols = 204
const ExpectedFrameMs = 1000 * FrameSymbols * (OFDM_NFFT + OFDM_NCP) / ADC_SamplingRate
const BlockPool = 64
const PhasePool = 128
const SyncFrameSize = 131072
const StatsMonitorPool = 32

function parse_si_hz(s::String)
    m = match(r"^([+-]?[0-9]*\.?[0-9]+)([kKmMgGtTpP]?)$", s)
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

function parse_bitstring01(s::String)
    isempty(s) && error("bitstring must not be empty")
    bits = Vector{Int}(undef, length(s))
    @inbounds for i in eachindex(s)
        c = s[i]
        if c == '0'
            bits[i] = 0
        elseif c == '1'
            bits[i] = 1
        else
            error("bitstring must contain only 0/1: $s")
        end
    end
    return bits
end

function parse_args(args)
    carrier = nothing
    uri = nothing
    diag = false
    show_const = true
    show_fft = false
    show_wave = false
    show_sync = false
    show_pilots = false
    fft_gain = 1.0
    pilot_offset0 = 1
    seg0_flip = false
    pilot_eq_only = false
    pilot_temporal_alpha = 0.2
    tmcc_dbpsk = false
    tmcc_sync_word = nothing
    extractor_free_run = false
    no_cpe = false
    no_slope = false
    src_poolsize = 384
    src_dispatch_burst = 32
    src_drop_backpressure = true
    const_update_interval = 0
    const_drop_log_interval = 0
    seq_trace = false
    seq_trace_log_interval = 200
    seq_trace_stage = nothing
    evm = false
    evm_mod = "qpsk"
    evm_log_interval = 10.0
    awgn_snr_db = nothing
    awgn_log_interval = 10.0
    framesync_unlock_threshold = 0.25
    framesync_unlock_confirm = 20
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "-c" || a == "--carrierFreq"
            i += 1
            i > length(args) && error("Missing value for $a")
            carrier = parse_si_hz(args[i])
        elseif a == "-i" || a == "--uri"
            i += 1
            i > length(args) && error("Missing value for $a")
            uri = args[i]
        elseif a == "--diag"
            diag = true
        elseif a == "--no-const"
            show_const = false
        elseif a == "--show-fft"
            show_fft = true
        elseif a == "--show-wave"
            show_wave = true
        elseif a == "--show-sync"
            show_sync = true
        elseif a == "--show-pilots"
            show_pilots = true
        elseif a == "--fft-gain"
            i += 1
            i > length(args) && error("Missing value for $a")
            fft_gain = parse(Float64, args[i])
        elseif a == "--pilot-offset0"
            i += 1
            i > length(args) && error("Missing value for $a")
            pilot_offset0 = parse(Int, args[i])
        elseif a == "--seg0-flip"
            seg0_flip = true
        elseif a == "--pilot-eq-only"
            pilot_eq_only = true
        elseif a == "--pilot-temporal-alpha"
            i += 1
            i > length(args) && error("Missing value for $a")
            pilot_temporal_alpha = parse(Float64, args[i])
        elseif a == "--tmcc-dbpsk"
            tmcc_dbpsk = true
        elseif a == "--tmcc-sync-word"
            i += 1
            i > length(args) && error("Missing value for $a")
            tmcc_sync_word = parse_bitstring01(args[i])
        elseif a == "--extractor-free-run"
            extractor_free_run = true
        elseif a == "--no-cpe"
            no_cpe = true
        elseif a == "--no-slope"
            no_slope = true
        elseif a == "--src-poolsize"
            i += 1
            i > length(args) && error("Missing value for $a")
            src_poolsize = parse(Int, args[i])
        elseif a == "--src-dispatch-burst"
            i += 1
            i > length(args) && error("Missing value for $a")
            src_dispatch_burst = parse(Int, args[i])
        elseif a == "--src-no-drop-backpressure"
            src_drop_backpressure = false
        elseif a == "--const-update-interval"
            i += 1
            i > length(args) && error("Missing value for $a")
            const_update_interval = parse(Int, args[i])
        elseif a == "--const-drop-log-interval"
            i += 1
            i > length(args) && error("Missing value for $a")
            const_drop_log_interval = parse(Int, args[i])
        elseif a == "--seq-trace"
            seq_trace = true
        elseif a == "--seq-trace-log-interval"
            i += 1
            i > length(args) && error("Missing value for $a")
            seq_trace_log_interval = parse(Int, args[i])
        elseif a == "--seq-trace-stage"
            i += 1
            i > length(args) && error("Missing value for $a")
            seq_trace_stage = args[i]
        elseif a == "--evm"
            evm = true
        elseif a == "--evm-mod"
            i += 1
            i > length(args) && error("Missing value for $a")
            evm_mod = lowercase(args[i])
        elseif a == "--evm-log-interval"
            i += 1
            i > length(args) && error("Missing value for $a")
            evm_log_interval = parse(Float64, args[i])
        elseif a == "--awgn-snr-db"
            i += 1
            i > length(args) && error("Missing value for $a")
            awgn_snr_db = parse(Float64, args[i])
        elseif a == "--awgn-log-interval"
            i += 1
            i > length(args) && error("Missing value for $a")
            awgn_log_interval = parse(Float64, args[i])
        elseif a == "--framesync-unlock-threshold"
            i += 1
            i > length(args) && error("Missing value for $a")
            framesync_unlock_threshold = parse(Float64, args[i])
        elseif a == "--framesync-unlock-confirm"
            i += 1
            i > length(args) && error("Missing value for $a")
            framesync_unlock_confirm = parse(Int, args[i])
        else
            error("Unknown argument: $a")
        end
        i += 1
    end
    carrier === nothing && error("Carrier frequency is required. Use -c/--carrierFreq (e.g. 473.142857M).")
    src_poolsize < 1 && error("--src-poolsize must be >= 1")
    src_dispatch_burst < 1 && error("--src-dispatch-burst must be >= 1")
    const_update_interval < 0 && error("--const-update-interval must be >= 0")
    const_drop_log_interval < 0 && error("--const-drop-log-interval must be >= 0")
    seq_trace_log_interval < 1 && error("--seq-trace-log-interval must be >= 1")
    evm_log_interval <= 0 && error("--evm-log-interval must be > 0")
    awgn_log_interval <= 0 && error("--awgn-log-interval must be > 0")
    framesync_unlock_threshold <= 0 && error("--framesync-unlock-threshold must be > 0")
    framesync_unlock_confirm < 1 && error("--framesync-unlock-confirm must be >= 1")
    !(evm_mod in ("qpsk", "16qam", "64qam")) &&
        error("--evm-mod must be one of qpsk/16qam/64qam")
    awgn_snr_db !== nothing && !isfinite(awgn_snr_db) &&
        error("--awgn-snr-db must be finite")
    (pilot_temporal_alpha < 0 || pilot_temporal_alpha > 1) &&
        error("--pilot-temporal-alpha must be in [0, 1]")
    return carrier, uri, diag, show_const, show_fft, show_wave, show_sync, show_pilots, fft_gain, pilot_offset0, seg0_flip, pilot_eq_only, pilot_temporal_alpha, tmcc_dbpsk, tmcc_sync_word, extractor_free_run, no_cpe, no_slope, src_poolsize, src_dispatch_burst, src_drop_backpressure, const_update_interval, const_drop_log_interval, seq_trace, seq_trace_log_interval, seq_trace_stage, evm, evm_mod, evm_log_interval, awgn_snr_db, awgn_log_interval, framesync_unlock_threshold, framesync_unlock_confirm
end

using ADFMCOMMS2

function connect_blocks!(src, sink)
    src === nothing && return
    sink === nothing && return
    SignalFlow.append_block!(src, sink)
    return
end

function main()
    prev_exit_on_sigint = Base.exit_on_sigint(false)
    restore_exit_on_sigint = prev_exit_on_sigint isa Bool
    carrier, uri, diag, show_const, show_fft, show_wave, show_sync, show_pilots, fft_gain, pilot_offset0, seg0_flip, pilot_eq_only, pilot_temporal_alpha, tmcc_dbpsk, tmcc_sync_word, extractor_free_run, no_cpe, no_slope, src_poolsize, src_dispatch_burst, src_drop_backpressure, const_update_interval, const_drop_log_interval, seq_trace, seq_trace_log_interval, seq_trace_stage, evm, evm_mod, evm_log_interval, awgn_snr_db, awgn_log_interval, framesync_unlock_threshold, framesync_unlock_confirm = parse_args(ARGS)
    # Miss-only logging to avoid adding load from OK logs.
    SignalFlow.SeqTrace.configure!(enabled = seq_trace,
                                   log_interval = seq_trace_log_interval,
                                   log_ok = false,
                                   stage_filter = seq_trace_stage)
    seq_trace && println("SeqTrace enabled: log_interval=", seq_trace_log_interval,
                         " log_ok=false",
                         " stage=", seq_trace_stage === nothing ? "all" : seq_trace_stage)
    # Keep diagnostic logging lightweight for remote/headless runs.
    diag_log_interval = diag ? 30.0 : 1.0
    diag_tmcc_log_interval = diag ? 30.0 : 1.0
    diag_tmcc_dbpsk_log_interval = diag ? 20.0 : 0.5
    diag_view_update_interval = diag ? 60 : 5
    const_update_interval = const_update_interval > 0 ? const_update_interval : diag_view_update_interval
    const_drop_log_interval = const_drop_log_interval > 0 ? const_drop_log_interval : (diag ? 2000 : 500)
    const_frame_size = diag ? 1024 : 2048
    pilot_frame_size = diag ? 96 : 256
    fft_frame_size = diag ? 2048 : 4096
    wave_frame_size = diag ? 2048 : 4096
    enable_rate_monitors = !diag
    SignalFlow.ISDBTPRBS.set_seg0_flip!(seg0_flip)
    println("ISDBTPRBS seg0 mapping flip: ", seg0_flip)
    println("PilotEQ temporal_alpha: ", round(pilot_temporal_alpha, digits = 3))
    evm && println("EVM monitor: mod=", evm_mod, " interval=", round(evm_log_interval, digits = 2), "s")
    awgn_snr_db !== nothing && println("AWGNInjector enabled: snr_db=", round(awgn_snr_db, digits = 2),
                                       " log_interval=", round(awgn_log_interval, digits = 2), "s")
    println("FrameSync unlock_threshold: ", round(framesync_unlock_threshold, digits = 3),
            " unlock_confirm: ", framesync_unlock_confirm)
    println("OFDM params: nfft=", OFDM_NFFT, " cp=", OFDM_NCP,
            " frame_ms_expected=", round(ExpectedFrameMs, digits = 3))
    if pilot_eq_only
        no_cpe = true
        no_slope = true
        println("PilotEQ-only mode: CPE/Slope are disabled.")
    end
    if uri === nothing
        uri = ADFMCOMMS2.scan("ip")[1]
        if isempty(uri)
            error("No SDR URI found via scan(\"ip\"). Use -i/--uri (e.g. ip:192.168.10.90).")
        end
    end

    rfsrc = ADFMCOMMS2Src.open(ComplexF32,
                               uri,
                               UInt64(round(carrier)),
                               UInt32(ADC_SamplingRate),
                               UInt32(BandWidth);
                               poolsize = src_poolsize,
                               dispatch_burst = src_dispatch_burst,
                               drop_on_backpressure = src_drop_backpressure,
                               backpressure_log_interval = diag ? 800 : 200)
    src_frame_size = length(rfsrc.ringbuffer.bufs[1].buf)
    if awgn_snr_db !== nothing
        awgn = AWGNInjector.CreateAWGNInjector(ComplexF32; snr_db = awgn_snr_db,
                                               frame_size = src_frame_size,
                                               log_stats = true,
                                               log_interval = awgn_log_interval,
                                               poolsize = BlockPool)
    else
        awgn = nothing
    end

    sync = ISDBTSymbolSync.CreateISDBTSymbolSync(; mode = 3,
                                                 gi = GI_Ratio,
                                                 samplerate = ADC_SamplingRate,
                                                 nfft_override = OFDM_NFFT,
                                                 search_symbols = 4,
                                                 cp_step = 1,
                                                 search_window = 8,
                                                 full_search_interval = 4000,
                                                 offset_penalty = 0.012,
                                                 max_drift = 4,
                                                 lock_metric_thresh = 0.0034,
                                                 unlock_metric_thresh = 0.0007,
                                                 lock_confirm = 4,
                                                 unlock_confirm = 20,
                                                 stats_update_interval = 10,
                                                 cfo_enabled = true,
                                                 enable_stats = show_sync,
                                                 log_stats = true,
                                                 poolsize = BlockPool,
                                                 frame_size = SyncFrameSize)
    snr = BandSNREstimator.CreateBandSNREstimator(; samplerate = ADC_SamplingRate,
                                                  fft_size = 4096,
                                                  signal_band = (-2.85e6, 2.85e6),
                                                  noise_bands = [(-3.15e6, -2.85e6), (2.85e6, 3.15e6)],
                                                  window = BandSNREstimator.Hann,
                                                  enable_stats = false,
                                                  stats_interval = 2.0,
                                                  log_stats = true,
                                                  log_interval = 2.0)
    fft = FFTBlock.CreateFFTBlock(ComplexF32, OFDM_NFFT;
                                  window = FFTBlock.Rectangular,
                                  scale = FFTBlock.FFTScaleSqrt,
                                  poolsize = BlockPool)
    fft_gain_block = GainBlock.CreateGainBlock(ComplexF32; gain = fft_gain,
                                               frame_size = OFDM_NFFT,
                                               poolsize = BlockPool)
    # Mode 3 / coherent modulation / Segment 0 TMCC positions from ARIB STD-B31 Table 3-15 (c).
    tmcc_seg0 = [101, 131, 286, 349]
    # Mode 3 / coherent modulation / Segment 0 AC1 positions (Table 3-15 (c)).
    ac1_seg0 = [7, 89, 206, 209, 226, 244, 377, 407]
    # CP for coherent modulation is at carrier 0 within the segment.
    cp_seg0 = [0]
    exclude_seg0 = vcat(tmcc_seg0, ac1_seg0, cp_seg0)
    tmcc_bins = [SignalFlow.ISDBTPRBS.seg0_carrier_to_bin(OFDM_NFFT, k, SegmentCarriers) for k in tmcc_seg0]
    tmcc_bins_flip = [SignalFlow.ISDBTPRBS.seg0_carrier_to_bin(OFDM_NFFT,
                                                               SegmentCarriers - 1 - k,
                                                               SegmentCarriers) for k in tmcc_seg0]
    exclude_bins = [SignalFlow.ISDBTPRBS.seg0_carrier_to_bin(OFDM_NFFT, k, SegmentCarriers) for k in exclude_seg0]
    frame_sync_core = ISDBTFrameSync.ISDBTFrameSyncCoreConfig(OFDM_NFFT, FrameSymbols, tmcc_bins, BlockPool)
    frame_sync_lock = ISDBTFrameSync.ISDBTFrameSyncLockConfig(lock_threshold = 0.65,
                                                              unlock_threshold = framesync_unlock_threshold,
                                                              lock_confirm = 2,
                                                              unlock_confirm = framesync_unlock_confirm)
    frame_sync_cycle = ISDBTFrameSync.ISDBTFrameSyncCycleConfig(expected_frame_ms = ExpectedFrameMs,
                                                                cycle_outlier_ratio = 0.15)
    frame_sync_log = ISDBTFrameSync.ISDBTFrameSyncLogConfig(log_interval = diag ? 1200 : 200,
                                                            cycle_log_interval = diag ? 240 : 20,
                                                            input_gap_log_interval_sec = diag ? 10.0 : 1.0)
    frame_sync = ISDBTFrameSync.CreateISDBTFrameSync(frame_sync_core;
                                                     lock = frame_sync_lock,
                                                     cycle = frame_sync_cycle,
                                                     log = frame_sync_log)
    tmcc_power = BinPowerMonitor.CreateBinPowerMonitor(; nfft = OFDM_NFFT,
                                                       bins = tmcc_bins,
                                                       label = "TMCC bins",
                                                       log_interval = diag_tmcc_log_interval,
                                                       samplerate = ADC_SamplingRate,
                                                       band_limit_hz = SegmentBandwidth / 2)
    tmcc_power_flip = BinPowerMonitor.CreateBinPowerMonitor(; nfft = OFDM_NFFT,
                                                            bins = tmcc_bins_flip,
                                                            label = "TMCC bins (flip)",
                                                            log_interval = diag_tmcc_log_interval,
                                                            samplerate = ADC_SamplingRate,
                                                            band_limit_hz = SegmentBandwidth / 2)
    if tmcc_dbpsk
        tmcc_dbpsk_norm = TMCCDBPSKDecoder.CreateTMCCDBPSKDecoder(; nfft = OFDM_NFFT,
                                                                  bins = tmcc_bins,
                                                                  label = "norm",
                                                                  frame_symbols = FrameSymbols,
                                                                  symbol_index_ref = frame_sync.symbol_index_ref,
                                                                  skip_ref_symbol = true,
                                                                  ref_symbol_index = 1,
                                                                  log_interval = diag_tmcc_dbpsk_log_interval,
                                                                  sync_word = tmcc_sync_word,
                                                                  poolsize = BlockPool)
        tmcc_dbpsk_flip = TMCCDBPSKDecoder.CreateTMCCDBPSKDecoder(; nfft = OFDM_NFFT,
                                                                  bins = tmcc_bins_flip,
                                                                  label = "flip",
                                                                  frame_symbols = FrameSymbols,
                                                                  symbol_index_ref = frame_sync.symbol_index_ref,
                                                                  skip_ref_symbol = true,
                                                                  ref_symbol_index = 1,
                                                                  log_interval = diag_tmcc_dbpsk_log_interval,
                                                                  sync_word = tmcc_sync_word,
                                                                  poolsize = BlockPool)
    else
        tmcc_dbpsk_norm = nothing
        tmcc_dbpsk_flip = nothing
    end
    pilot_corr = PilotCorrelationMonitor.CreatePilotCorrelationMonitor(; nfft = OFDM_NFFT,
                                                                       pilot_spacing = 12,
                                                                       pilot_offset0 = pilot_offset0,
                                                                       pilot_offset_step = 3,
                                                                       segment_carriers = SegmentCarriers,
                                                                       segment_index = 0,
                                                                       label = "seg0",
                                                                       log_interval = diag_log_interval)
    pilot_eq = ISDBTPilotEqualizer.CreateISDBTPilotEqualizer(; nfft = OFDM_NFFT,
                                                             pilot_spacing = 12,
                                                             pilot_offset0 = pilot_offset0,
                                                             pilot_offset_step = 3,
                                                             output_mode = 2,
                                                             samplerate = ADC_SamplingRate,
                                                             band_limit_hz = SegmentBandwidth / 2,
                                                             segment_carriers = SegmentCarriers,
                                                             segment_index = 0,
                                                             auto_sp_phase = false,
                                                             symbol_index_ref = frame_sync.symbol_index_ref,
                                                             temporal_alpha = pilot_temporal_alpha,
                                                             log_stats = diag,
                                                             log_interval = diag_log_interval,
                                                             poolsize = BlockPool)
    if no_slope
        slope = nothing
    else
        slope = ISDBTPhaseSlopeCorrector.CreateISDBTPhaseSlopeCorrector(; nfft = OFDM_NFFT,
                                                                        pilot_spacing = 12,
                                                                        pilot_offset0 = pilot_offset0,
                                                                        pilot_offset_step = 3,
                                                                        samplerate = ADC_SamplingRate,
                                                                        band_limit_hz = SegmentBandwidth / 2,
                                                                        segment_carriers = SegmentCarriers,
                                                                        segment_index = 0,
                                                                        alpha = 0.1,
                                                                        max_slope_step = 0.0015,
                                                                        max_intercept_step_deg = 10.0,
                                                                        pilot_min_mag = 0.2,
                                                                        pilot_trim_ratio = 0.15,
                                                                        # Wider hysteresis + longer confirms to avoid gate chatter.
                                                                        max_fit_rms = 0.30,
                                                                        max_fit_rms_off = 0.45,
                                                                        min_used_pilots = 24,
                                                                        min_used_ratio = 0.65,
                                                                        update_confirm = 3,
                                                                        update_fail_confirm = 3,
                                                                        min_slope_step = 7.5e-5,
                                                                        min_intercept_step_deg = 0.4,
                                                                        auto_sp_phase = false,
                                                                        symbol_index_ref = frame_sync.symbol_index_ref,
                                                                        gap_freeze_ref = frame_sync.gap_freeze_ref,
                                                                        log_stats = diag,
                                                                        log_interval = diag_log_interval,
                                                                        poolsize = PhasePool)
    end
    if no_cpe
        cpe = nothing
    else
        cpe = ISDBTCPECorrector.CreateISDBTCPECorrector(; nfft = OFDM_NFFT,
                                                        pilot_spacing = 12,
                                                        pilot_offset0 = pilot_offset0,
                                                        pilot_offset_step = 3,
                                                        samplerate = ADC_SamplingRate,
                                                        band_limit_hz = SegmentBandwidth / 2,
                                                        segment_carriers = SegmentCarriers,
                                                        segment_index = 0,
                                                        cpe_alpha = 0.05,
                                                        cpe_max_step_deg = 10.0,
                                                        pilot_min_mag = 0.2,
                                                        pilot_trim_ratio = 0.15,
                                                        # Keep gate stable under transient confidence dips.
                                                        min_update_conf = 0.30,
                                                        min_update_conf_off = 0.20,
                                                        conf_gain_floor = 0.05,
                                                        update_confirm = 3,
                                                        update_fail_confirm = 3,
                                                        min_phase_step_deg = 0.4,
                                                        auto_sp_phase = false,
                                                        symbol_index_ref = frame_sync.symbol_index_ref,
                                                        gap_freeze_ref = frame_sync.gap_freeze_ref,
                                                        log_stats = diag,
                                                        log_interval = diag_log_interval,
                                                        poolsize = PhasePool)
    end
    data_symbol_ref = extractor_free_run ? nothing : frame_sync.symbol_index_ref
    data_carriers = ISDBTDataCarrierExtractor.CreateISDBTDataCarrierExtractor(; nfft = OFDM_NFFT,
                                                                              samplerate = ADC_SamplingRate,
                                                                              band_limit_hz = SegmentBandwidth / 2,
                                                                              pilot_spacing = 12,
                                                                              pilot_offset0 = pilot_offset0,
                                                                              pilot_offset_step = 3,
                                                                              exclude_dc = true,
                                                                              exclude_edge_bins = 0,
                                                                              tps_positions = Int[],
                                                                              exclude_carriers = exclude_seg0,
                                                                              segment_carriers = SegmentCarriers,
                                                                              segment_index = 0,
                                                                              # Keep extractor phase deterministic from frame sync.
                                                                              auto_sp_phase = false,
                                                                              log_stats = diag,
                                                                              log_interval = diag_log_interval,
                                                                              symbol_index_ref = data_symbol_ref,
                                                                              poolsize = BlockPool)
    if show_pilots
        pilot_extract = ISDBTPilotExtractor.CreateISDBTPilotExtractor(; nfft = OFDM_NFFT,
                                                                      pilot_spacing = 12,
                                                                      pilot_offset0 = pilot_offset0,
                                                                      pilot_offset_step = 3,
                                                                      segment_carriers = SegmentCarriers,
                                                                      segment_index = 0,
                                                                      auto_sp_phase = false,
                                                                      normalize = true,
                                                                      symbol_index_ref = frame_sync.symbol_index_ref,
                                                                      poolsize = BlockPool)
    else
        pilot_extract = nothing
    end
    if show_const
        constellation = ConstellationView.CreateView(UInt64(ADC_SamplingRate);
                                                     frame_size = const_frame_size,
                                                     axis_limit = 5.0,
                                                     title = "ISDB-T Constellation",
                                                     update_interval = const_update_interval,
                                                     drop_log_interval = const_drop_log_interval)
    else
        constellation = nothing
    end
    if show_pilots
        pilot_view = ConstellationView.CreateView(UInt64(ADC_SamplingRate);
                                                  frame_size = pilot_frame_size,
                                                  axis_limit = 3.0,
                                                  title = "ISDB-T Pilots",
                                                  update_interval = diag_view_update_interval)
    else
        pilot_view = nothing
    end
    if show_fft
        fft_view = FFTView.CreateView(UInt64(ADC_SamplingRate);
                                      frame_size = fft_frame_size,
                                      axis_limit = 1.0,
                                      title = "ISDB-T FFT",
                                      update_interval = diag_view_update_interval)
    else
        fft_view = nothing
    end
    if show_wave
        wave_view = WaveformView.CreateView(UInt64(ADC_SamplingRate);
                                            frame_size = wave_frame_size,
                                            axis_limit = 0.5,
                                            title = "ISDB-T Waveform",
                                            update_interval = diag_view_update_interval)
    else
        wave_view = nothing
    end
    if enable_rate_monitors
        mon_sync = RateMonitor.CreateRateMonitor(ComplexF32; label = "ISDBTSymbolSync in")
        mon_fft = RateMonitor.CreateRateMonitor(ComplexF32; label = "FFTBlock in")
        mon_pilot = RateMonitor.CreateRateMonitor(ComplexF32; label = "PilotEQ in")
    else
        mon_sync = nothing
        mon_fft = nothing
        mon_pilot = nothing
    end
    SignalFlow.reset_flow_graph!()

    connect_blocks!(rfsrc, snr)
    sync_src = rfsrc
    if awgn !== nothing
        connect_blocks!(rfsrc, awgn)
        sync_src = awgn
    end
    if mon_sync !== nothing
        connect_blocks!(sync_src, mon_sync)
        connect_blocks!(mon_sync, sync)
    else
        connect_blocks!(sync_src, sync)
    end
    if diag
        stats_sync = SignalStatsMonitor.CreateSignalStatsMonitor(; frame_size = OFDM_NFFT,
                                                                 label = "SymbolSync out",
                                                                 log_interval = diag_log_interval,
                                                                 poolsize = StatsMonitorPool)
    else
        stats_sync = nothing
    end
    stats_fft = nothing
    stats_piloteq = nothing
    if stats_sync !== nothing
        connect_blocks!(sync, stats_sync)
    end
    if wave_view !== nothing
        connect_blocks!(sync, wave_view)
    end
    # Keep FrameSync path short: sync -> fft -> frame_sync
    # (monitor is attached in parallel to avoid extra hop on the critical path).
    connect_blocks!(sync, fft)
    if mon_fft !== nothing
        connect_blocks!(sync, mon_fft)
    end
    # Register frame_sync first so FFT dispatch hits the timing-critical sink before
    # heavy downstream branches.
    connect_blocks!(fft, frame_sync)
    connect_blocks!(fft, fft_gain_block)
    connect_blocks!(fft_gain_block, tmcc_power)
    connect_blocks!(fft_gain_block, tmcc_power_flip)
    connect_blocks!(fft_gain_block, pilot_corr)
    if diag
        stats_fft = SignalStatsMonitor.CreateSignalStatsMonitor(; frame_size = OFDM_NFFT,
                                                                label = "FFT out",
                                                                log_interval = diag_log_interval,
                                                                poolsize = StatsMonitorPool)
        connect_blocks!(fft_gain_block, stats_fft)
    end
    if fft_view !== nothing
        connect_blocks!(fft_gain_block, fft_view)
    end
    if mon_pilot !== nothing
        connect_blocks!(fft_gain_block, mon_pilot)
        connect_blocks!(mon_pilot, pilot_eq)
    else
        connect_blocks!(fft_gain_block, pilot_eq)
    end
    if diag
        stats_piloteq = SignalStatsMonitor.CreateSignalStatsMonitor(; frame_size = OFDM_NFFT,
                                                                    label = "PilotEQ out",
                                                                    log_interval = diag_log_interval,
                                                                    poolsize = StatsMonitorPool)
        connect_blocks!(pilot_eq, stats_piloteq)
        pilot_corr_eq = PilotCorrelationMonitor.CreatePilotCorrelationMonitor(; nfft = OFDM_NFFT,
                                                                              pilot_spacing = 12,
                                                                              pilot_offset0 = pilot_offset0,
                                                                              pilot_offset_step = 3,
                                                                              segment_carriers = SegmentCarriers,
                                                                              segment_index = 0,
                                                                              label = "seg0_eq",
                                                                              log_interval = diag_log_interval)
        connect_blocks!(pilot_eq, pilot_corr_eq)
    end
    prev_block = pilot_eq
    if slope !== nothing
        connect_blocks!(prev_block, slope)
        if diag
            stats_slope = SignalStatsMonitor.CreateSignalStatsMonitor(; frame_size = OFDM_NFFT,
                                                                      label = "PhaseSlope out",
                                                                      log_interval = diag_log_interval,
                                                                      poolsize = StatsMonitorPool)
            connect_blocks!(slope, stats_slope)
        end
        prev_block = slope
    end
    if cpe !== nothing
        connect_blocks!(prev_block, cpe)
        if diag
            stats_cpe = SignalStatsMonitor.CreateSignalStatsMonitor(; frame_size = OFDM_NFFT,
                                                                    label = "CPE out",
                                                                    log_interval = diag_log_interval,
                                                                    poolsize = StatsMonitorPool)
            connect_blocks!(cpe, stats_cpe)
        end
        prev_block = cpe
    end
    # TMCC DBPSK is more reliable after pilot/slope/CPE correction than raw FFT branch.
    if tmcc_dbpsk_norm !== nothing
        connect_blocks!(prev_block, tmcc_dbpsk_norm)
        connect_blocks!(prev_block, tmcc_dbpsk_flip)
    end
    # Frame sync is driven from phase-independent FFT output with minimum hop count.
    # Data carriers must use equalized/corrected symbols while referencing frame_sync index.
    connect_blocks!(prev_block, data_carriers)
    if diag
        stats_data = SignalStatsMonitor.CreateSignalStatsMonitor(; frame_size = length(data_carriers.outbuf),
                                                                 label = "DataCarriers out",
                                                                 log_interval = diag_log_interval,
                                                                 poolsize = StatsMonitorPool)
        connect_blocks!(data_carriers, stats_data)
    end
    if evm
        evm_mon = ISDBTEVMMonitor.CreateISDBTEVMMonitor(; frame_size = length(data_carriers.outbuf),
                                                        modulation = evm_mod,
                                                        label = "DataCarriers",
                                                        log_interval = evm_log_interval,
                                                        poolsize = StatsMonitorPool)
        connect_blocks!(data_carriers, evm_mon)
    end
    if constellation !== nothing
        connect_blocks!(data_carriers, constellation)
    end
    if pilot_view !== nothing && pilot_extract !== nothing
        # PilotEQ単体検証がしやすいように、常にPilotEQ出力後のパイロットを表示する。
        connect_blocks!(pilot_eq, pilot_extract)
        connect_blocks!(pilot_extract, pilot_view)
    end

    println("Press Ctrl-C to stop.")
    interrupted = false
    try
        while true
            sleep(1.0)
        end
    catch e
        if e isa InterruptException
            interrupted = true
            println("Interrupt received. Stopping blocks...")
        else
            rethrow()
        end
    finally
        # Keep shutdown deterministic even when INT arrives during cleanup.
        shutdown_once = function ()
            try
                SignalFlow.stop_flow_graph!(timeout_sec = 0.5, clear_graph = true)
            catch e
                if !(e isa InterruptException)
                    println("shutdown warning: ", e)
                end
            end
            try
                SignalFlow.AsyncLogger.stop_default_logger!()
            catch e
                if !(e isa InterruptException)
                    println("shutdown warning (async logger): ", e)
                end
            end
        end

        # INT can race with cleanup; retry with SIGINT disabled first, then best effort.
        shutdown_done = false
        for _ in 1:2
            try
                Base.disable_sigint() do
                    shutdown_once()
                end
                shutdown_done = true
                break
            catch e
                if !(e isa InterruptException)
                    rethrow()
                end
            end
        end
        if !shutdown_done
            try
                shutdown_once()
            catch
            end
        end
        if restore_exit_on_sigint
            Base.exit_on_sigint(prev_exit_on_sigint)
        end
        if interrupted
            println("Shutdown complete.")
            exit(130)
        end
    end
end

main()
