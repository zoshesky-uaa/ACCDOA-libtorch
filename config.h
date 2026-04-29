#pragma once
#include <atomic>
// Time Window (Output) = 3s
struct SystemConfig {
	// Controllable parameters:
	size_t sample_rate; // Sample rate for audio capture (e.g., 16000 Hz)
	size_t fft_size; // FFT size for the STFT
	size_t mel_bins; // Number of Mel bands for the log-mel spectrogram
	size_t hop_length; // Hop length
	double target_res; // Target output resolution in second (i.e. 0.1s for 100ms)
	size_t batch_size; // Batch size for training
	size_t batch_amount; // Number of batches to process for training;
	int64_t se_count; // Maximum unique sound events for SED head
	int64_t track_count; // Maximum amount of overlapping events for DOAE head

	// Calculated/Constant parameters:
	size_t channels = 4; // Number of audio channels (e.g., 4 for first-order ambisonics)
	int64_t time_window = 3;
	int64_t patch_size = 16; // Patch size (P) (h x w kernel)
	int64_t patch_overlap = 6; // Patch overlap (O) 
	int64_t enc_layers = 12; // Encoder layers (L) 
	// --------------------------------
	// These are specific dimension for distilling from other transformer models:
	int64_t att_headers = 12; // Attention heads (h) : 12 
	int64_t embed_dim = 768; // (h x 64) 
	// --------------------------------
	size_t frame_time_seq = time_window * (sample_rate / hop_length); // Frames per time window 
	size_t frame_max = frame_time_seq * batch_size * batch_amount; //Simulation maximum length in frames
	int64_t conv_stride = patch_size - patch_overlap; //Convolution stride (S) : (P - O)
	size_t fft_bins = fft_size / 2 + 1; // Number of frequency bins from the FFT
	size_t history_size = fft_size - hop_length; // Number of samples that overlap between consecutive STFT frames
	int64_t feature_channels = 3; // (Mel, IV_X, IV_Y)
	int64_t space_dim = 2; // (X, Y)
	/*
	Temporal (time-features) Patches (n_t) : 29 (floor((T - P) / S) + 1)
	Frequency (mel-features) Patches (n_f) : 12 (floor((M - P) / S) + 1))
	Total Patches (n) = (n_t * n_f)
	*/
	int64_t n_t = (time_window * (sample_rate / hop_length) - patch_size) / conv_stride + 1;
	int64_t n_f = (mel_bins - patch_size) / conv_stride + 1;
	int64_t num_patches = n_t * n_f; // Total Patches (n) (n_t * n_f)
	int64_t t_prime = time_window / target_res;
	int64_t total_seq = t_prime + num_patches; // Total sequence length (seq) (t' + n)
	std::atomic<bool> on{ true }; // Control flag


	SystemConfig(size_t sample_rate, 
		size_t fft_size, 
		size_t mel_bins, 
		size_t hop_length, 
		double target_res, 
		size_t batch_size, 
		size_t batch_amount, 
		int64_t se_count, 
		int64_t track_count)
		: sample_rate(sample_rate),
		fft_size(fft_size),
		hop_length(hop_length),
		mel_bins(mel_bins),
		batch_size(batch_size),
		batch_amount(batch_amount),
		se_count(se_count),
		track_count(track_count),
		target_res(target_res) {
	}

	// The default constructor creates a approximately 6 minute runtime 
	SystemConfig()
		: sample_rate(16000),
		channels(4),
		fft_size(512),
		hop_length(160),
		target_res(0.1),
		mel_bins(128),
		batch_size(24), 
		batch_amount(5),
		se_count(5),
		track_count(3) {
	}
};