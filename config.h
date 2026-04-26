#pragma once
#include <atomic>

struct SystemConfig {
	size_t sample_rate; // Sample rate for audio capture (e.g., 16000 Hz)
	size_t channels; // Number of audio channels (e.g., 4 for first-order ambisonics)
	size_t fft_size; // FFT size for the STFT
	size_t hop_length; // Hop length for the STFT, typically fft_size / 2
	size_t fft_bins; // Number of FFT bins (fft_size / 2 + 1)
	size_t history_size; // Number of samples to keep in the history buffer (fft_size - hop_length)
	size_t mel_bins; // Number of Mel bands for the log-mel spectrogram
	size_t feature_dim; // Total feature dimension (mel_bins + 2 * fft_bins)
	size_t frame_max; // Simulation maximum length in frames
	std::atomic<bool> on;

	SystemConfig(size_t sr, size_t ch, size_t fft, size_t hop, size_t mel, size_t frame_max)
		: sample_rate(sr), channels(ch), fft_size(fft), hop_length(hop), 
		  fft_bins(fft / 2 + 1), history_size(fft - hop), mel_bins(mel), feature_dim(mel + 2 * (fft / 2 + 1)), 
		frame_max(frame_max), on(true) {
	}

	SystemConfig()
		: sample_rate(16000),
		channels(4),
		fft_size(512),
		hop_length(160),
		fft_bins(512 / 2 + 1),
		history_size(512 - 160),
		mel_bins(128),
		feature_dim(3 * 128),
		frame_max(50000),
		on(true) {
	}
};