#pragma once
#include <atomic>
#include <vector>
#include <optional>
#include <iostream>
#include <thread>
#include <string>
#include <cstdint>
#include <cmath>
#include <nlohmann/json.hpp>

enum ModelType : std::uint8_t {
	SED = 1,
	DOA = 5
};

enum class DatasetType : std::uint8_t {
	SED_FEATURES,
	DOA_FEATURES,
	SED_LABELS,
	DOA_LABELS
};

// Time Window (Output) = 3s
struct SystemConfig {
	// Controllable parameters:
	size_t sample_rate; // Sample rate for audio capture (e.g., 16000 Hz)
	size_t fft_size; // FFT size for the STFT
	size_t mel_bins; // Number of Mel bands for the log-mel spectrogram
	size_t hop_length; // Hop length
	double target_res; // Target output resolution in second (i.e. 0.1s for 100ms)
	size_t batch_size; // Batch size for training
	int64_t se_count; // Maximum unique sound events for SED head
	int64_t track_count; // Maximum amount of overlapping events for DOAE head

	// Calculated/Constant parameters:
	size_t epochs = 50; // Number of training epochs
	size_t warmup_epochs = 5; // Number of warmup epochs for learning rate scheduling
	size_t batch_amount = 5; // Number of batches to process for training;
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
	double input_frame_time = static_cast<double>(hop_length) / static_cast<double>(sample_rate); // Time per input frame (e.g., 0.01s for 10ms hop at 16kHz)
	size_t frame_time_seq = (static_cast<size_t>(time_window) * sample_rate) / hop_length; // Frames per time window 
	size_t frame_max = frame_time_seq * batch_size * batch_amount; //Simulation maximum length in frames
	int64_t conv_stride = patch_size - patch_overlap; //Convolution stride (S) : (P - O)
	size_t fft_bins = (fft_size / 2) + 1; // Number of frequency bins from the FFT
	size_t history_size = fft_size - hop_length; // Number of samples that overlap between consecutive STFT frames
	/*
	Temporal (time-features) Patches (n_t) : 29 (floor((T - P) / S) + 1)
	Frequency (mel-features) Patches (n_f) : 12 (floor((M - P) / S) + 1))
	Total Patches (n) = (n_t * n_f)
	*/
	int64_t t_prime = static_cast<int64_t>(std::llround(static_cast<double>(time_window) / target_res));
	size_t label_max = static_cast<size_t>(t_prime) * batch_size * batch_amount;
	size_t inference_amount = static_cast<size_t>(std::llround(target_res * (static_cast<double>(sample_rate) / static_cast<double>(hop_length)))); // Number of frames to infer on per inference step (e.g., 10 for 100ms)
	int64_t n_t = ((static_cast<int64_t>(frame_time_seq) - patch_size) / conv_stride) + 1;
    int64_t n_f = ((static_cast<int64_t>(mel_bins) - patch_size) / conv_stride) + 1;
    int64_t num_patches = n_t * n_f; // Total Patches (n) (n_t * n_f)
    int64_t total_seq = t_prime + num_patches; // Total sequence length (seq) (t' + n)
	/*
	SED Features (sed_featureset)
	Concept: 1-channel log-mel spectrogram.
	 read_buffer: [1, config.frame_time_seq, config.mel_bins] (e.g., [1, 300, 128])
	x_in: [config.batch_size, 1, config.frame_time_seq, config.mel_bins] (e.g., [24, 1, 300, 128])

	DOA Features (doa_featureset)
	Concept: 5-channel features (1 log-mel + 4 intensity vectors).
	read_buffer: [5, config.frame_time_seq, config.mel_bins] (e.g., [5, 300, 128])
	x_in: [config.batch_size, 5, config.frame_time_seq, config.mel_bins] (e.g., [24, 5, 300, 128])

	SED Labels (sed_labelset)
	Concept: Binary flag reference per class track label.
	read_buffer: [1, config.frame_time_seq, (se_count * track_count * 1)]
	x_in: [config.batch_size, 1, config.frame_time_seq, (se_count * track_count * 1)]

	DOA Labels (doa_labelset)
	Concept: Flattened Cartesian coordinates (X, Y).
	read_buffer: [1, config.frame_time_seq, (se_count * track_count * 2)]
	x_in: [config.batch_size, 1, config.frame_time_seq, (se_count * track_count * 2)]
	*/
	std::vector<size_t> sed_fet_buffer_dim = {1, frame_time_seq, mel_bins}; // SED feature buffer dimension
	std::vector<size_t> doa_fet_buffer_dim = {5, frame_time_seq, mel_bins}; // DOA feature buffer dimension
	std::vector<size_t > sed_label_buffer_dim = { 1, static_cast<size_t>(t_prime), static_cast<size_t>(se_count * track_count * 1) }; // SED label buffer dimension
	std::vector<size_t> doa_label_buffer_dim = { 1, static_cast<size_t>(t_prime), static_cast<size_t>(se_count * track_count * 2) }; // DOA label buffer dimension
	std::atomic<bool> on{ true }; // Control flag


	SystemConfig(size_t sample_rate, 
		size_t fft_size, 
		size_t mel_bins, 
		size_t hop_length, 
		double target_res, 
		size_t batch_size, 
		int64_t se_count, 
		int64_t track_count)
		: sample_rate(sample_rate),
		fft_size(fft_size),
		hop_length(hop_length),
		mel_bins(mel_bins),
		batch_size(batch_size),
		se_count(se_count),
		track_count(track_count),
		target_res(target_res) {
	}

	// The default constructor creates a approximately 6 minute runtime, batch amount is hard-coded to 5.
	SystemConfig()
		: sample_rate(16000),
		channels(4),
		fft_size(512),
		hop_length(160),
		target_res(0.1),
		mel_bins(128),
		batch_size(24), 
		se_count(2),
		track_count(3) {
	}
};


static constexpr int DEBUG_LIMIT = 26;
template<typename Cmd>
inline std::optional<Cmd> read_input(SystemConfig& config, bool JSON) {
	std::string raw_input;
	int debug_count = 1;

	std::cout << (JSON ? "Provide JSON signature:" : "Processing... Type 'exit' to stop.") << '\n';
	while (debug_count < DEBUG_LIMIT) {
		if (!std::getline(std::cin, raw_input) || raw_input == "exit") {break;}
		if (raw_input.empty()) {continue;}
		if (JSON) {
			if (nlohmann::json::accept(raw_input)) {
				try {
					auto json = nlohmann::json::parse(raw_input);
					return json.get<Cmd>();
				}
				catch (const nlohmann::json::exception& e) {
					std::cerr << "JSON Mapping Error: " << e.what() << '\n';
				}
				catch (const std::exception& e) {
					std::cerr << "Input Error: " << e.what() << '\n';
				}
			}
		} else {
			std::cerr << "Invalid JSON syntax. Try again." << '\n';
		}
		std::cout << "Provide JSON signature (Attempt " << ++debug_count << "):" << '\n';
	}
	config.on.store(false);
	std::cout << "Stopping application." << '\n';
	return std::nullopt;
}

template<typename Task, typename Cmd>
inline int model_process() {
	SystemConfig config = SystemConfig();
	auto cmd = read_input<Cmd>(config, true);
	if (cmd.has_value()) {
		// Read for exit command
		std::jthread exit_thread([&config](const std::stop_token& stop_token) {
			read_input<nlohmann::json>(config, false);
			});
		// Operate on model task
		std::jthread model_thread([cmd, &config](const std::stop_token& stop_token) {
			try {
				Task instance(cmd.value(), config);
			}
			catch (const std::exception& e) {
				std::cerr << "Task error: " << e.what() << '\n';
				config.on.store(false, std::memory_order_relaxed);
			}
		});
		while (config.on.load(std::memory_order_relaxed)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}
	// Prevents a readline hang
	std::exit(0); 
	return 0;
}