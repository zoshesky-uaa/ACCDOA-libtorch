#pragma once
#include <atomic>
#include <vector>
#include <optional>
#include <iostream>
#include <thread>
#include <string>
#include <cstdint>
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
struct ConfigData {
    // 1. Controllable parameters
    size_t sample_rate;
    size_t fft_size;
    size_t mel_bins;
    size_t hop_length;
    double target_res;
    size_t batch_size;
    int64_t se_count;
    int64_t track_count;

    // 2. Training/Model parameters
    size_t epochs;
    size_t warmup_epochs;
    size_t batch_amount;
    size_t channels;
    int64_t time_window;
    int64_t patch_size;
    int64_t patch_overlap;
    int64_t enc_layers;
    int64_t att_headers;
    int64_t embed_dim;

    // 3. Calculated sequence variables
    double input_frame_time;
    size_t frame_time_seq;
    size_t frame_max;
    int64_t conv_stride;
    size_t fft_bins;
    size_t history_size;
    int64_t t_prime;
    size_t label_max;
    size_t inference_amount;
    int64_t n_t;
    int64_t n_f;
    int64_t num_patches;
    int64_t total_seq;

    // 4. Buffer dimensions
    std::vector<size_t> sed_fet_buffer_dim;
    std::vector<size_t> doa_fet_buffer_dim;
    std::vector<size_t> sed_label_buffer_dim;
    std::vector<size_t> doa_label_buffer_dim;

	// The Giant Object Mapper
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(ConfigData,
        sample_rate, fft_size, mel_bins, hop_length, target_res, batch_size, se_count, track_count,
        epochs, warmup_epochs, batch_amount, channels, time_window, patch_size, patch_overlap, enc_layers,
        att_headers, embed_dim, input_frame_time, frame_time_seq, frame_max, conv_stride, fft_bins, history_size,
        t_prime, label_max, inference_amount, n_t, n_f, num_patches, total_seq,
        sed_fet_buffer_dim, doa_fet_buffer_dim, sed_label_buffer_dim, doa_label_buffer_dim)
};

struct SystemConfig : public ConfigData {
    // Control flag (Not mapped in JSON)
    std::atomic<bool> on{ true };
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
					nlohmann::from_json(json, static_cast<ConfigData&>(config));
					config.on.store(true);            	
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
	std::cout << "Stopping application." << '\n';
	return std::nullopt;
}

template<typename Task, typename Cmd>
inline int model_process() {
	SystemConfig config;
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