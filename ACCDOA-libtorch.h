#pragma once
#include "config.h"
#include "audio/features.h"
#include "audio/audio.h"
#include "audio/fsm.h"
#include <nlohmann/json.hpp>
#include "model/model.h"
// xtensor
#include <xtensor/containers/xarray.hpp>
#include "z5/multiarray/xtensor_access.hxx"
// torch
#include <torch/torch.h>

class ACCDOA {
	private:
		bool training;
		// Shared buffer for active features
		xt::xtensor<float, 2> sed_features;
		xt::xtensor<float, 2> doa_features;
		
		// Process classes
		AudioDevice audio_device;
		FeatureExtractor feature_extractor;
		Writer writer;
		SystemConfig& config;

		// Model and optimizer
		M2M_AST sed_model;
		M2M_AST doa_model;
		std::unique_ptr<torch::optim::AdamW> sed_optimizer;
		std::unique_ptr<torch::optim::AdamW> doa_optimizer;

		int tick = 0;
	public:
		ACCDOA(const bool training_mode, 
				const std::string device_name,
				const std::string zarr_path,	
				SystemConfig& config) 
			: training(training_mode), 
			sed_features({1, config.mel_bins}, 0.0f),
			doa_features({5, config.mel_bins}, 0.0f),
			audio_device(device_name, config),
			feature_extractor(config, sed_features, doa_features),
			writer(zarr_path, config, training_mode),
			config(config),
			sed_model(config),
			doa_model(config)
			{			
			// Initialize the model and optimizer
			init(sed_model, training_mode);
			init(doa_model, training_mode);
			torch::optim::AdamWOptions opt_options(1e-4); 
			opt_options.weight_decay(0.01);
			sed_optimizer = std::make_unique<torch::optim::AdamW>(sed_model->parameters(), opt_options);
			doa_optimizer = std::make_unique<torch::optim::AdamW>(doa_model->parameters(), opt_options);
			audio_device.start();

			std::cout << "START" << std::endl;
			while (config.on) {
				if (feature_extractor.feature_extract(audio_device)) {
					writer.add_frame(sed_features, doa_features);
					if (writer.count >= config.frame_max) {
						config.on.store(false);
					}
					std::cout << std::to_string(tick++) << std::endl;
				}
			}
			if (writer.count != config.frame_max) {
				std::cerr << "Warning: Early termination detected. Potential data loss, delete associated files." << std::endl;
				return;
			}
			if (training) {
				std::cout << "Begining traning process." << std::endl;
				// to be finished
			}
			std::cout << "END" << std::endl;
		}
		~ACCDOA() {
			config.on.store(false);
			sed_optimizer.reset();
			doa_optimizer.reset();
			sed_model = nullptr;
			doa_model = nullptr;
			// CUDA sync operation, probably need this around most CUDA operations seems like a wrapper for cudaDeviceSynchronize().
			if (torch::cuda::is_available()) {
				torch::cuda::synchronize();
			}
		}
};

