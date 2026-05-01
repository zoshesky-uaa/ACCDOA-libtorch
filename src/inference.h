#pragma once
#include "../include/config.h"
#include "../include/features.h"
#include "../include/audio.h"
#include "../include/model.h"

// xtensor
#include <xtensor/containers/xarray.hpp>

// torch
#include <torch/torch.h>

struct InferenceCmd {
	std::string device_name = "";
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(InferenceCmd, device_name)
};

struct InferenceBuffer {
    const SystemConfig& config;
    xt::xtensor<float, 3> chunk_buffer; // CPU Buffer: [Channels, inference_amount, Mel]
    torch::Tensor x_in;                 // GPU Tensor: [1, Channels, 300, Mel]
    size_t frames_added = 0;
	int current_local_idx = 0;

    InferenceBuffer(const SystemConfig& config, ModelType type) : config(config) {
        // Initialize 3-second context: [C, 300, 128]
        chunk_buffer = xt::empty<float>({size_t(type), config.inference_amount, config.mel_bins});
        
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32) // Keep base FP32 for blob wrapping
            .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        x_in = torch::zeros({ 1, (int64_t)type, (int64_t)config.frame_time_seq, (int64_t)config.mel_bins }, options);
		if (torch::cuda::is_available()) {
				x_in = x_in.to(torch::MemoryFormat::ChannelsLast);
		}
	}

	bool add_frame(const xt::xtensor<float, 2>& features) {
        // features shape: [Channels, Mel_bins]
        xt::noalias(xt::view(chunk_buffer, xt::all(), current_local_idx, xt::all())) = features;
        current_local_idx++;

        if (current_local_idx >= config.inference_amount) {
            flush_to_tensor();
            current_local_idx = 0;
            return true; // Signals that a new chunk is ready for inference
        }
        return false;
    }

	void flush_to_tensor() {
        bool use_cuda = torch::cuda::is_available();
        // 1. Wrap the filled CPU chunk_buffer in a torch::Tensor (Zero-copy on CPU)
        torch::Tensor chunk_view = torch::from_blob(
            chunk_buffer.data(),
            { 1, x_in.size(1), (int64_t)config.inference_amount, (int64_t)config.mel_bins },
            torch::kFloat32
        );

        // 2. Roll the GPU tensor left by 'inference_amount' (Native CUDA operation) (wrapped)
        x_in = torch::roll(x_in, -static_cast<int64_t>(config.inference_amount), /*dims=*/2);

        // 3. Safely copy the new chunk to the tail of x_in (Handles CPU->GPU DMA)
        int64_t tail_start = config.frame_time_seq - config.inference_amount;
        x_in.narrow(/*dim=*/2, tail_start, config.inference_amount).copy_(chunk_view, /*non_blocking=*/use_cuda);
    }
};

class Inference {
private:
	SystemConfig& config;
	// Shared buffer for active features
	xt::xtensor<float, 2> sed_features;
	xt::xtensor<float, 2> doa_features;

	// Process classes
	AudioDevice audio_device;
	FeatureExtractor feature_extractor;

	// Model and optimizer
	M2M_AST sed_model;
	M2M_AST doa_model;

	// Inference buffers
	InferenceBuffer sed_buffer;
	InferenceBuffer doa_buffer;

public:
	Inference(const InferenceCmd& cmd,
		SystemConfig& config)
		: config(config),
		sed_model(config, ModelType::SED),
		doa_model(config, ModelType::DOA),
		sed_features({ 1, config.mel_bins }, 0.0f),
		doa_features({ 5, config.mel_bins }, 0.0f),
		audio_device(cmd.device_name, config),
		feature_extractor(config, sed_features, doa_features),
		sed_buffer(config, ModelType::SED),
		doa_buffer(config, ModelType::DOA) {

		// Initialize the model
		sed_model->init();
		doa_model->init();

		// Explicitly set models to evaluation mode
		sed_model->eval(); 
        doa_model->eval();
		
		// Start audio capture and feature extraction loop
		audio_device.start();
		warmup(); // Optional warmup to fill buffers and stabilize performance
		std::cout << "Models initialized for inference." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1)); 
		std::cout << "START" << std::endl;
		while (config.on) {
			if (feature_extractor.feature_extract(audio_device)) {
				bool ready_sed = sed_buffer.add_frame(sed_features);
                bool ready_doa = doa_buffer.add_frame(doa_features);
				if (ready_sed && ready_doa) {
					sed_model->inference(sed_buffer.x_in);
					doa_model->inference(doa_buffer.x_in);	
				}
			}	
		}
		std::cout << "END" << std::endl;
	}

	void warmup() {
		for (int i = 0; i < config.frame_time_seq; ++i) {
			if (feature_extractor.feature_extract(audio_device)) {
				sed_buffer.add_frame(sed_features);
				doa_buffer.add_frame(doa_features);
			}
		}
	}
	~Inference() {
		config.on.store(false);
		sed_model = nullptr;
		doa_model = nullptr;

		// CUDA sync operation (cudaDeviceSynchronize wrapper)
		if (torch::cuda::is_available()) {
			torch::cuda::synchronize();
		}
	}
};