#pragma once
#include "../include/config.h"
#include "../include/fsm.h"
#include "../include/model.h"

// xtensor
#include <xtensor/containers/xarray.hpp>

// torch
#include <torch/torch.h>

// c10, for CUDA memory management
#include <c10/cuda/CUDACachingAllocator.h>

struct TrainCmd {
	std::string device_name;
	std::string zarr_dir; // Directory of zarr files, will iterate "trial_x" zarrs for their data.
	std::int16_t zarr_amount = 0; // Ideally a divisible of 5, >= 100.
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(TrainCmd, device_name, zarr_dir, zarr_amount)
};

class Train {
private:
	SystemConfig& config;;
	// Zarr Dir
	std::string zarr_dir;
	static std::unique_ptr<torch::optim::AdamW> optimizer_generate(M2M_AST& model) {
		torch::optim::AdamWOptions opt_options(1e-4);
		opt_options.weight_decay(0.01);
		return std::make_unique<torch::optim::AdamW>(model->parameters(), opt_options);
	}

	std::string zarr_path_generator(int trial_num) {
		return zarr_dir + "/trial_" + std::to_string(trial_num);
	}

	void run_model(ModelType type, const std::string& save_name, int zarr_amount) {
		std::string stage_name = (type == ModelType::SED) ? "SED" : "DOA";
		std::cout << "Starting " << stage_name << " training for " << zarr_amount << " zarr files." << '\n';
		// 1. Model Initialization
		M2M_AST model(config, type);
        model->init();

		// Generate optimizer for the model
		auto optimizer = optimizer_generate(model);
        model->set_optimizer(optimizer.get());

		float best_val_loss = std::numeric_limits<float>::max();

		// 3. The Epoch Loop
        for (int epoch = 0; epoch < config.epochs && config.on.load(std::memory_order_relaxed); ++epoch) {
            model->adjust_learning_rate(static_cast<float>(epoch), 
								static_cast<float>(config.epochs), 
								static_cast<float>(config.warmup_epochs), 
								1e-4);
            
            float epoch_train_loss = 0.0F;
            float epoch_val_loss = 0.0F;

            for (int zarr_count = 0; zarr_count < zarr_amount; ++zarr_count) {
                std::string zarr_path = zarr_path_generator(zarr_count);
                Reader reader(zarr_path, config);

                // Dynamically route the datasets based on the model type
                auto [t_loss, v_loss] = model->batch_train(
                    config,
                    (type == ModelType::SED) ? &reader.sed_featureset : nullptr,
                    (type == ModelType::DOA) ? &reader.doa_featureset : nullptr,
                    &reader.sed_labelset,
                    (type == ModelType::DOA) ? &reader.doa_labelset : nullptr
                );

                epoch_train_loss += t_loss;
                epoch_val_loss += v_loss;
            }

            // Average the losses across zarr files for the epoch
            epoch_train_loss /= static_cast<float>(zarr_amount);
            epoch_val_loss /= static_cast<float>(zarr_amount);

            std::cout << "Epoch " << epoch + 1 << "/" << config.epochs 
                      << " | Train: " << epoch_train_loss 
                      << " | Val: " << epoch_val_loss << '\n';

            // Save the best model based on validation loss
            if (epoch_val_loss < best_val_loss) {
                best_val_loss = epoch_val_loss;
                torch::save(model, save_name);
                std::cout << "--> New best " << stage_name << " model saved!" << '\n';
            }
        }

        if (torch::cuda::is_available()) {
			// Ensure all CUDA operations are completed before exiting the function
			torch::cuda::synchronize();
			// Release the VRAM back to the OS for other stages or processes
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
	}
public:
	Train(const TrainCmd& cmd,
		SystemConfig& config)
		: config(config), zarr_dir(cmd.zarr_dir) {
		// Stage 1 SED Training, Stage 2 DOA Training
		this->run_model(ModelType::SED, "m2m_ast_sed.pt", cmd.zarr_amount);
		this->run_model(ModelType::DOA, "m2m_ast_doa.pt", cmd.zarr_amount);
	}

	~Train() {
		config.on.store(false, std::memory_order_relaxed);
	}
};

