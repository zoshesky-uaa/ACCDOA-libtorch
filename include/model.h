#pragma once
#include <torch/torch.h>
#include "config.h"
#include "dataset.h"

// WIP
struct PatchEmbeddingImpl : torch::nn::Module {
    torch::nn::Conv2d proj{ nullptr };

    PatchEmbeddingImpl(const SystemConfig& config, enum ModelType model_type) {
        torch::nn::Conv2dOptions opts(model_type, config.embed_dim, config.patch_size);
        opts.stride(config.conv_stride);
        // Convolution 2d based on the channels, embedding dimensions, and patch size
        proj = register_module("Linear Projection", torch::nn::Conv2d(opts));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = proj->forward(x);
        return x.flatten(2).transpose(1, 2);
    }
};
TORCH_MODULE(PatchEmbedding);

struct M2M_ASTImpl : torch::nn::Module {
    PatchEmbedding patch_embed{ nullptr };
    torch::Tensor cls_tokens;
    torch::Tensor pos_embed;
    torch::nn::TransformerEncoder encoder{ nullptr };
	torch::nn::Linear head{ nullptr };
    int64_t t_prime;
	ModelType model_type;
    torch::optim::Optimizer* optimizer = nullptr;

    M2M_ASTImpl(const SystemConfig& config, enum ModelType model_type): 
        t_prime(config.t_prime), model_type(model_type) {
        patch_embed = register_module("Patch Embedding", PatchEmbedding(config, model_type));

        cls_tokens = register_parameter("Classification Tokens", torch::randn({ 1, t_prime, config.embed_dim }));
        pos_embed = register_parameter("Position Embedding", torch::randn({ 1, config.total_seq, config.embed_dim }));

        torch::nn::TransformerEncoderLayerOptions layer_opts(config.embed_dim, config.att_headers);
        layer_opts.dropout(0.1).activation(torch::kGELU);
        torch::nn::TransformerEncoderOptions encoder_opts(torch::nn::TransformerEncoderLayer(layer_opts), config.enc_layers);
        encoder = register_module("Transformer Encoder", torch::nn::TransformerEncoder(encoder_opts));

        // Linear layers = Dense layers without activation
        switch (model_type) {
            case ModelType::SED:
                head = register_module("SED", torch::nn::Linear(config.embed_dim, config.se_count));
                break;
            case ModelType::DOA:
                head = register_module("DOAE", torch::nn::Linear(config.embed_dim, config.se_count * 2 * config.track_count));
                break;
        };
    }

    torch::Tensor forward(torch::Tensor x) {
        int64_t batch_size = x.size(0);

        torch::Tensor patches = patch_embed->forward(x);
        torch::Tensor expanded_cls = cls_tokens.expand({ batch_size, -1, -1 });

        x = torch::cat({ expanded_cls, patches }, 1);
        x = x + pos_embed;
        x = x.transpose(0, 1);
        x = encoder->forward(x);
        x = x.transpose(0, 1);

        torch::Tensor output_tokens = x.slice(1, 0, t_prime);

		// SED head uses sigmoid for multi-label classification, DOAE head uses tanh for regression; activation
        switch (model_type) {
            case ModelType::SED:
                return torch::sigmoid(head->forward(output_tokens));
            case ModelType::DOA:
                return torch::tanh(head->forward(output_tokens));
		};
    }


    torch::Tensor loss(torch::Tensor& prediction, 
                        torch::Tensor& sed_target, 
                        torch::Tensor& doa_target) {
        // Predictions do not have the channel dimension, squeeze it out.
        torch::Tensor s_target = sed_target.squeeze(1);
		if (model_type == ModelType::SED) {
            return torch::nn::functional::binary_cross_entropy(prediction, s_target);
        }   else {
            torch::Tensor d_target = doa_target.squeeze(1);
            torch::Tensor active_mask = (s_target > 0.5).to(torch::kFloat16);
            // Expand mask to match spatial dimensions (x, y). Repeats [B, T', Cl, 1] -> [B, T', Cl, 2]
            active_mask = active_mask.repeat_interleave(2, /*dim=*/-1);
            auto mse_opts = torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone);
            torch::Tensor raw_mse = torch::nn::functional::mse_loss(prediction, d_target, mse_opts);
            return (raw_mse * active_mask).sum() / (active_mask.sum() + 1e-8);
        }
    }

    void set_optimizer(torch::optim::Optimizer* opt) {
        this->optimizer = opt;
    }

    void adjust_learning_rate(int epoch, int total_epochs, int warmup_epochs, float target_lr) {
        if (!this->optimizer) return;
        double new_lr = 0.0;
        // Linear warmup for the first 'warmup_epochs', then cosine annealing for the rest of the training
        if (epoch < warmup_epochs) {
            // Phase 1: Linear Warmup (0 -> target_lr)
            new_lr = target_lr * (static_cast<double>(epoch + 1) / warmup_epochs);
        } else {
            // Phase 2: Cosine Annealing (target_lr -> 0)
            double progress = static_cast<double>(epoch - warmup_epochs) / (total_epochs - warmup_epochs);
            new_lr = 0.5 * target_lr * (1.0 + std::cos(M_PI * progress));
        }
        // Update the learning rate for all parameter groups in the optimizer
        for (auto& param_group : this->optimizer->param_groups()) {
            if (param_group.has_options()) {
                static_cast<torch::optim::AdamWOptions&>(param_group.options()).set_lr(new_lr);
            }
        }
    }
    std::tuple<float, float> batch_train(
        const SystemConfig& config,
        DatasetProcessor* sed_featureset,
        DatasetProcessor* doa_featureset,
        DatasetProcessor* sed_labelset,
        DatasetProcessor* doa_labelset) {
        torch::Tensor x_in, sed_target, doa_target;
        float total_train_loss = 0.0f;
        this->train();
        // Training loop per epoch
        for (int batch_idx = 0; batch_idx < (config.batch_amount-1) && config.on.load(); ++batch_idx) {
            if (this->model_type == ModelType::SED) {
                x_in = sed_featureset->batch();
                sed_target = sed_labelset->batch();
            }
            else {
                x_in = doa_featureset->batch();
                doa_target = doa_labelset->batch();
                sed_target = sed_labelset->batch();
            }
            this->optimizer->zero_grad();
            torch::Tensor prediction = this->forward(x_in);
            torch::Tensor loss = this->loss(prediction, sed_target, doa_target);
            loss.backward();
            // Gradient clipping to prevent exploding gradients
            torch::nn::utils::clip_grad_norm_(this->parameters(), /*max_norm=*/1.0);
            this->optimizer->step();

            total_train_loss += loss.to(torch::kFloat32).item<float>();
        }

        // Validation loop per epoch
        this->eval();
        torch::NoGradGuard no_grad;
        if (this->model_type == ModelType::SED) {
            x_in = sed_featureset->batch();
            sed_target = sed_labelset->batch();
        }
        else {
            x_in = doa_featureset->batch();
            doa_target = doa_labelset->batch();
            sed_target = sed_labelset->batch();
        }
        torch::Tensor val_prediction = this->forward(x_in);
        torch::Tensor val_loss = this->loss(val_prediction, sed_target, doa_target);

        sed_featureset->read_reset();
        doa_featureset->read_reset();
        sed_labelset->read_reset();
        doa_labelset->read_reset(); 
        // Return average training loss and validation loss for the epoch
        return {(total_train_loss / config.batch_amount), val_loss.to(torch::kFloat32).item<float>()};
    }
    // WIP will need to make a return
    void inference(torch::Tensor& x_in) {
        torch::NoGradGuard no_grad;
        torch::Tensor prediction = this->forward(x_in);
        // Post-processing of predictions will go here later
	}

    void init() {
        torch::nn::init::normal_(this->cls_tokens, 0.0, 0.02);
        this->cls_tokens.clamp_(-0.04, 0.04);

        torch::nn::init::normal_(this->pos_embed, 0.0, 0.02);
        this->pos_embed.clamp_(-0.04, 0.04);
        // Initialize whatever head was registered
        if (this->head) {
            torch::nn::init::normal_(this->head->weight, 0.0, 0.02);
            this->head->weight.clamp_(-0.04, 0.04);
            torch::nn::init::constant_(this->head->bias, 0.0);
        }

        // Move model to GPU if available
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available. Moving model to GPU." << std::endl;
            this->to(torch::kCUDA, torch::kFloat32);
            // Explicitly set 4D weights (Convolution) to ChannelsLast (NHWC)
            for (auto& param : this->parameters()) {
                if (param.dim() == 4) {
                    param.set_data(param.data().contiguous(torch::MemoryFormat::ChannelsLast));
                }
            }
        }
        else {
            std::cout << "CUDA is not available. Running on CPU." << std::endl;
            this->to(torch::kCPU);
        }
    }


};
TORCH_MODULE(M2M_AST);

// Takes pretained positional embeddings from a ViT model and interpolates them to fit the M2M_AST's new input dimensions.
// use with caution not tested
inline void interpolate_pos_encoding(M2M_AST& model, torch::Tensor pretrained_pos_embed, const SystemConfig& config) {
    torch::NoGradGuard no_grad;

    // 1. Separate the CLS tokens from the spatial patches
    // Pretrained ViT usually has 1 or 2 CLS/Distill tokens
    int64_t num_extra_tokens = 1;
    torch::Tensor extra_tokens = pretrained_pos_embed.slice(1, 0, num_extra_tokens);
    torch::Tensor patch_tokens = pretrained_pos_embed.slice(1, num_extra_tokens);

    // 2. Reshape to the original source grid (e.g., 14x14)
    int64_t old_grid = std::sqrt(patch_tokens.size(1));
    patch_tokens = patch_tokens.reshape({ 1, old_grid, old_grid, config.embed_dim })
        .permute({ 0, 3, 1, 2 }); // [B, C, H, W] for interpolation

    // 3. Interpolate to your target audio grid (29x12)
    torch::Tensor interpolated_patches = torch::nn::functional::interpolate(
        patch_tokens,
        torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>({ config.n_t, config.n_f }))
        .mode(torch::kBilinear)
        .align_corners(false)
    );

    // 4. Flatten back and recombine with your t_prime classification tokens
    interpolated_patches = interpolated_patches.permute({ 0, 2, 3, 1 }).flatten(1, 2);

    // Expand your averaged CLS tokens to match t_prime
    torch::Tensor averaged_cls = extra_tokens.mean(1, true).expand({ 1, config.t_prime, -1 });

    // Final pos_embed for your M2M_AST
    model->pos_embed.copy_(torch::cat({ averaged_cls, interpolated_patches }, 1));
}