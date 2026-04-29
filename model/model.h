#include <torch/torch.h>
#include "../config.h"

// WIP
struct PatchEmbeddingImpl : torch::nn::Module {
    torch::nn::Conv2d proj{ nullptr };

    PatchEmbeddingImpl(const SystemConfig& config) {
        torch::nn::Conv2dOptions opts(config.feature_channels, config.embed_dim, config.patch_size);
        opts.stride(config.conv_stride);
        proj = register_module("proj", torch::nn::Conv2d(opts));
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
    torch::nn::Linear sed_head{ nullptr };
    torch::nn::Linear doae_head{ nullptr };

    M2M_ASTImpl(const SystemConfig& config) {
        patch_embed = register_module("patch_embed", PatchEmbedding(config));

        cls_tokens = register_parameter("cls_tokens", torch::randn({ 1, config.t_prime, config.embed_dim }));
        pos_embed = register_parameter("pos_embed", torch::randn({ 1, config.total_seq, config.embed_dim }));

        torch::nn::TransformerEncoderLayerOptions layer_opts(config.embed_dim, config.att_headers);
        layer_opts.dropout(0.1).activation(torch::kGELU);

        torch::nn::TransformerEncoderOptions encoder_opts(torch::nn::TransformerEncoderLayer(layer_opts), config.enc_layers);
        encoder = register_module("encoder", torch::nn::TransformerEncoder(encoder_opts));

        // Linear layers = Dense layers without activation
        sed_head = register_module("sed_head", torch::nn::Linear(config.embed_dim, config.se_count));
        doae_head = register_module("doae_head", torch::nn::Linear(config.embed_dim, config.se_count * config.space_dim * config.track_count));
    }

    torch::Tensor forward(torch::Tensor x, const SystemConfig& config) {
        int64_t batch_size = x.size(0);

        torch::Tensor patches = patch_embed->forward(x);
        torch::Tensor expanded_cls = cls_tokens.expand({ batch_size, -1, -1 });

        x = torch::cat({ expanded_cls, patches }, 1);
        x = x + pos_embed;

        x = x.transpose(0, 1);
        x = encoder->forward(x);
        x = x.transpose(0, 1);

        torch::Tensor output_tokens = x.slice(1, 0, config.t_prime);

		// SED head uses sigmoid for multi-label classification, DOAE head uses tanh for regression; activation
        torch::Tensor sed_out = torch::sigmoid(sed_head->forward(output_tokens));
        torch::Tensor doae_out = torch::tanh(doae_head->forward(output_tokens));

        return torch::cat({ sed_out, doae_out }, -1);
    }
};
TORCH_MODULE(M2M_AST);

void init(M2M_AST& model, bool training_mode) {
    torch::NoGradGuard no_grad;
	// Set the model to training or evaluation mode
    if (training_mode) {
        model->train();
    }
    else {
        model->eval();
    }

    // 1. Fixed Initialization (Using Normal + Manual Truncation)
	// We use a std of 0.02 and clamp to [-0.04, 0.04] (2-sigma truncation) (trunacted normal not available in C++ API)
    torch::nn::init::normal_(model->cls_tokens, 0.0, 0.02);
    model->cls_tokens.clamp_(-0.04, 0.04);

    torch::nn::init::normal_(model->pos_embed, 0.0, 0.02);
    model->pos_embed.clamp_(-0.04, 0.04);

    // Initialize Heads with zeroed biases
    torch::nn::init::normal_(model->sed_head->weight, 0.0, 0.02);
    model->sed_head->weight.clamp_(-0.04, 0.04);
    torch::nn::init::constant_(model->sed_head->bias, 0.0);

    torch::nn::init::normal_(model->doae_head->weight, 0.0, 0.02);
    model->doae_head->weight.clamp_(-0.04, 0.04);
    torch::nn::init::constant_(model->doae_head->bias, 0.0);

    // Move model to GPU if available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available. Moving model to GPU." << std::endl;
        model->to(torch::kCUDA, torch::kFloat16);
        // Explicitly set 4D weights (Convolution) to ChannelsLast (NHWC)
        for (auto& param : model->parameters()) {
            if (param.dim() == 4) {
                param.set_data(param.data().contiguous(torch::MemoryFormat::ChannelsLast));
            }
        }
    }
    else {
        std::cout << "CUDA is not available. Running on CPU." << std::endl;
		model->to(torch::kCPU);
    }
}


torch::Tensor calculate_seld_loss(torch::Tensor predictions, torch::Tensor targets, const SystemConfig& config) {
    // 1. Slice the concatenated tensor back into SED and DOAE components
    // Shape is [Batch, t_prime, Features]
    torch::Tensor sed_pred = predictions.slice(/*dim=*/2, /*start=*/0, /*end=*/config.se_count);
    torch::Tensor doae_pred = predictions.slice(/*dim=*/2, /*start=*/config.se_count);

    torch::Tensor sed_target = targets.slice(/*dim=*/2, /*start=*/0, /*end=*/config.se_count);
    torch::Tensor doae_target = targets.slice(/*dim=*/2, /*start=*/config.se_count);

    // 2. SED Loss: Binary Cross Entropy
    // Since your model already applied torch::sigmoid in the forward pass, we use standard BCE
    torch::Tensor sed_loss = torch::nn::functional::binary_cross_entropy(sed_pred, sed_target);

    // 3. DOAE Masking Logic
    // Create a binary mask where ground truth SED > 0.5 (event is active)
    torch::Tensor active_mask = (sed_target > 0.5).to(torch::kFloat16);

    // The DOAE output has 'space_dim * track_count' values for EVERY 'se_count'.
    // We must repeat the mask along the last dimension so it aligns with the DOAE shape.
    int64_t repeats = config.space_dim * config.track_count;
    active_mask = active_mask.repeat_interleave(repeats, /*dim=*/-1);

    // 4. Calculate Raw MSE without reduction
    auto mse_opts = torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone);
    torch::Tensor raw_doae_loss = torch::nn::functional::mse_loss(doae_pred, doae_target, mse_opts);

    // 5. Apply the Mask
    torch::Tensor masked_doae_loss = raw_doae_loss * active_mask;

    // 6. Calculate Average DOAE Loss safely (avoid division by zero)
    // Add a tiny epsilon in case the entire batch is absolute silence
    torch::Tensor doae_loss = masked_doae_loss.sum() / (active_mask.sum() + 1e-8);

    // Combine tasks (You can add scaling factors here later if DOAE dominates SED)
    return sed_loss + doae_loss;
}

void training_batch(
    M2M_AST& model,
    torch::optim::AdamW& optimizer,
    torch::Tensor x_in,
    torch::Tensor ground_truth,
    const SystemConfig& config)
{
    // 1. Enable Training Mode (Activates Dropout and LayerNorm tracking)
    model->train();

    // 2. Prep Tensors for Tensor Cores (Must match the model's init state)
    // x_in expected shape: [Batch, 3, 300, 128]
    torch::Tensor input = x_in.to(torch::kCUDA, torch::kFloat16)
        .contiguous(torch::MemoryFormat::ChannelsLast);

    torch::Tensor target = ground_truth.to(torch::kCUDA, torch::kFloat16)
        .contiguous(torch::MemoryFormat::ChannelsLast);

    // 3. Clear old gradients
    optimizer.zero_grad();

    // 4. Forward Pass
    torch::Tensor predictions = model->forward(input, config);

    // 5. Calculate Masked Loss
    torch::Tensor loss = calculate_seld_loss(predictions, target, config);

    // 6. Backward Pass (Calculate new gradients)
    loss.backward();

    // 7. Update Weights
    optimizer.step();

    std::cout << "[Train] Step Loss: " << loss.item<float>() << std::endl;
}


// Takes pretained positional embeddings from a ViT model and interpolates them to fit the M2M_AST's new input dimensions.
// use with caution not tested
void interpolate_pos_encoding(M2M_AST& model, torch::Tensor pretrained_pos_embed, const SystemConfig& config) {
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
    torch::Tensor averaged_cls = extra_tokens.mean(0, true).expand({ 1, config.t_prime, -1 });

    // Final pos_embed for your M2M_AST
    model->pos_embed.copy_(torch::cat({ averaged_cls, interpolated_patches }, 1));
}