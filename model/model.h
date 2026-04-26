#include <torch/torch.h>

struct PatchEmbeddingImpl : torch::nn::Module {
    torch::nn::Conv2d proj{ nullptr };
    // For M2M-AST, we use a 16x16 kernel with a stride of 10 to maintain the 6-sample overlap
    PatchEmbeddingImpl(int in_channels = 3, int embed_dim = 768) {
        torch::nn::Conv2dOptions opts(in_channels, embed_dim, 16);
        opts.stride(10);
        proj = register_module("proj", torch::nn::Conv2d(opts));
    }
    torch::Tensor forward(torch::Tensor x) {
        // x shape in: [B, 3, 300, 128] 
        // x shape out: [B, 768, 29, 12] (29 temporal steps, 12 frequency steps)
        x = proj->forward(x);

        // Flatten the 2D spatial grid (29 * 12 = 348 patches) and transpose
        // Final shape: [B, 348, 768]
        return x.flatten(2).transpose(1, 2);
    }
};
TORCH_MODULE(PatchEmbedding);

struct M2M_ASTImpl : torch::nn::Module {
    int t_prime;
    int embed_dim;
    PatchEmbedding patch_embed{ nullptr };

    // Learnable tensors
    torch::Tensor cls_tokens;
    torch::Tensor pos_embed;

    // Standard Transformer Encoder
    torch::nn::TransformerEncoder encoder{ nullptr };

    // Dual Heads for Sound Event Detection (SED) and 2D Direction of Arrival (DOAE)
    torch::nn::Linear sed_head{ nullptr };
    torch::nn::Linear doae_head{ nullptr };

    M2M_ASTImpl(int output_resolution_frames = 30, int num_classes = 13)
        : t_prime(output_resolution_frames), embed_dim(768) {

        // 1. Initialize patch embedding (3 channels)
        patch_embed = register_module("patch_embed", PatchEmbedding(3, embed_dim));

        // 2. Initialize the t' classification tokens 
        // For a 3-second clip at 100ms resolution, t_prime is 30. Shape: [1, 30, 768]
        cls_tokens = register_parameter("cls_tokens", torch::randn({ 1, t_prime, embed_dim }));

        // 3. Initialize Positional embeddings
        // Must cover both the 30 classification tokens and the 348 audio patches
        int num_patches = 348;
        pos_embed = register_parameter("pos_embed", torch::randn({ 1, t_prime + num_patches, embed_dim }));

        // 4. Build the PyTorch Transformer Encoder
        torch::nn::TransformerEncoderLayerOptions encoder_layer_opts(embed_dim, 12);
        encoder_layer_opts.dropout(0.1).activation(torch::kGELU);
        torch::nn::TransformerEncoderLayer encoder_layer(encoder_layer_opts);

        torch::nn::TransformerEncoderOptions encoder_opts(encoder_layer, 12);
        encoder = register_module("encoder", torch::nn::TransformerEncoder(encoder_opts));

        // 5. Build Output Heads
        // SED outputs a probability per class [e.g., 13]
        sed_head = register_module("sed_head", torch::nn::Linear(embed_dim, num_classes));
        // DOAE outputs X and Y coordinates per class [e.g., 26]
        doae_head = register_module("doae_head", torch::nn::Linear(embed_dim, num_classes * 2));
    }

    torch::Tensor forward(torch::Tensor x) {
        int batch_size = x.size(0);

        // Extract patches -> [B, 348, 768]
        torch::Tensor patches = patch_embed->forward(x);

        // Expand cls_tokens to match the current batch -> [B, 30, 768]
        torch::Tensor batch_cls_tokens = cls_tokens.expand({ batch_size, -1, -1 });

        // Concatenate cls_tokens to the beginning of the patch sequence -> [B, 378, 768]
        x = torch::cat({ batch_cls_tokens, patches }, 1);

        // Add positional context to the entire sequence
        x = x + pos_embed;

        // The LibTorch TransformerEncoder expects the sequence dimension first: [Sequence, Batch, Embed]
        x = x.transpose(0, 1);
        x = encoder->forward(x);
        x = x.transpose(0, 1); // Transpose back to [Batch, Sequence, Embed]

        // Slice out ONLY the classification tokens used for output predictions
        // We take the first 't_prime' elements along the sequence dimension (dim 1)
        torch::Tensor output_tokens = x.slice(1, 0, t_prime);

        // Route through specific activations for multi-task learning
        torch::Tensor sed_out = torch::sigmoid(sed_head->forward(output_tokens));
        torch::Tensor doae_out = torch::tanh(doae_head->forward(output_tokens));

        // Concatenate the final outputs. 
        // For 13 classes, this returns a unified ACCDOA-style tensor of shape: [B, 30, 39]
        return torch::cat({ sed_out, doae_out }, /*dim=*/ -1);
    }
};
TORCH_MODULE(M2M_AST);