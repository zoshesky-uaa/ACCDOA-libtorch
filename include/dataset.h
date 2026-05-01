#pragma once
#include "config.h"
// z5
#include <z5/factory.hxx>
#include <z5/attributes.hxx>
#include "z5/filesystem/handle.hxx"

// xtensor
#include <xtensor/containers/xarray.hpp>
#include "z5/multiarray/xtensor_access.hxx" // IWYU pragma: keep

// torch
#include <torch/torch.h>

struct DatasetProcessor {
    const SystemConfig& config;
    std::shared_ptr<z5::Dataset> ds;
    z5::types::ShapeType write_chunk;
    z5::types::ShapeType read_chunk;
    z5::types::ShapeType chunk_shape;
    z5::types::ShapeType ds_shape;
    torch::Tensor x_in;
    xt::xtensor<float, 3> read_buffer;
    int batch_idx = 0;

    void write(const xt::xtensor<float, 3>& data) {
        ds->writeChunk(write_chunk, data.data());
        std::cout << "Write chunk at offset: [" << write_chunk[1] << "]" << std::endl;
        write_chunk[1]++;
    }

    void read() {
        ds->readChunk(read_chunk, read_buffer.data());
        std::cout << "Read chunk at offset: [" << read_chunk[1] << "]" << std::endl;
        read_chunk[1]++;
    }

    torch::Tensor batch() {
        batch_idx = 0;
        bool use_cuda = torch::cuda::is_available();
        while (batch_idx < config.batch_size && config.on) {
            read();
            torch::Tensor chunk_view = torch::from_blob(
                read_buffer.data(),
                { (int64_t)chunk_shape[0], (int64_t)chunk_shape[1], (int64_t)chunk_shape[2] },
                torch::kFloat32
            );
            x_in[batch_idx].copy_(chunk_view, /*non_blocking=*/use_cuda);
            batch_idx++;
        }
        return x_in;
    }

    void read_reset() {
        read_chunk = { 0, 0, 0 };
    }

    DatasetProcessor(z5::filesystem::handle::File root,
        const std::string& name,
        size_t channels,
        const SystemConfig& config,
        DatasetType type) : config(config) {

        // Define dataset shape and chunk shape based on the dataset type
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

        switch (type) {
        case DatasetType::SED_FEATURES:
            chunk_shape = { config.sed_fet_buffer_dim[0], config.sed_fet_buffer_dim[1], config.sed_fet_buffer_dim[2] };
            ds_shape = { config.sed_fet_buffer_dim[0], config.frame_max, config.sed_fet_buffer_dim[2] };
            x_in = torch::empty({ (int64_t)config.batch_size, (int64_t)config.sed_fet_buffer_dim[0], (int64_t)config.sed_fet_buffer_dim[1], (int64_t)config.sed_fet_buffer_dim[2] }, options);
            break;
        case DatasetType::DOA_FEATURES:
            chunk_shape = { config.doa_fet_buffer_dim[0], config.doa_fet_buffer_dim[1], config.doa_fet_buffer_dim[2] };
            ds_shape = { config.doa_fet_buffer_dim[0], config.frame_max, config.doa_fet_buffer_dim[2] };
            x_in = torch::empty({ (int64_t)config.batch_size, (int64_t)config.doa_fet_buffer_dim[0], (int64_t)config.doa_fet_buffer_dim[1], (int64_t)config.doa_fet_buffer_dim[2] }, options);
            break;
        case DatasetType::SED_LABELS:
            chunk_shape = { config.sed_label_buffer_dim[0], config.sed_label_buffer_dim[1], config.sed_label_buffer_dim[2] };
            ds_shape = { config.sed_label_buffer_dim[0], config.frame_max, config.sed_label_buffer_dim[2] };
            x_in = torch::empty({ (int64_t)config.batch_size, (int64_t)config.sed_label_buffer_dim[0], (int64_t)config.sed_label_buffer_dim[1], (int64_t)config.sed_label_buffer_dim[2] }, options);
            break;
        case DatasetType::DOA_LABELS:
            chunk_shape = { config.doa_label_buffer_dim[0], config.doa_label_buffer_dim[1], config.doa_label_buffer_dim[2] };
            ds_shape = { config.doa_label_buffer_dim[0], config.frame_max, config.doa_label_buffer_dim[2] };
            x_in = torch::empty({ (int64_t)config.batch_size, (int64_t)config.doa_label_buffer_dim[0], (int64_t)config.doa_label_buffer_dim[1], (int64_t)config.doa_label_buffer_dim[2] }, options);
            break;
        };

        assert(ds_shape[1] % chunk_shape[1] == 0 &&
            ds_shape[2] % chunk_shape[2] == 0 &&
            "Dataset_shape must be divisible by chunk_shape in all dimensions");

        // 2. Force ChannelsLast layout for 4D convolution/patching efficiency 
        if (torch::cuda::is_available()) {
            x_in = x_in.to(torch::MemoryFormat::ChannelsLast);
        }

        // For Writer/Reader operations
        if (z5::filesystem::handle::Dataset(root, name).exists()) {
            ds = z5::openDataset(root, name);
        }
        else {
            z5::types::CompressionOptions cOpts = { {"codec", "zstd"}, {"level", 3}, {"shuffle", 1}, {"blocksize", 0} };
            ds = z5::createDataset(root, name, "float32", ds_shape, chunk_shape, "blosc", cOpts);
        }

        // Initialize offsets and read buffer, offsets not intended to be reset, del/free construction after batching.
        write_chunk = { 0, 0, 0 };
        read_chunk = { 0, 0, 0 };
        read_buffer = xt::empty<float>(std::array<size_t, 3>{chunk_shape[0], chunk_shape[1], chunk_shape[2]});
    };

    ~DatasetProcessor() {
        x_in = torch::Tensor();
        read_buffer = xt::xtensor<float, 3>();
        if (ds) {
            ds.reset();
        }
    }
};