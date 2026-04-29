#pragma once
#include "../config.h"
#include "../model/model.h"
// miniaudio
#include <miniaudio.h>
// z5
#include <z5/factory.hxx>
#include <z5/attributes.hxx>
#include "z5/filesystem/handle.hxx"
// xtensor
#include <xtensor/containers/xarray.hpp>
#include "z5/multiarray/xtensor_access.hxx"

struct Chunk {
    xt::xtensor<float, 3> sed_features;
    xt::xtensor<float, 3> doa_features;

    Chunk(int chunk_sz, int mel_bins)
        : sed_features({ (size_t)1, (size_t)chunk_sz, (size_t)mel_bins }),
        doa_features({ (size_t)5, (size_t)chunk_sz, (size_t)mel_bins }) {
    }
};

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

    xt::xarray<float> read() {
        ds->readChunk(read_chunk, read_buffer.data());
        read_chunk[1]++;
		std::cout << "Read chunk at offset: [" << read_chunk[1] << "]" << std::endl;
        return (xt::adapt(read_buffer.data(), read_buffer.size(),
            xt::no_ownership(), read_buffer.shape()));
    }

    void batch() {
        batch_idx = 0;
        bool use_cuda = torch::cuda::is_available();
        while (batch_idx < config.batch_size) {
			read();
            torch::Tensor chunk_view = torch::from_blob(
                read_buffer.data(),
                { (int64_t)chunk_shape[0], (int64_t)chunk_shape[1], (int64_t)chunk_shape[2] },
                torch::kFloat32
            );
            x_in[batch_idx].copy_(chunk_view, /*non_blocking=*/use_cuda);
			batch_idx++;
        }
    }

    DatasetProcessor(z5::filesystem::handle::File root, 
                        const std::string& name, 
                        size_t channels, 
                        const SystemConfig& config,
                        bool training_mode) : config(config) {
        chunk_shape = { channels , config.frame_time_seq, (size_t)config.mel_bins };
        ds_shape = { channels, config.frame_max, (size_t)config.mel_bins };

        assert(ds_shape[1] % chunk_shape[1] == 0 &&
            ds_shape[2] % chunk_shape[2] == 0 &&
            "Dataset_shape must be divisible by chunk_shape in all dimensions");
        // Define compression options for the datasets
        z5::types::CompressionOptions cOpts = {{"codec", "zstd"}, {"level", 3}, {"shuffle", 1}, {"blocksize", 0}};
        ds = z5::createDataset(root, name, "float32", ds_shape, chunk_shape, "blosc", cOpts);

		// Initialize offsets and read buffer
        write_chunk = { 0, 0, 0};
        read_chunk = { 0, 0, 0};
        read_buffer = xt::empty<float>(std::array<size_t, 3>{chunk_shape[0], chunk_shape[1], chunk_shape[2]});
        
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        int64_t b_size = training_mode ? (int64_t)config.batch_size : 1;
        x_in = torch::empty({ b_size, (int64_t)channels, (int64_t)config.frame_time_seq, (int64_t)config.mel_bins }, options);

        // 2. Force ChannelsLast layout for 4D convolution/patching efficiency 
        if (torch::cuda::is_available()) {
            x_in = x_in.to(torch::MemoryFormat::ChannelsLast);
        }
    };

	// Delete default constructor to prevent uninitialized instances
	DatasetProcessor() = delete;
};

class Writer {
    private:
        // z5 uses xtensor-like views to handle multi-dimensional data
        z5::filesystem::handle::File root;
        DatasetProcessor sed_fproc;
        DatasetProcessor doa_fproc;

		const bool training_mode;
        const SystemConfig& config;
		int current_local_idx = 0;

		// Maximum number of chunks (or tasks) that can be queued for writing at once.
		int task_limit = 8; 
		Chunk* active_chunk = nullptr;
		ma_rb task_queue;
        ma_rb free_pool;
        std::thread worker;

		// Helper functions to acquire task pointers from the ring buffers
        static bool rb_acquire_read_task_ptr(ma_rb& rb, void*& ptr) {
            size_t bytes = sizeof(Chunk*);
            return ma_rb_acquire_read(&rb, &bytes, &ptr) == MA_SUCCESS && bytes == sizeof(Chunk*);
        }

        static bool rb_acquire_write_task_ptr(ma_rb& rb, void*& ptr) {
            size_t bytes = sizeof(Chunk*);
            return ma_rb_acquire_write(&rb, &bytes, &ptr) == MA_SUCCESS && bytes == sizeof(Chunk*);
        }

        void thread_loop() {
            while (config.on) {
                void* readPtr;
                Chunk* full_chunk = nullptr;
                if (rb_acquire_read_task_ptr(task_queue, readPtr)) {
					Chunk* full_chunk = nullptr;
                    std::memcpy(&full_chunk, readPtr, sizeof(Chunk*));
                    ma_rb_commit_read(&task_queue, sizeof(Chunk*));
                    // Write the chunks to the respective datasets at the correct offset
                    sed_fproc.write(full_chunk->sed_features);
                    doa_fproc.write(full_chunk->doa_features);
                    // Write an empty task pointer back to the free pool for reuse.
                    void* writePtr;
                    if (rb_acquire_write_task_ptr(free_pool, writePtr)) {
                        std::memcpy(writePtr, &full_chunk, sizeof(Chunk*));
                        ma_rb_commit_write(&free_pool, sizeof(Chunk*));
                    }
                }
                else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        };
    public:
        int count = 0;


        Writer(const std::string& path, const SystemConfig& config, const bool training_mode)
			: config(config), training_mode(training_mode),
            root(z5::filesystem::handle::File(path)),
            sed_fproc(DatasetProcessor(root, "sed_features", 1, config, training_mode)),
            doa_fproc(DatasetProcessor(root, "doa_features", 5, config, training_mode)) {
            if (training_mode) {
			    // Initialize ring buffers for task management
                ma_rb_init_ex(sizeof(Chunk*), task_limit, 0, nullptr, nullptr, &task_queue);
                ma_rb_init_ex(sizeof(Chunk*), task_limit, 0, nullptr, nullptr, &free_pool);

                for (int i = 0; i < task_limit; ++i) {
                    Chunk* t = new Chunk(config.frame_time_seq, config.mel_bins);
                
                    void* writePtr; size_t sz = sizeof(Chunk*);
                    if (ma_rb_acquire_write(&free_pool, &sz, &writePtr) == MA_SUCCESS && sz == sizeof(Chunk*)) {
                        std::memcpy(writePtr, &t, sizeof(Chunk*));
                        ma_rb_commit_write(&free_pool, sizeof(Chunk*));
                    }
                }

                worker = std::thread(&Writer::thread_loop, this);
            }
        }

		bool add_frame(xt::xtensor<float, 2>& sed_features, xt::xtensor<float, 2>& doa_features) {
            if (active_chunk == nullptr && training_mode) {
                // Acquire a new task pointer from the free pool
                void* readPtr;
                if (rb_acquire_read_task_ptr(free_pool, readPtr)) {
                    std::memcpy(&active_chunk, readPtr, sizeof(Chunk*));
                    ma_rb_commit_read(&free_pool, sizeof(Chunk*));
                }
                else {
                    // Buffer Underrun
                    return false;
                }
            }

            xt::view(active_chunk->sed_features, current_local_idx) = sed_features;
            xt::view(active_chunk->doa_features, current_local_idx) = doa_features;
            current_local_idx++;

            if (current_local_idx >= config.frame_time_seq && training_mode) {
                void* writePtr;
                if (rb_acquire_write_task_ptr(task_queue, writePtr)) {
                    std::memcpy(writePtr, &active_chunk, sizeof(Chunk*));
                    ma_rb_commit_write(&task_queue, sizeof(Chunk*));
                }

				// Drop active task pointer and local index for the next cssssssssssssssssssssssssssssssssssssssssssssssssssssssssssshunk
                active_chunk = nullptr;
                current_local_idx = 0;
				count += config.frame_time_seq;
            }

			if (current_local_idx >= (config.frame_time_seq/10) && !training_mode) {
                // Inference branch, updating x_in every 1/10 of a window to have a dynamic prediction


                // Reset
                current_local_idx = 0;
            }

            return true;
        }

        ~Writer() {
            if (worker.joinable()) worker.join();

			// Clean up the active chunk if it exists
            if (active_chunk != nullptr) delete active_chunk;

            // Clean up any remaining tasks in the free pool and task queue
            void* readPtr; Chunk* chunk_to_delete = nullptr;
            while (rb_acquire_read_task_ptr(free_pool, readPtr)) {
                std::memcpy(&chunk_to_delete, readPtr, sizeof(Chunk*));
                ma_rb_commit_read(&free_pool, sizeof(Chunk*));
                delete chunk_to_delete;
                chunk_to_delete = nullptr;
            }

            while (rb_acquire_read_task_ptr(task_queue, readPtr)) {
                std::memcpy(&chunk_to_delete, readPtr, sizeof(Chunk*));
                ma_rb_commit_read(&task_queue, sizeof(Chunk*));
                delete chunk_to_delete;
                chunk_to_delete = nullptr;
            }

            ma_rb_uninit(&task_queue);
            ma_rb_uninit(&free_pool);
        }
};