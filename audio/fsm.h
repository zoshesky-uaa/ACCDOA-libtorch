#pragma once
#include <vector>
#include <miniaudio.h>
#include <z5/factory.hxx>
#include <z5/attributes.hxx>
#include "../ACCDOA-libtorch.h"
#include <xtensor/xarray.hpp>


struct Task {
    std::vector<float> mel;
    std::vector<float> iv_x;
    std::vector<float> iv_y;
    size_t offset;

    Task(int chunk_sz, int mel_bins, int fft_bins)
        : mel(chunk_sz* mel_bins, 0.0f),
        iv_x(chunk_sz* fft_bins, 0.0f),
        iv_y(chunk_sz* fft_bins, 0.0f),
        offset(0) {
    }
};

class Writer {
    private:
        // z5 uses xtensor-like views to handle multi-dimensional data
        std::shared_ptr<z5::Dataset> mel_ds, iv_x_ds, iv_y_ds;


        const SystemConfig& config;
        int chunk_size = 500;
		int current_local_idx = 0;

		// Maximum number of chunks (or tasks) that can be queued for writing at once.
		int task_limit = 8; 
		Task* active_task = nullptr;
		ma_rb task_queue;
        ma_rb free_pool;
        std::thread worker;

        void thread_loop() {
            while (config.on) {
                void* readPtr;
                Task* full_task = nullptr;
                if (ma_rb_acquire_read(&task_queue, 1, &readPtr) == MA_SUCCESS) {
                    std::memcpy(&full_task, readPtr, sizeof(Task*));
                    ma_rb_commit_read(&task_queue, 1);

                    // Write the chunks to the respective datasets at the correct offset
                    mel_ds->writeSubarray({ full_task->offset, 0 }, { (size_t)chunk_size, (size_t)config.mel_bins }, full_task->mel.data());
                    iv_x_ds->writeSubarray({ full_task->offset, 0 }, { (size_t)chunk_size, (size_t)config.fft_bins }, full_task->iv_x.data());
                    iv_y_ds->writeSubarray({ full_task->offset, 0 }, { (size_t)chunk_size, (size_t)config.fft_bins }, full_task->iv_y.data());

                    // Write an empty task pointer back to the free pool for reuse.
                    void* writePtr;
                    if (ma_rb_acquire_write(&free_pool, 1, &writePtr) == MA_SUCCESS) {
                        std::memcpy(writePtr, &full_task, sizeof(Task*));
                        ma_rb_commit_write(&free_pool, 1);
                    }
                }
                else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        };
    public:
        int count = 0;
        Writer(const std::string& path, const SystemConfig& config)
            : config(config) {

            // Compression options
            z5::DatasetOptions options;
            options.compression = z5::Compression("blosc", { {"codec", "zstd"}, {"level", 3} });

            // 3. Create the root group and features sub-group
            auto root_handle = z5::filesystem::createGroup(path, true);
            auto features_group = z5::filesystem::createGroup(path + "/features", true);

			// Define the shape and chunk size for the features dataset
            std::vector<size_t> mel_shape = { config.frame_max, (size_t)config.mel_bins };
            std::vector<size_t> mel_chunks = { (size_t)chunk_size, (size_t)config.mel_bins };
            std::vector<size_t> fft_shape = { config.frame_max, (size_t)config.fft_bins };
            std::vector<size_t> fft_chunks = { (size_t)chunk_size, (size_t)config.fft_bins };

			// Create datasets for log-mel and intensity vectors
            mel_ds = z5::filesystem::createDataset(features_group, "mel", "float32", mel_shape, mel_chunks, options);
            iv_x_ds = z5::filesystem::createDataset(features_group, "iv_x", "float32", fft_shape, fft_chunks, options);
            iv_y_ds = z5::filesystem::createDataset(features_group, "iv_y", "float32", fft_shape, fft_chunks, options);

			// Initialize ring buffers for task management
            ma_rb_init(sizeof(Task*), task_limit, nullptr, nullptr, &task_queue);
            ma_rb_init(sizeof(Task*), task_limit, nullptr, nullptr, &free_pool);

            for (int i = 0; i < task_limit; ++i) {
                Task* new_task = new Task(chunk_size, config.mel_bins, config.fft_bins);
                void* writePtr;
                if (ma_rb_acquire_write(&free_pool, 1, &writePtr) == MA_SUCCESS) {
                    std::memcpy(writePtr, &new_task, sizeof(Task*));
                    ma_rb_commit_write(&free_pool, 1);
                }
            }

            worker = std::thread(&Writer::thread_loop, this);
        }

		bool addFrame(const std::vector<float>& features) {
            if (active_task == nullptr) {
                // Acquire a new task pointer from the free pool
                void* readPtr;
                if (ma_rb_acquire_read(&free_pool, 1, &readPtr) == MA_SUCCESS) {
                    std::memcpy(&active_task, readPtr, sizeof(Task*));
                    ma_rb_commit_read(&free_pool, 1);
                    // Set the offset for this task based on the global count of frames processed so far
                    active_task->offset = count;
                }
                else {
                    // Buffer Underrun
                    return false;
                }
            }
			// Reset local index for the new task
            float* mel_dest = active_task->mel.data() + (current_local_idx * config.mel_bins);
            float* ivx_dest = active_task->iv_x.data() + (current_local_idx * config.fft_bins);
            float* ivy_dest = active_task->iv_y.data() + (current_local_idx * config.fft_bins);
                
			// Copy the features into the task's buffers
            const float* src = features.data();
            std::memcpy(mel_dest, src, config.mel_bins * sizeof(float));
            std::memcpy(ivx_dest, src + config.mel_bins, config.fft_bins * sizeof(float));
            std::memcpy(ivy_dest, src + config.mel_bins + config.fft_bins, config.fft_bins * sizeof(float));
			current_local_idx++;

            if (current_local_idx >= chunk_size) {
                void* writePtr;
                if (ma_rb_acquire_write(&task_queue, 1, &writePtr) == MA_SUCCESS) {
                    std::memcpy(writePtr, &active_task, sizeof(Task*));
                    ma_rb_commit_write(&task_queue, 1);
                }

				// Drop active task pointer and local index for the next chunk
                active_task = nullptr;
                current_local_idx = 0;
				count += chunk_size;
            }
            return true;
        }

        ~Writer() {
            if (worker.joinable()) worker.join();

			// Clean up the active task if it exists
            if (active_task != nullptr) delete active_task;
            void* readPtr;
            Task* task_to_delete = nullptr;

			// Clean up any remaining tasks in the free pool and task queue
            while (ma_rb_acquire_read(&free_pool, 1, &readPtr) == MA_SUCCESS) {
                std::memcpy(&task_to_delete, readPtr, sizeof(Task*));
                ma_rb_commit_read(&free_pool, 1);
                delete task_to_delete;
            }
            
            while (ma_rb_acquire_read(&task_queue, 1, &readPtr) == MA_SUCCESS) {
                std::memcpy(&task_to_delete, readPtr, sizeof(Task*));
                ma_rb_commit_read(&task_queue, 1);
                delete task_to_delete;
			}
        }
};