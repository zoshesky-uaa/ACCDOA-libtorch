#pragma once
#include <xtensor/containers/xarray.hpp>
#include <vector>
#include <miniaudio.h>
#include <z5/factory.hxx>
#include <z5/attributes.hxx>
#include "z5/filesystem/handle.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "../config.h"



struct Task {
    // We use 2D xtensors: [Frames per Chunk, Feature Bins]
    xt::xtensor<float, 2> mel;
    xt::xtensor<float, 2> iv_x;
    xt::xtensor<float, 2> iv_y;
    z5::types::ShapeType offset_coord;

    Task(int chunk_sz, int mel_bins, int fft_bins)
        : mel({ (size_t)chunk_sz, (size_t)mel_bins }),
        iv_x({ (size_t)chunk_sz, (size_t)fft_bins }),
        iv_y({ (size_t)chunk_sz, (size_t)fft_bins }),
        offset_coord({ 0, 0 }) {}
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
        size_t sz = sizeof(Task*);

        void thread_loop() {
            while (config.on) {
                void* readPtr;
                Task* full_task = nullptr;
               
                if (ma_rb_acquire_read(&task_queue, &sz, &readPtr) == MA_SUCCESS && sz == sizeof(Task*)) {
                    Task* full_task;
                    std::memcpy(&full_task, readPtr, sizeof(Task*));
                    ma_rb_commit_read(&task_queue, sizeof(Task*));

                    // Write the chunks to the respective datasets at the correct offset
                    z5::multiarray::writeSubarray<float>(*mel_ds, full_task->mel, full_task->offset_coord.begin());
                    z5::multiarray::writeSubarray<float>(*iv_x_ds, full_task->iv_x, full_task->offset_coord.begin());
                    z5::multiarray::writeSubarray<float>(*iv_y_ds, full_task->iv_y, full_task->offset_coord.begin());
                    
                    // Write an empty task pointer back to the free pool for reuse.
                    void* writePtr;
                    if (ma_rb_acquire_write(&free_pool, &sz, &writePtr) == MA_SUCCESS && sz == sizeof(Task*)) {
                        std::memcpy(writePtr, &full_task, sizeof(Task*));
                        ma_rb_commit_write(&free_pool, sizeof(Task*));
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

            // We assume the directory to be a zarr provided without the feature subgroup and datasets.
            z5::filesystem::handle::File f(path);

            // Features sub-group
            z5::createGroup(f, "features");
            z5::filesystem::handle::Group features_group(f, "features");

			// Define the shape and chunk size for the features dataset
            std::vector<size_t> mel_shape = { config.frame_max, (size_t)config.mel_bins };
            std::vector<size_t> mel_chunks = { (size_t)chunk_size, (size_t)config.mel_bins };
            std::vector<size_t> fft_shape = { config.frame_max, (size_t)config.fft_bins };
            std::vector<size_t> fft_chunks = { (size_t)chunk_size, (size_t)config.fft_bins };

			// Define compression options for the datasets
            z5::types::CompressionOptions cOpts;
            cOpts["codec"] = std::string("zstd");
            cOpts["level"] = 3;
            cOpts["shuffle"] = 1;
            cOpts["blocksize"] = 0;

            // Create datasets for log-mel and intensity vectors
            mel_ds = z5::createDataset(
                features_group, "mel", "float32",
                mel_shape, mel_chunks,
                "blosc", cOpts
            );
			iv_x_ds = z5::createDataset(
                features_group, "iv_x", "float32",
                fft_shape, fft_chunks,
                "blosc", cOpts
			);
            iv_y_ds = z5::createDataset(
                features_group, "iv_y", "float32",
                fft_shape, fft_chunks,
                "blosc", cOpts
            );

			// Initialize ring buffers for task management
            ma_rb_init_ex(sizeof(Task*), task_limit, 0, nullptr, nullptr, &task_queue);
            ma_rb_init_ex(sizeof(Task*), task_limit, 0, nullptr, nullptr, &free_pool);

            for (int i = 0; i < task_limit; ++i) {
                Task* t = new Task(chunk_size, config.mel_bins, config.fft_bins);
                
                void* writePtr;
                if (ma_rb_acquire_write(&free_pool, &sz, &writePtr) == MA_SUCCESS && sz == sizeof(Task*)) {
                    std::memcpy(writePtr, &t, sizeof(Task*));
                    ma_rb_commit_write(&free_pool, sizeof(Task*));
                }
            }

            worker = std::thread(&Writer::thread_loop, this);
        }

		bool addFrame(const std::vector<float>& features) {
            if (active_task == nullptr) {
                // Acquire a new task pointer from the free pool
                void* readPtr;
                if (ma_rb_acquire_read(&free_pool, &sz, &readPtr) == MA_SUCCESS && sz == sizeof(Task*)) {
                    std::memcpy(&active_task, readPtr, sizeof(Task*));
                    ma_rb_commit_read(&free_pool, sizeof(Task*));
                    // Set the offset for this task based on the global count of frames processed so far
                    active_task->offset_coord = { (size_t)count, 0 };
                }
                else {
                    // Buffer Underrun
                    return false;
                }
            }

			// Copy features into active task's xtensor
            std::copy(features.begin(),
                features.begin() + config.mel_bins,
                &active_task->mel(current_local_idx, 0));

            std::copy(features.begin() + config.mel_bins,
                features.begin() + config.mel_bins + config.fft_bins,
                &active_task->iv_x(current_local_idx, 0));

            std::copy(features.begin() + config.mel_bins + config.fft_bins,
                features.end(),
                &active_task->iv_y(current_local_idx, 0));

            current_local_idx++;

            if (current_local_idx >= chunk_size) {
                void* writePtr;
                if (ma_rb_acquire_write(&task_queue, &sz, &writePtr) == MA_SUCCESS && sz == sizeof(Task*)) {
                    std::memcpy(writePtr, &active_task, sizeof(Task*));
                    ma_rb_commit_write(&task_queue, sizeof(Task*));
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
            while (ma_rb_acquire_read(&free_pool, &sz, &readPtr) == MA_SUCCESS && sz == sizeof(Task*)) {
                std::memcpy(&task_to_delete, readPtr, sizeof(Task*));
                ma_rb_commit_read(&free_pool, sizeof(Task*));
                delete task_to_delete;
            }
            
            while (ma_rb_acquire_read(&task_queue, &sz, &readPtr) == MA_SUCCESS && sz == sizeof(Task*)) {
                std::memcpy(&task_to_delete, readPtr, sizeof(Task*));
                ma_rb_commit_read(&task_queue, sizeof(Task*));
                delete task_to_delete;
			}

            ma_rb_uninit(&task_queue);
            ma_rb_uninit(&free_pool);
        }
};