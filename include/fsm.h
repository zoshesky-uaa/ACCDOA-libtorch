#pragma once
#include "config.h"
#include "dataset.h"
// miniaudio
#include <miniaudio.h>

struct Chunk {
    xt::xtensor<float, 3> sed_features;
    xt::xtensor<float, 3> doa_features;

    Chunk(int chunk_sz, int mel_bins)
        : sed_features({ (size_t)1, (size_t)chunk_sz, (size_t)mel_bins }),
        doa_features({ (size_t)5, (size_t)chunk_sz, (size_t)mel_bins }) { 
    }
};

class Reader {
    public:
        DatasetProcessor sed_labelset;
        DatasetProcessor doa_labelset;
		DatasetProcessor sed_featureset;
		DatasetProcessor doa_featureset;
        Reader(const std::string& path, const SystemConfig& config)
            :   sed_labelset(DatasetProcessor(z5::filesystem::handle::File(path), "sed_labels", ModelType::SED, config, DatasetType::SED_LABELS)),
                doa_labelset(DatasetProcessor(z5::filesystem::handle::File(path), "doa_labels", ModelType::DOA, config, DatasetType::DOA_LABELS)),
                sed_featureset(DatasetProcessor(z5::filesystem::handle::File(path), "sed_features", ModelType::SED, config, DatasetType::SED_FEATURES)),
                doa_featureset(DatasetProcessor(z5::filesystem::handle::File(path), "doa_features", ModelType::DOA, config, DatasetType::DOA_FEATURES))
        {}
};

class Writer {
    private:
        // z5 uses xtensor-like views to handle multi-dimensional data
        z5::filesystem::handle::File root;
        const SystemConfig& config;
		int current_local_idx = 0;

		// Maximum number of chunks (or tasks) that can be queued for writing at once.
        int task_limit = 50; 
		Chunk* active_chunk = nullptr;
		ma_rb task_queue;
        ma_rb free_pool;
        std::thread worker;

		// Helper functions to acquire task pointers from the ring buffers
        static bool rb_acquire_read_task_ptr(ma_rb& ring_buffer, void*& ptr) {
            size_t bytes = sizeof(Chunk*);
            return ma_rb_acquire_read(&ring_buffer, &bytes, &ptr) == MA_SUCCESS && bytes == sizeof(Chunk*);
        }

        static bool rb_acquire_write_task_ptr(ma_rb& ring_buffer, void*& ptr) {
            size_t bytes = sizeof(Chunk*);
            return ma_rb_acquire_write(&ring_buffer, &bytes, &ptr) == MA_SUCCESS && bytes == sizeof(Chunk*);
        }

        void thread_loop() {
            while (true) {
                void* readPtr;
                Chunk* full_chunk = nullptr;
                if (rb_acquire_read_task_ptr(task_queue, readPtr)) {
                    full_chunk = *reinterpret_cast<Chunk**>(readPtr);
                    ma_rb_commit_read(&task_queue, sizeof(Chunk*));
                    // Queue is fully drained. Safely exit the thread.
                    if (full_chunk == nullptr) {
                        break; 
                    }
                    // Write the chunks to the respective datasets at the correct offset
                    sed_featureset.write(full_chunk->sed_features);
                    doa_featureset.write(full_chunk->doa_features);
                    // Write an empty task pointer back to the free pool for reuse.
                    void* writePtr;
                    if (rb_acquire_write_task_ptr(free_pool, writePtr)) {
                        *reinterpret_cast<Chunk**>(writePtr) = full_chunk;
                        ma_rb_commit_write(&free_pool, sizeof(Chunk*));
                    }
                }
                else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        };
    public:
        size_t count = 0;
        DatasetProcessor sed_featureset;
        DatasetProcessor doa_featureset;

        Writer(const std::string& path, const SystemConfig& config)
			: config(config),
            root(z5::filesystem::handle::File(path)),
            sed_featureset(DatasetProcessor(root, "sed_features", ModelType::SED, config, DatasetType::SED_FEATURES)),
            doa_featureset(DatasetProcessor(root, "doa_features", ModelType::DOA, config, DatasetType::DOA_FEATURES)) {
			
            size_t total_buffer_size = sizeof(Chunk*) * task_limit;
            // Initialize ring buffers for task management
            ma_rb_init_ex(total_buffer_size, task_limit, 0, nullptr, nullptr, &task_queue);
            ma_rb_init_ex(total_buffer_size, task_limit, 0, nullptr, nullptr, &free_pool);
            
            for (int i = 0; i < task_limit; ++i) {
                auto pchunk = std::make_unique<Chunk>(config.frame_time_seq, config.mel_bins);
                void* writePtr; size_t size = sizeof(Chunk*);
                if (ma_rb_acquire_write(&free_pool, &size, &writePtr) == MA_SUCCESS && size == sizeof(Chunk*)) {
                    Chunk* raw_ptr = pchunk.release();
                    *reinterpret_cast<Chunk**>(writePtr) = raw_ptr;
                    ma_rb_commit_write(&free_pool, sizeof(Chunk*));
                }
                else {
                    throw std::runtime_error("Failed to initialize free pool with task pointers.");
                }  
            }

            worker = std::thread(&Writer::thread_loop, this);
        }

		bool add_frame(xt::xtensor<float, 2>& sed_features, xt::xtensor<float, 2>& doa_features) {
            if (active_chunk == nullptr) {
                // Acquire a new task pointer from the free pool
                void* readPtr;
                if (rb_acquire_read_task_ptr(free_pool, readPtr)) {
                    active_chunk = *reinterpret_cast<Chunk**>(readPtr);
                    ma_rb_commit_read(&free_pool, sizeof(Chunk*));
                }
                else {
                    // Buffer Underrun
                    return false;
                }
            }
            
            xt::noalias(xt::view(active_chunk->sed_features, xt::all(), current_local_idx, xt::all())) = sed_features;
            xt::noalias(xt::view(active_chunk->doa_features, xt::all(), current_local_idx, xt::all())) = doa_features;
            current_local_idx++;

            if (current_local_idx >= config.frame_time_seq) {
                void* writePtr;
                if (rb_acquire_write_task_ptr(task_queue, writePtr)) {
                    *reinterpret_cast<Chunk**>(writePtr) = active_chunk;
                    ma_rb_commit_write(&task_queue, sizeof(Chunk*));
                } else {
                    // Buffer Full! Return the active chunk to the free pool to prevent a memory leak.
                    std::cerr << "Warning: Disk write bottleneck. Dropping chunk." << '\n';
                    void* freePtr;
                    if (rb_acquire_write_task_ptr(free_pool, freePtr)) {
                        *reinterpret_cast<Chunk**>(freePtr) = active_chunk;
                        ma_rb_commit_write(&free_pool, sizeof(Chunk*));
                    }
                }

				// Drop active task pointer and local index for the next chunk
                active_chunk = nullptr;
                current_local_idx = 0;
				count += config.frame_time_seq;
            }
            return true;
        }

        ~Writer() {
            // Block untill room in the task queue to write a nullptr to kill the loop
            void* writePtr;
            while (!rb_acquire_write_task_ptr(task_queue, writePtr)) {
                std::this_thread::yield(); 
            }
            *reinterpret_cast<Chunk**>(writePtr) = nullptr;
            ma_rb_commit_write(&task_queue, sizeof(Chunk*));

            if (worker.joinable()) { worker.join(); }

			// Clean up the active chunk if it exists
            delete active_chunk;

            // Clean up any remaining tasks in the free pool and task queue
            void* readPtr; Chunk* chunk_to_delete = nullptr;
            while (rb_acquire_read_task_ptr(free_pool, readPtr)) {
                chunk_to_delete = *reinterpret_cast<Chunk**>(readPtr);
                ma_rb_commit_read(&free_pool, sizeof(Chunk*));
                delete chunk_to_delete;
            }

            ma_rb_uninit(&task_queue);
            ma_rb_uninit(&free_pool);
        }
};