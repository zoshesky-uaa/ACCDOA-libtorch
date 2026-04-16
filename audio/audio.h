#pragma once
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>
#include <stdlib.h>
#include <stdio.h>


enum class DeviceType {
	Playback,
	Recording
};

class AudioDevice {
	private:
		ma_pcm_rb ringBuffer;
		ma_context context;
		
		static void playback_data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {

		};

		static void capture_data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
			// Retrieve the class ring buffer from pUserData
			ma_pcm_rb* pRingBuffer = (ma_pcm_rb*)pDevice->pUserData;
			if (pRingBuffer == NULL || pInput == NULL) return;

			//Tracking pointer to current position in input buffer
			const float* pRunningInput = (const float*)pInput;
			ma_uint32 framesWrittenTotal = 0;

			while (framesWrittenTotal < frameCount) {
				ma_uint32 framesRemaining = frameCount - framesWrittenTotal;
				ma_uint32 framesToAcquire = framesRemaining;
				void* pWriteBuffer;

				// Acquires whatever is remaining and write it.
				ma_result r = ma_pcm_rb_acquire_write(pRingBuffer, &framesToAcquire, &pWriteBuffer);

				if (r != MA_SUCCESS || framesToAcquire == 0) break;

				// Copying frames to ring buffer
				size_t bytesToCopy = framesToAcquire * ma_get_bytes_per_frame(pDevice->capture.format, pDevice->capture.channels);
				memcpy(pWriteBuffer, pRunningInput, bytesToCopy);

				// Commit to move the write pointer forward
				ma_pcm_rb_commit_write(pRingBuffer, framesToAcquire);

				// Update offsets for the next iteration, each frame contains 1 sample per channel
				framesWrittenTotal += framesToAcquire;
				pRunningInput += (framesToAcquire * pDevice->capture.channels);
			}
		};
	public:		
		const ma_uint32 channels = 4;
		const ma_format sample_format = ma_format_f32;
		const ma_uint32 sample_rate = 16000;
		ma_device device;
		int framelimit;

		AudioDevice(DeviceType device_type, char* device_name, int tick_rate) {
			// Force WASAPI backend 
			ma_backend backends[] = { ma_backend_wasapi };
			ma_context_config contextConfig = ma_context_config_init();
			ma_result r = ma_context_init(backends, 1, &contextConfig, &context);
			if (r != MA_SUCCESS) throw std::runtime_error("Failed to initialize context: " + std::to_string(r));

			ma_device_info* pCaptureDevices;
			ma_uint32 captureDeviceCount;
			ma_device_info* pPlaybackDevices;
			ma_uint32 playbackDeviceCount;

			r = ma_context_get_devices(&context,
				&pPlaybackDevices,
				&playbackDeviceCount,
				&pCaptureDevices,
				&captureDeviceCount);
			if (r != MA_SUCCESS) {
				ma_context_uninit(&context);
				throw std::runtime_error("Failed to recieve device list: " + std::to_string(r));
			}

			int device_count; ma_device_info* devices; ma_device_config deviceConfig;
			if (device_type == DeviceType::Playback) {
				device_count = playbackDeviceCount;
				devices = pPlaybackDevices;
				ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
				deviceConfig.dataCallback = AudioDevice::playback_data_callback;
			} else {
				device_count = captureDeviceCount;
				devices = pCaptureDevices;
				ma_device_config deviceConfig = ma_device_config_init(ma_device_type_capture);
				deviceConfig.dataCallback = AudioDevice::capture_data_callback;
			}

			ma_device_info* selected_device = NULL;
			for (ma_uint32 i = 0; i < device_count; i++) {
				if (strstr(devices[i].name, device_name) != NULL) {
					selected_device = &devices[i];
					break;
				}
			}

			if (selected_device == NULL) {
				ma_context_uninit(&context);
				throw std::runtime_error("Failed to find device with name: " + std::string(device_name));
			}

			if (r != MA_SUCCESS) {
				ma_context_uninit(&context);
				throw std::runtime_error("Failed to initialize ring buffer: " + std::to_string(r));
			}

			// Retrieve the id and set the configuration for the device
			deviceConfig.capture.pDeviceID = &selected_device->id;
			deviceConfig.capture.shareMode = ma_share_mode_shared;
			deviceConfig.capture.format = sample_format;

			// Tick_rate should be roughly SAMPLING_FREQUENCY/(FFT_SIZE/2) or 512 for 1024 point FFT
			framelimit = sample_rate / tick_rate;
			deviceConfig.periodSizeInFrames = framelimit;

			// Configuration default, match your microphone settings to match else it will not work.
			r = ma_pcm_rb_init(
				sample_format,
				channels,
				(framelimit * 64),
				NULL,
				NULL,
				&ringBuffer
			);

			deviceConfig.capture.channels = channels;
			deviceConfig.sampleRate = sample_rate;
			// Passes a reference to the ring buffer to pUserData so it can be access in a C-callback function
			deviceConfig.pUserData = &ringBuffer;
			ma_channel quadChannelMap[4] = {
			MA_CHANNEL_FRONT_LEFT,
			MA_CHANNEL_FRONT_RIGHT,
			MA_CHANNEL_BACK_LEFT,
			MA_CHANNEL_BACK_RIGHT
			};
			deviceConfig.capture.pChannelMap = quadChannelMap;

			// WASAPI settings, disables resampling and enforces low latency shared mode
			deviceConfig.wasapi.noAutoConvertSRC = MA_TRUE;      // Disable automatic sample rate conversion  
			deviceConfig.wasapi.noDefaultQualitySRC = MA_TRUE;   // Use highest quality resampling  
			deviceConfig.wasapi.usage = ma_wasapi_usage_pro_audio; // Set thread priority for pro audio

			r = ma_device_init(&context, &deviceConfig, &device);
			if (r != MA_SUCCESS) {
				ma_context_uninit(&context);
				throw std::runtime_error("Failed to initialize device: " + std::to_string(r));
			}

			r = ma_device_start(&device);
			if (r != MA_SUCCESS) {
				ma_device_uninit(&device);
				ma_context_uninit(&context);
				throw std::runtime_error("Failed to start device: " + std::to_string(r));
			}
		};

		bool read(float* buffer) {
			// Simple check to see if there is enough data to read else return
			if (ma_pcm_rb_available_read(&ringBuffer) < framelimit) return false;

			ma_uint32 framesReadTotal = 0;
			float* pRunningOutput = buffer;
			while (framesReadTotal < framelimit) {
				ma_uint32 framesToAcquire = framelimit - framesReadTotal;
				void* pReadBuffer;

				// Acquire the next chunk of data from the ring buffer
				ma_result r = ma_pcm_rb_acquire_read(&ringBuffer, &framesToAcquire, &pReadBuffer);
				if (r != MA_SUCCESS || framesToAcquire == 0) return false;

				// Copy the acquired data to the output buffer
				size_t bytesToCopy = framesToAcquire * ma_get_bytes_per_frame(sample_format, channels);
				memcpy(pRunningOutput, pReadBuffer, bytesToCopy);

				// Commit to move the read pointer forward
				ma_pcm_rb_commit_read(&ringBuffer, framesToAcquire);

				// Update offsets for the next iteration
				framesReadTotal += framesToAcquire;
				pRunningOutput += (framesToAcquire * channels);
			}

			if (framesReadTotal == framelimit) return true;
		};

		~AudioDevice() {
			ma_device_uninit(&device);
			ma_pcm_rb_uninit(&ringBuffer);
			ma_context_uninit(&context);
		};
};