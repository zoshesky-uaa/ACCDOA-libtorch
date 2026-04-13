#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <vector>
#include <kfr/base.hpp>
#include <kfr/dsp.hpp>
#include <kfr/dft.hpp>

using namespace kfr;

class FeatureExtractor {
	private:
		const int fft_size = 1024;
		const int hop_length = 512;
		const int num_bins = fft_size / 2 + 1;
		const int channels = 4;
		const float log_max_vol = std::log1p(fft_size / 2.0f);

		// KFR plan for real-valued DFT, initialized with the FFT size
		dft_plan_real<float> plan = dft_plan_real<float>(fft_size);
		// Temporary buffer for the DFT, size determined by the plan
		univector<u8> temp_buffer = univector<u8>(plan.temp_size);
		// Hann window for the FFT
		univector<float> window = window_hann<float>(fft_size);
		// Linear history buffer to hold 2 frame sets of audio for each channel
		std::vector<univector<float>> buffers =
			std::vector<univector<float>>(channels, univector<float>(fft_size, 0.0f));

		std::vector<std::vector<float>> prepare_audio(float* buffer) {
			std::vector<std::vector<float>> planar_audio(channels, std::vector<float>(hop_length, 0.0f));
			for (int frame = 0; frame < hop_length; ++frame) {
				for (int channel = 0; channel < channels; ++channel) {
					// Calculate the 1D index of the interleaved buffer
					int interleaved_index = (frame * channels) + channel;

					// Assign the value to the correct channel and frame
					planar_audio[channel][frame] = buffer[interleaved_index];
				}
			}
			return planar_audio;
		};
	public:
		std::vector<float> extract(float* buffer) {
			auto audio = prepare_audio(buffer);
			// Output vector
			std::vector<univector<complex<float>>> freqs(channels, univector<complex<float>>(num_bins));
			for (int channel = 0; channel < channels; ++channel) {
				// Memmove (shift left) the last set of frames over
				std::memmove(buffers[channel].data(),
					buffers[channel].data() + hop_length,
					(fft_size - hop_length) * sizeof(float));

				// Copy the a new set of frames at the end
				std::copy(audio[channel].begin(),
					audio[channel].end(),
					buffers[channel].begin() + (fft_size - hop_length));

				// Apply window function to the full buffer
				univector<float> windowed_signal = buffers[channel] * window;

				// KFR signature: execute(output, input, temp_buffer)
				plan.execute(freqs[channel], windowed_signal, temp_buffer);
			}
			// Output feature vector, 7 features
			// 1. Logarithmic amplitude for each of the 4 channels (1-4)
			// 2. Phase for each of the 4 channels (relative to the first channel) (5-7)
			std::vector<float> features(7 * num_bins);

			// Pointer to current position in the output feature vector
			float* current_feature_ptr = features.data();

			for (int channel = 0; channel < channels; ++channel) {
				// Logarithmic normalization over the current features
				// log1p(freq_amplitude) / log1p(max_val)
				univector_ref<float> amp_out(current_feature_ptr, num_bins);
				amp_out = log(cabs(freqs[channel]) + 1.0f) / log_max_vol;
				current_feature_ptr += num_bins;
			}

			univector<float> ref_phase = carg(freqs[0]);
			for (int channel = 1; channel < channels; ++channel) {
				univector_ref<float> phase_out(current_feature_ptr, num_bins);
				phase_out = carg(freqs[channel]) - ref_phase;

				//Normalize phase to [-pi, pi]
				for (int i = 0; i < num_bins; ++i) {
					float wrapped = std::fmod(phase_out[i] + M_PI, 2.0 * M_PI);
					if (wrapped < 0.0f) {
						wrapped += 2.0 * M_PI;
					}
					wrapped -= M_PI;

					// Normalize to [-1, 1]
					phase_out[i] = wrapped / M_PI;
				}

				current_feature_ptr += num_bins;
			}

			return features;
		}
};