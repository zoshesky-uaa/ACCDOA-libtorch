#pragma once
#include <cmath>
#include <algorithm>
#include <vector>
#include "audio.h"
#include "../config.h"
// kfr
#include <kfr/base.hpp>
#include <kfr/dsp.hpp>
#include <kfr/dft.hpp>
#include <kfr/audio.hpp>
#include <kfr/simd.hpp>
// xtensor
#include <xtensor/containers/xarray.hpp>
#include "z5/multiarray/xtensor_access.hxx"

class MelFilterBank {
	private:
		static constexpr float f_sp = 200.0f / 3.0f;
		static constexpr float min_log_hz = 1000.0f;
		static constexpr float min_log_mel = 15.0f; // 1000.0 / f_sp
		static constexpr float logstep = 0.068812317f; // ln(6.4) / 27.0
		float hz_to_mel(float freq) {
			if (freq < min_log_hz) return freq / f_sp;
			return min_log_mel + std::log(freq / min_log_hz) / logstep;
		}

		float mel_to_hz(float mel) {
			if (mel < min_log_mel) return mel * f_sp;
			return min_log_hz * std::exp(logstep * (mel - min_log_mel));
		}
	public:
		std::vector<kfr::univector<float>> filters;
		MelFilterBank(size_t sr, size_t n_fft, size_t n_mels) {
			const size_t n_bins = n_fft / 2 + 1;
			filters.assign(n_mels, kfr::univector<float>(n_bins, 0.0f));

			// 1. Generate FFT bin frequencies (linear)
			kfr::univector<float> fft_freqs(n_bins);
			for (size_t i = 0; i < n_bins; ++i) fft_freqs[i] = (static_cast<float>(i) * sr) / n_fft;

			// 2. Generate Mel-spaced frequencies in Hz
			float fmin = 0.0f;
			float fmax = static_cast<float>(sr) / 2.0f;
			float min_mel = hz_to_mel(fmin);
			float max_mel = hz_to_mel(fmax);

			std::vector<float> mel_f(n_mels + 2);
			for (size_t i = 0; i < n_mels + 2; ++i) {
				float m = min_mel + i * (max_mel - min_mel) / (n_mels + 1);
				mel_f[i] = mel_to_hz(m);
			}

			// 3. Build triangular filters
			for (size_t i = 0; i < n_mels; ++i) {
				float lower_hz = mel_f[i];
				float center_hz = mel_f[i + 1];
				float upper_hz = mel_f[i + 2];

				float f_diff_lower = center_hz - lower_hz;
				float f_diff_upper = upper_hz - center_hz;

				// Slaney normalization factor (enorm)
				float enorm = 2.0f / (upper_hz - lower_hz);

				for (size_t j = 0; j < n_bins; ++j) {
					float f = fft_freqs[j];
					if (f > lower_hz && f < upper_hz) {
						float weight_lower = (f - lower_hz) / f_diff_lower;
						float weight_upper = (upper_hz - f) / f_diff_upper;
						filters[i][j] = std::max(0.0f, std::min(weight_lower, weight_upper)) * enorm;
					}
				}
			}
		}
};

class FeatureExtractor {
	private:
		const SystemConfig& config;
		xt::xtensor<float, 2>& sed_features;
		xt::xtensor<float, 2>& doa_features;
		const float log_max_vol;

		// constants
		static constexpr float pi = kfr::c_pi<float>;
		static constexpr float pi2 = 2.0f * pi;

		// KFR plan for real-valued DFT, initialized with the FFT size
		kfr::dft_plan_real<float> plan = kfr::dft_plan_real<float>(config.fft_size);	

		// Temporary buffer for the DFT, size determined by the plan
		kfr::univector<uint8_t> temp_buffer = kfr::univector<uint8_t>(plan.temp_size);

		// Hann window for the FFT
		kfr::univector<float> window = kfr::window_hann<float>(config.fft_size);

		// Mel filter bank for 64 Mel bands, initialized with the 16kHz sample rate
		MelFilterBank mel = MelFilterBank(config.sample_rate, config.fft_size, config.mel_bins);

		// typedefs
		using fftdata = kfr::univector<kfr::complex<float>>;
		std::array<fftdata, 4> ch_freqs = {
			fftdata(config.fft_bins, 0.0f),
			fftdata(config.fft_bins, 0.0f),
			fftdata(config.fft_bins, 0.0f),
			fftdata(config.fft_bins, 0.0f)
		};
		fftdata w_freq = fftdata(config.fft_bins, 0.0f);
		fftdata x_freq = fftdata(config.fft_bins, 0.0f);
		fftdata y_freq = fftdata(config.fft_bins, 0.0f);
		using buffer = kfr::univector<float>;
		buffer planar_buffer = buffer(config.fft_size * config.channels, 0.0f);

		// Scratch space for intermediate calculations
		kfr::univector<float> r_window = kfr::univector<float>(config.fft_size);
		kfr::univector<float> mag_temp = kfr::univector<float>(config.fft_bins);
		kfr::univector<float> mel_temp = kfr::univector<float>(config.mel_bins);
		kfr::univector<float> linear_temp = kfr::univector<float>(config.fft_bins);
		kfr::univector<kfr::complex<float>> conj_temp = kfr::univector<kfr::complex<float>>(config.fft_bins);

		void log_mel_normalize(kfr::univector_ref<kfr::complex<float>> freqs, float* mel_ptr) {
			mag_temp = kfr::cabs(freqs);
			for (size_t m = 0; m < config.mel_bins; ++m) {
				mel_temp[m] = kfr::dotproduct(mag_temp, mel.filters[m]);
			}
			kfr::univector_ref<float> mel_out(mel_ptr, config.mel_bins);
			mel_out = kfr::log10(mel_temp + 1e-7f);
		};

		void calc_mel_iv(kfr::univector_ref<kfr::complex<float>> w,
			kfr::univector_ref<kfr::complex<float>> conj_w,
			kfr::univector_ref<kfr::complex<float>> directional,
			float* iv_ptr) {
			kfr::univector_ref<float> iv_out(iv_ptr, config.mel_bins);
			linear_temp = kfr::real(conj_w * directional) / (kfr::cabssqr(w) + 1e-7f);
			// Compress the linear intensity vector into Mel bands for consistency
			for (size_t m = 0; m < config.mel_bins; ++m) {
				// Sum of the current triangular filter weights
				float filter_sum = kfr::sum(mel.filters[m]);

				// Dot product the linear IV with the filter, then normalize by the filter sum
				iv_out[m] = kfr::dotproduct(linear_temp, mel.filters[m]) / (filter_sum + 1e-7f);
			}
		}

		void extract(std::vector<float>& buffer) {
			// Assumption: Hop length is half the FFT size.
			// Position in the channel block for the history buffer.
			float* b_ptr = planar_buffer.data();
			for (int ch = 0; ch < config.channels; ++ch) {
				// Create a reference to the current channel's buffer block
				kfr::univector_ref<float> ch_buffer(b_ptr, config.fft_size);
				// Shift the buffer to the left for the channel block
				ch_buffer.slice(0, config.history_size) = ch_buffer.slice(config.hop_length, config.history_size);
				// 2. Insert New Data: Place the hop_length new samples at the end
				kfr::strided_channel<float> ch_data{ buffer.data() + ch, config.hop_length, (size_t)config.channels };
				ch_buffer.slice(config.history_size, config.hop_length) = ch_data;
				// Apply window function to the whole channel buffer and execute the FFT
				r_window = ch_buffer * window;
				plan.execute(ch_freqs[ch], r_window, temp_buffer);
				b_ptr += config.fft_size;
			}

			// Compute the spatial features (W, x, y) from the channel FFTs
			w_freq = ch_freqs[1] + ch_freqs[0] + ch_freqs[3] + ch_freqs[2]; // Omni
			x_freq = (ch_freqs[1] + ch_freqs[0]) - (ch_freqs[3] + ch_freqs[2]); // Front-Back
			y_freq = (ch_freqs[1] + ch_freqs[3]) - (ch_freqs[0] + ch_freqs[2]); // Left-Right
			kfr::univector_ref<kfr::complex<float>> conj_w(conj_temp.data(), config.fft_bins);
			conj_w = kfr::cconj(w_freq);

			// --- 1. SED Features (1 Channel: Logmel Omni) ---
			log_mel_normalize(w_freq, xt::row(sed_features, 0).data());

			// --- 2. DOAE Features (5 Channels) ---
			// 1-3. Logmel features for W, X, Y
			log_mel_normalize(w_freq, xt::row(doa_features, 0).data());
			log_mel_normalize(x_freq, xt::row(doa_features, 1).data());
			log_mel_normalize(y_freq, xt::row(doa_features, 2).data());

			// 4-5. Intensity Vectors for X, Y
			calc_mel_iv(w_freq, conj_w, x_freq, xt::row(doa_features, 3).data());
			calc_mel_iv(w_freq, conj_w, y_freq, xt::row(doa_features, 4).data());
		};

	public:
		FeatureExtractor(const SystemConfig& config, xt::xtensor<float, 2>& sed, xt::xtensor<float, 2>& doa)
			: config(config), 
			log_max_vol(std::log1p(static_cast<float>(config.fft_size) / 2.0f)),
			sed_features(sed),
			doa_features(doa) {}

		bool feature_extract(AudioDevice& audioDevice) {
			std::vector<float> buffer(static_cast<size_t>(audioDevice.framelimit * audioDevice.channels));
			while (!audioDevice.read(buffer.data())) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			extract(buffer);
			return true;
		};
};