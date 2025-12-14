/// @file
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "pitch_analyzer.h"

using namespace std;

namespace upc {

  // ------------------------------------------------------------
  // Autocorrelation (FIX IMPORTANT: use x.size(), not r.size())
  // ------------------------------------------------------------
  void PitchAnalyzer::autocorrelation(const vector<float> &x, vector<float> &r) const {
    const size_t N = x.size();
    const size_t L = std::min(r.size(), N);

    for (size_t l = 0; l < L; ++l) {
      double acc = 0.0;
      for (size_t n = 0; n + l < N; ++n) {
        acc += (double)x[n] * (double)x[n + l];
      }
      r[l] = (float)acc;
    }

    // per seguretat si r és més gran que N
    for (size_t l = L; l < r.size(); ++l)
      r[l] = 0.0f;
  }

  // ------------------------------------------------------------
  // Window
  // ------------------------------------------------------------
  void PitchAnalyzer::set_window(Window win_type) {
    if (frameLen == 0) return;

    window.resize(frameLen);
    constexpr float PI = 3.14159265358979323846f;

    switch (win_type) {
      case HAMMING:
        for (unsigned int i = 0; i < frameLen; i++) {
          window[i] = 0.54f - 0.46f * std::cos((2.0f * PI * (float)i) / (float)(frameLen - 1));
        }
        break;

      case RECT:
        window.assign(frameLen, 1.0f);
        break;

      default:
        window.assign(frameLen, 1.0f);
        break;
    }
  }

  // ------------------------------------------------------------
  // f0 range
  // ------------------------------------------------------------
  void PitchAnalyzer::set_f0_range(float min_F0, float max_F0) {
    npitch_min = (unsigned int)(samplingFreq / max_F0);
    if (npitch_min < 2) npitch_min = 2;

    npitch_max = 1 + (unsigned int)(samplingFreq / min_F0);

    // frameLen should include at least 2*T0
    if (npitch_max > frameLen / 2)
      npitch_max = frameLen / 2;
  }

  // ------------------------------------------------------------
  // Voiced/Unvoiced decision (MENYS AGRESSIU + usa umaxnorm)
  // ------------------------------------------------------------
  bool PitchAnalyzer::unvoiced(float pot_db, float r1norm, float rmaxnorm) const {
    const float MIN_POT_DB = -35.0f;  // abans -10 (massa alt)
    const float MIN_R1NORM = 0.10f;   // abans 0.80 (molt agressiu)

    // IMPORTANT:
    // Aquí faig servir "umaxnorm" com a membre de classe (llindar passat al constructor).
    // Si al teu .h es diu umaxnorm_ o similar, canvia aquesta línia.
    const float MIN_RMAXNORM = umaxnorm;

    if (pot_db < MIN_POT_DB) return true;
    if (rmaxnorm < MIN_RMAXNORM) return true;

    // Aquest criteri sovint fa més mal que bé si és alt; el deixem molt suau.
    if (r1norm < MIN_R1NORM) return true;

    return false; // voiced
  }

  // ------------------------------------------------------------
  // compute_pitch: millor cerca de lag + parabolic interpolation
  // ------------------------------------------------------------
  float PitchAnalyzer::compute_pitch(vector<float> &x) const {
    if (x.size() != frameLen) return -1.0f;

    // Window input frame
    for (unsigned int i = 0; i < x.size(); ++i)
      x[i] *= window[i];

    vector<float> r(npitch_max);
    autocorrelation(x, r);

    const float eps = 1e-12f;
    float r0 = std::max(r[0], eps);

    // Potència en dB (normalitzada una mica millor)
    float pot_db = 10.0f * std::log10((r0 / (float)frameLen) + eps);

    float r1norm = (r.size() > 1) ? (r[1] / r0) : 0.0f;

    // 1) troba un punt d'inici robust: primer mínim local a partir de npitch_min
    unsigned int lag_min = npitch_min;
    for (unsigned int l = npitch_min + 1; l + 1 < npitch_max; ++l) {
      if (r[l - 1] > r[l] && r[l] <= r[l + 1]) {
        lag_min = l;
        break;
      }
    }

    // 2) busca el màxim a [lag_min, npitch_max)
    unsigned int lag = lag_min;
    float rmax = r[lag_min];
    for (unsigned int l = lag_min + 1; l < npitch_max; ++l) {
      if (r[l] > rmax) {
        rmax = r[l];
        lag = l;
      }
    }

    float rmaxnorm = rmax / r0;

    if (unvoiced(pot_db, r1norm, rmaxnorm))
      return 0.0f;

    // 3) interpolació parabòlica per refinar el lag (millora MSE i estabilitat)
    float lag_f = (float)lag;
    if (lag > 0 && lag + 1 < npitch_max) {
      float y1 = r[lag - 1];
      float y2 = r[lag];
      float y3 = r[lag + 1];
      float denom = (y1 - 2.0f * y2 + y3);

      if (std::fabs(denom) > 1e-12f) {
        float delta = 0.5f * (y1 - y3) / denom;
        if (delta > -0.5f && delta < 0.5f)
          lag_f += delta;
      }
    }

    return (float)samplingFreq / lag_f;
  }

} // namespace upc
