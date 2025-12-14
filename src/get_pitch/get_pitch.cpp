#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string.h>
#include <errno.h>

#include "wavfile_mono.h"
#include "pitch_analyzer.h"
#include "docopt.h"

#define FRAME_LEN   0.030
#define FRAME_SHIFT 0.015

using namespace std;
using namespace upc;

static const char USAGE[] = R"(
get_pitch - Pitch Estimator

Usage:
    get_pitch [options] <input-wav> <output-txt>
    get_pitch (-h | --help)
    get_pitch --version

Options:
    -m, --umaxnorm FLOAT   llindar decisió sonor/sord per a rmaxnorm [default: 0.25]
    -h, --help             Show this screen
    --version              Show the version of the project
)";

static inline float median_of(vector<float> v) {
    if (v.empty()) return 0.0f;
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, const char *argv[]) {
    auto args = docopt::docopt(USAGE, {argv + 1, argv + argc}, true, "2.2");

    string input_wav  = args["<input-wav>"].asString();
    string output_txt = args["<output-txt>"].asString();
    float umaxnorm = stof(args["--umaxnorm"].asString());

    // Read input
    unsigned int rate;
    vector<float> x;
    if (readwav_mono(input_wav, rate, x) != 0) {
        cerr << "Error reading input file " << input_wav << " (" << strerror(errno) << ")\n";
        return -2;
    }

    int n_len   = (int)(rate * FRAME_LEN);
    int n_shift = (int)(rate * FRAME_SHIFT);

    // --- IMPORTANT: per ara deixem preprocessat "suau" per no matar voiced ---
    const bool  ENABLE_PREEMPH = false;   // <- abans pot fer baixar rmaxnorm
    const float PREEMPH_A      = 0.97f;

    const bool  ENABLE_CLIP    = false;   // <- abans pot fer baixar rmaxnorm
    const float CLIP_FRAC      = 0.10f;

    // Fallback (loose) per recuperar voiced
    float umaxnorm_loose = std::max(0.12f, umaxnorm * 0.5f); // ex: 0.25 si strict=0.5
    PitchAnalyzer analyzer_strict(n_len, rate, PitchAnalyzer::RECT, 50, 500, umaxnorm);
    PitchAnalyzer analyzer_loose (n_len, rate, PitchAnalyzer::RECT, 50, 500, umaxnorm_loose);

    vector<float> f0;
    vector<float> rms_list;
    f0.reserve(x.size() / n_shift);
    rms_list.reserve(x.size() / n_shift);

    // ---------- PASS 1: strict + RMS ----------
    for (auto it = x.begin(); it + n_len < x.end(); it += n_shift) {
        vector<float> frame(it, it + n_len);

        // DC removal per frame
        double mean = 0.0;
        for (float s : frame) mean += s;
        mean /= frame.size();
        for (auto &s : frame) s -= (float)mean;

        // preemphasis (OFF per defecte)
        if (ENABLE_PREEMPH) {
            for (int n = (int)frame.size() - 1; n >= 1; --n)
                frame[n] = frame[n] - PREEMPH_A * frame[n - 1];
        }

        // center clipping CORRECTE (OFF per defecte)
        if (ENABLE_CLIP) {
            float maxAbs = 0.0f;
            for (float s : frame) maxAbs = std::max(maxAbs, std::fabs(s));
            float thr = CLIP_FRAC * maxAbs;
            if (thr > 0.0f) {
                for (auto &s : frame) {
                    float a = std::fabs(s);
                    if (a < thr) s = 0.0f;
                    else s = (s >= 0.0f ? 1.0f : -1.0f) * (a - thr);
                }
            }
        }

        // RMS
        double e = 0.0;
        for (float s : frame) e += (double)s * (double)s;
        float rms = (frame.empty()) ? 0.0f : (float)std::sqrt(e / frame.size());
        rms_list.push_back(rms);

        // f0 strict
        float f = analyzer_strict(frame.begin(), frame.end());
        f0.push_back(f);
    }

    // ---------- Decideix llindars d’energia (robust) ----------
    float rms_med = median_of(rms_list);
    if (rms_med < 1e-6f) rms_med = 1e-6f;

    // força a 0 els frames molt silenciosos
    float rms_silence_thr  = 0.30f * rms_med;

    // si és prou energètic i el strict ha donat 0, prova el loose
    float rms_fallback_thr = 2.00f * rms_med;

    // ---------- PASS 2: fallback loose només on té sentit ----------
    size_t idx = 0;
    for (auto it = x.begin(); it + n_len < x.end(); it += n_shift, ++idx) {
        if (idx >= f0.size()) break;

        if (rms_list[idx] < rms_silence_thr) {
            f0[idx] = 0.0f;
            continue;
        }

        if (f0[idx] == 0.0f && rms_list[idx] >= rms_fallback_thr) {
            vector<float> frame(it, it + n_len);

            // DC removal per frame (mateix que abans)
            double mean = 0.0;
            for (float s : frame) mean += s;
            mean /= frame.size();
            for (auto &s : frame) s -= (float)mean;

            // (aquí expressament NO fem clip ni preemph per no perjudicar)
            float f = analyzer_loose(frame.begin(), frame.end());
            f0[idx] = f; // si segueix 0, ok
        }
    }

    // ---------- Post-processat lleuger: mediana (ignora zeros) ----------
    if (!f0.empty()) {
        vector<float> f0_med(f0.size());
        const int W = 1;
        vector<float> win;
        win.reserve(2 * W + 1);

        for (size_t i = 0; i < f0.size(); ++i) {
            win.clear();
            for (int k = -W; k <= W; ++k) {
                int j = (int)i + k;
                if (j >= 0 && j < (int)f0.size() && f0[j] > 0.0f) win.push_back(f0[j]);
            }
            if (win.empty()) f0_med[i] = 0.0f;
            else {
                sort(win.begin(), win.end());
                f0_med[i] = win[win.size() / 2];
            }
        }
        f0.swap(f0_med);
    }

    // Write output
    ofstream os(output_txt);
    if (!os.good()) {
        cerr << "Error reading output file " << output_txt << " (" << strerror(errno) << ")\n";
        return -3;
    }

    os << 0 << '\n';
    for (float v : f0) os << v << '\n';
    os << 0 << '\n';

    return 0;
}
