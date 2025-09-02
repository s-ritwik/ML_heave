// tester_mp4.cpp — LibTorch + OpenCV LSTM tester (fixed tuple/list usage)

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <string>
#include <vector>
#include <filesystem>
#include <thread>   // for sleep_for

using namespace std;

// ------------------------------ INPUTS ------------------------------------
static const int   test_time_sec  = 150; // seconds to simulate
static const char* model_path     = "seq_lstm/noisyLSTM_models_seq/noisy_D1_LSTM_40_6_1024_512/epoch_400_script.pt";
static const char* test_csv_path  = "seq/train_data_normalised/D1H3_normalised.csv";

static const float noise_std      = 0.05f; // Gaussian noise on test-time input
static const int   sampling_rate  = 20;    // Hz (video fps & pacing)

// ----------------------------- HELPERS ------------------------------------
static const int NAME_RATE_HZ = 20; // how seq/out seconds are encoded in the model name
static const float meters_to_cm = 25.0f;

// Read first column of CSV into vector<float>
static vector<float> read_csv_first_col(const string& path) {
    vector<float> data;
    ifstream in(path);
    if (!in) {
        throw runtime_error("Failed to open CSV: " + path);
    }
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        size_t pos = line.find(',');
        string tok = (pos == string::npos) ? line : line.substr(0, pos);
        try {
            data.push_back(static_cast<float>(stod(tok)));
        } catch (...) {
            // skip header/invalid
        }
    }
    return data;
}

static tuple<int,int,vector<int64_t>> parse_model_path(const string& path) {
    // Extract _(GRU|LSTM)_(seq)_(out)_(hs...) where seq/out in seconds @NAME_RATE_HZ
    regex rgx("_(GRU|LSTM)_([0-9]+)_([0-9]+)_([0-9_]+)");
    smatch m;
    if (!regex_search(path, m, rgx) || m.size() < 5) {
        throw runtime_error("Model path does not match expected pattern '_(GRU|LSTM)_<seq>_<out>_<hs...>': " + path);
    }
    int seq_steps = stoi(m[2]) * NAME_RATE_HZ;
    int out_steps = stoi(m[3]) * NAME_RATE_HZ;
    vector<int64_t> hidden_sizes;
    string hs = m[4];
    size_t start = 0;
    while (true) {
        size_t p = hs.find('_', start);
        string num = (p == string::npos) ? hs.substr(start) : hs.substr(start, p - start);
        hidden_sizes.push_back(stoll(num));
        if (p == string::npos) break;
        start = p + 1;
    }
    return {seq_steps, out_steps, hidden_sizes};
}

static string build_output_path(const string& model_path) {
    string out_dir = "seq/noisyprediction_videos";
    std::filesystem::create_directories(out_dir);
    string parent = std::filesystem::path(model_path).parent_path().filename().string();
    string base   = std::filesystem::path(model_path).stem().string(); // epoch_XXX_script
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s_%s_%d_Hz.mp4", parent.c_str(), base.c_str(), sampling_rate);
    return out_dir + "/" + string(buf);
}

// Map (t,y) to pixel coordinates within a panel rectangle
struct PlotRect {
    int x, y, w, h;
    float tmin, tmax; // seconds
    float ymin, ymax; // value units (cm)
};

static cv::Point2f to_px(float t, float y, const PlotRect& R) {
    float nx = (t - R.tmin) / (R.tmax - R.tmin);
    float ny = (y - R.ymin) / (R.ymax - R.ymin);
    float px = R.x + nx * R.w;
    float py = R.y + (1.0f - ny) * R.h;
    return cv::Point2f(px, py);
}

static void draw_axes(cv::Mat& img, const PlotRect& R, const string& xlabel, const string& ylabel, const string& title) {
    cv::rectangle(img, cv::Rect(R.x, R.y, R.w, R.h), cv::Scalar(220,220,220), 1);
    cv::putText(img, title, cv::Point(R.x, R.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(10,10,10), 1);
    cv::putText(img, ylabel, cv::Point(R.x - 55, R.y + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(10,10,10), 1);
    cv::putText(img, xlabel, cv::Point(R.x + R.w - 60, R.y + R.h + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(10,10,10), 1);

    // zero-time line
    if (R.tmin < 0.0f && R.tmax > 0.0f) {
        auto p0 = to_px(0.0f, R.ymin, R);
        auto p1 = to_px(0.0f, R.ymax, R);
        cv::line(img, p0, p1, cv::Scalar(180,180,180), 1, cv::LINE_AA);
    }
}

static void draw_polyline(cv::Mat& img, const vector<float>& t, const vector<float>& y, const PlotRect& R, const cv::Scalar& color, int thickness=2, int style=0) {
    if (t.size() < 2 || y.size() != t.size()) return;
    for (size_t i = 1; i < t.size(); ++i) {
        auto p0 = to_px(t[i-1], y[i-1], R);
        auto p1 = to_px(t[i],   y[i],   R);
        if (style == 1) { // dashed
            const int segs = 16;
            for (int s = 0; s < segs; ++s) {
                float a0 = float(s)/segs, a1 = float(s+0.5)/segs;
                cv::Point2f q0 = p0 + (p1 - p0) * a0;
                cv::Point2f q1 = p0 + (p1 - p0) * a1;
                cv::line(img, q0, q1, color, thickness, cv::LINE_AA);
            }
        } else {
            cv::line(img, p0, p1, color, thickness, cv::LINE_AA);
        }
    }
}

int main() {
    try {
        // ---------------------------- SETUP ---------------------------------------
        auto [sequence_length, output_size, hidden_sizes] = parse_model_path(model_path);

        // auto device = (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        // torch::Device dev(device);
        torch::Device dev(torch::kCUDA);

        // cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << endl;

        // Load TorchScript module
        torch::jit::script::Module module = torch::jit::load(model_path, dev);
        module.eval();

        // Data
        vector<float> test_data = read_csv_first_col(test_csv_path);

        // ---------------------------- RANGES --------------------------------------
        int total_steps = test_time_sec * sampling_rate;
        int start_index = std::max(sequence_length, output_size);
        int end_index   = start_index + total_steps;
        if (end_index + output_size > (int)test_data.size()) {
            end_index   = (int)test_data.size() - output_size;
            total_steps = end_index - start_index;
        }
        cout << "Testing for " << (total_steps / (double)sampling_rate) << " seconds.\n";
        cout << "start_index: " << start_index << " | end_index: " << end_index
             << " | len(test_data): " << test_data.size() << "\n";

        // Errors & timing
        vector<double> prediction_times;
        vector<double> absolute_errors;
        vector<double> errors_3s, errors_4s, errors_5s;
        int steps_3s = 3 * sampling_rate, steps_4s = 4 * sampling_rate, steps_5s = 5 * sampling_rate;

        // Hidden state: GenericList of Tuple(h, c)
        c10::impl::GenericList state_list(c10::TensorType::get());
        for (auto hs : hidden_sizes) {
            auto h0 = torch::zeros({1, 1, hs}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            auto c0 = torch::zeros({1, 1, hs}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            state_list.push_back(c10::IValue(h0));
            state_list.push_back(c10::IValue(c0));
        }

        // Video
        const int W = 1280, H = 720;
        string out_mp4 = build_output_path(model_path);
        cv::VideoWriter vw(out_mp4, cv::VideoWriter::fourcc('a','v','c','1'), sampling_rate, cv::Size(W,H));
        if (!vw.isOpened()) throw runtime_error("Failed to open VideoWriter: " + out_mp4);

        auto testing_start = chrono::high_resolution_clock::now();

        // History buffer (what we actually fed, *noisy*, in cm)
        vector<float> noisy_hist_cm; noisy_hist_cm.reserve(sequence_length);

        // Plot panel rects
        int margin = 60;
        PlotRect top  {margin+40, margin,           W - margin*2 - 40, (H - margin*3)/2, -float(sequence_length)/sampling_rate, float(output_size)/sampling_rate, -30.0f, 30.0f};
        PlotRect bot  {margin+40, margin*2 + top.h, W - margin*2 - 40, (H - margin*3)/2, 0.0f, float(output_size)/sampling_rate, 0.0f, 18.0f};

        // RNG for Gaussian noise
        std::mt19937 rng(42);
        std::normal_distribution<float> norm(0.0f, noise_std);

        int error_start_index = start_index + sequence_length;

        for (int i = start_index; i < end_index; ++i) {
            auto iter_start = chrono::high_resolution_clock::now();

            // Noisy input (streaming, one tick)
            float clean_val = test_data[i];
            float noisy_val = clean_val + norm(rng);
            noisy_hist_cm.push_back(noisy_val * meters_to_cm);
            if ((int)noisy_hist_cm.size() > sequence_length) noisy_hist_cm.erase(begin(noisy_hist_cm));

            // Build input tensor [1,1,1]
            auto x = torch::empty({1,1,1}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
            x[0][0][0] = noisy_val;

            // Forward with state
            auto t0 = chrono::high_resolution_clock::now();
            auto out_iv = module.forward({c10::IValue(x), c10::IValue(state_list)}).toTuple();
            auto t1 = chrono::high_resolution_clock::now();
            double dt_ms = chrono::duration<double>(t1 - t0).count() * 1000.0;
            prediction_times.push_back(dt_ms / 1000.0);

            // y: [1, output_size]
            torch::Tensor y = out_iv->elements()[0].toTensor();

            // next_state: List[Tensor] = [h0, c0, h1, c1, ...]
            c10::List<c10::IValue> next_state_any = out_iv->elements()[1].toList();

            // detach and keep same flat structure
            c10::impl::GenericList new_state(c10::TensorType::get());
            for (int si = 0; si < (int)next_state_any.size(); ++si) {
                auto t = next_state_any.get(si).toTensor().detach();
                new_state.push_back(c10::IValue(t));
            }
            state_list = new_state;

            // Prepare truth & error
            vector<float> true_future_cm(output_size);
            for (int k = 0; k < output_size; ++k) true_future_cm[k] = test_data[i + 1 + k] * meters_to_cm;

            auto y_cpu = y.to(torch::kCPU).contiguous();
            const float* yp = y_cpu.data_ptr<float>();
            vector<float> pred_future_cm(output_size);
            for (int k = 0; k < output_size; ++k) pred_future_cm[k] = yp[k] * meters_to_cm;

            vector<float> abs_err(output_size);
            for (int k = 0; k < output_size; ++k) abs_err[k] = fabs(true_future_cm[k] - pred_future_cm[k]);

            if (i >= error_start_index) {
                double mean_all = 0.0;
                for (float v : abs_err) mean_all += v;
                mean_all /= abs_err.size();
                absolute_errors.push_back(mean_all);

                auto meanN = [&](int N)->double{
                    N = std::min(N, (int)abs_err.size());
                    if (N<=0) return 0.0;
                    double s=0.0; for (int j=0;j<N;++j) s+=abs_err[j]; return s/N;
                };
                errors_3s.push_back(meanN(steps_3s));
                errors_4s.push_back(meanN(steps_4s));
                errors_5s.push_back(meanN(steps_5s));
            }

            // ------------------------- RENDER FRAME -------------------------
            cv::Mat frame(H, W, CV_8UC3, cv::Scalar(255,255,255));

            // Titles
            auto elapsed = chrono::duration<double>(chrono::high_resolution_clock::now() - testing_start).count();
            char titlebuf[512];
            snprintf(titlebuf, sizeof(titlebuf),
                     "Elapsed: %.2fs / %ds   |   Model: %s",
                     elapsed, test_time_sec, std::filesystem::path(model_path).filename().string().c_str());
            cv::putText(frame, titlebuf, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(10,10,10), 2);

            // Top panel axes
            draw_axes(frame, top, "Time (s)", "Position (cm)", "Noisy history + future (true vs predicted)");

            // Build history timeline (variable length up to sequence_length)
            vector<float> t_hist, y_hist;
            int n_hist = (int)noisy_hist_cm.size();
            t_hist.reserve(n_hist);
            y_hist.reserve(n_hist);
            for (int k = 0; k < n_hist; ++k) {
                float t = -float(n_hist - k) / sampling_rate; // from -Tin..-1/sr
                t_hist.push_back(t);
                y_hist.push_back(noisy_hist_cm[k]);
            }
            draw_polyline(frame, t_hist, y_hist, top, cv::Scalar(50,50,50), 2, 0); // noisy history

            // Future time base
            vector<float> t_future(output_size);
            for (int k = 0; k < output_size; ++k) t_future[k] = float(k+1) / sampling_rate;

            // True future (dashed green) & predicted (blue)
            draw_polyline(frame, t_future, true_future_cm, top, cv::Scalar(60,180,60), 2, 1);
            draw_polyline(frame, t_future, pred_future_cm,  top, cv::Scalar(40,40,200), 2, 0);

            // Per-tick timing box
            double cur_ms = dt_ms;
            double avg_ms = 0.0;
            if (!prediction_times.empty()) {
                double s=0.0; for (double v: prediction_times) s+=v*1000.0; avg_ms = s / prediction_times.size();
            }
            char infobuf[256];
            snprintf(infobuf, sizeof(infobuf),
                     "Pred time: %.2f ms  (avg %.2f ms)\nNoise sigma=%.3f",
                     cur_ms, avg_ms, noise_std);
            int bx = top.x + top.w - 320, by = top.y + top.h - 60;
            cv::rectangle(frame, cv::Rect(bx, by-40, 310, 50), cv::Scalar(255,255,255), -1);
            cv::rectangle(frame, cv::Rect(bx, by-40, 310, 50), cv::Scalar(180,180,180), 1);
            cv::putText(frame, infobuf, cv::Point(bx+10, by-20), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(10,10,10), 1);

            // Bottom panel: absolute error
            draw_axes(frame, bot, "Time (s)", "Error (cm)", "Absolute error");
            draw_polyline(frame, t_future, abs_err, bot, cv::Scalar(180,80,80), 2, 0);

            vw << frame;

            // pacing
            auto iter_end = chrono::high_resolution_clock::now();
            double elapsed_s = chrono::duration<double>(iter_end - iter_start).count();
            double target_s  = 1.0 / sampling_rate;
            if (elapsed_s < target_s) {
                std::this_thread::sleep_for(std::chrono::duration<double>(target_s - elapsed_s));
            }
        }

        vw.release();

        // --------------------------- METRICS & PRINTS -----------------------------
        auto mean = [](const vector<double>& v)->double{
            if (v.empty()) return 0.0;
            double s=0.0; for (double x: v) s+=x; return s/v.size();
        };

        cout << "\nSaved video: " << out_mp4 << "\n";
        cout << "Average Prediction Time: " << mean(prediction_times) << " s\n";
        cout << "Average Absolute Error (first 3s): " << mean(errors_3s) << " cm\n";
        cout << "Average Absolute Error (first 4s): " << mean(errors_4s) << " cm\n";
        cout << "Average Absolute Error (first 5s): " << mean(errors_5s) << " cm\n";
        cout << "Total Average Absolute Error: " << mean(absolute_errors) << " cm\n";

    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
// ---------------------------------------------------------------------------