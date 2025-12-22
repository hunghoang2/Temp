#pragma once
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

namespace drone {

struct DroneTrackerConfig {
    // --- Preprocess ---
    bool use_clahe = true;
    double clahe_clip = 2.0;
    cv::Size clahe_tile = {8, 8};

    bool use_tophat = true;                 // rất hợp bright small target
    int tophat_kernel = 9;                  // odd: 7/9/11
    int median_ksize = 3;                   // 3 or 5

    // --- Threshold / blob ---
    double thr_k_sigma = 3.0;               // thr = mean + k*std; tune 2.5..4.0
    int min_area = 3;                       // pixels
    int max_area = 600;                     // pixels (tune)
    float max_aspect = 6.0f;                // reject long streaks / big regions

    // --- Gating / association ---
    float gate_base = 35.0f;                // px, base radius
    float gate_vel_gain = 2.0f;             // extra px per px/frame of speed
    float gate_max = 180.0f;                // hard cap px

    // --- Cost weights ---
    float w_dist = 1.0f;                    // distance term weight
    float w_area = 0.6f;                    // area similarity weight
    float w_contrast = 0.8f;                // peak-bg contrast similarity weight

    // --- Template adaptation (drone stats) ---
    float stats_lr = 0.05f;                 // update rate for area/contrast reference (0..1)

    // --- Kalman ---
    // State: [x, y, vx, vy]
    float q_pos = 2.0f;                     // process noise position
    float q_vel = 10.0f;                    // process noise velocity
    float r_meas = 6.0f;                    // measurement noise

    // --- Track management ---
    int min_hits_to_lock = 2;               // how many consecutive matches to be confident
    int max_misses = 10;                    // after this -> lost
    int reacquire_fullscan_every = 0;       // 0=always gate; >0 fullscan each N misses (optional)
};

struct TrackResult {
    std::optional<cv::Rect2f> bbox;         // bbox in original frame coords
    bool locked = false;                    // confident tracking state
    int hits = 0;
    int misses = 0;
    cv::Point2f center = {0,0};             // estimated center (Kalman)
};

class DroneTracker {
public:
    explicit DroneTracker(const DroneTrackerConfig& cfg = {});
    void reset();

    // Optional manual init (user-selected bbox). Helps first lock.
    void init(const cv::Mat& frame, const cv::Rect2f& bbox);

    // Main update. Returns bbox if found/estimated.
    TrackResult update(const cv::Mat& frame);

private:
    struct Candidate {
        cv::Point2f c;
        cv::Rect bbox;
        int area = 0;
        float peak = 0.f;
        float bg = 0.f;
        float contrast = 0.f;               // peak - bg
        float aspect = 1.f;
    };

    cv::Mat preprocessTo8UGray(const cv::Mat& frame) const;
    std::vector<Candidate> detectCandidates(const cv::Mat& gray8) const;

    std::optional<int> associateCandidate(
        const std::vector<Candidate>& cands,
        const cv::Point2f& pred,
        float gate_radius,
        float ref_area,
        float ref_contrast
    ) const;

    void kalmanInit(float x, float y);
    void kalmanPredict();
    void kalmanCorrect(float x, float y);

    DroneTrackerConfig cfg_;

    // Kalman filter
    cv::KalmanFilter kf_;
    bool kf_inited_ = false;

    // Track stats reference
    float ref_area_ = 50.f;
    float ref_contrast_ = 30.f;

    // Track mgmt
    int hits_ = 0;
    int misses_ = 0;
    bool locked_ = false;

    // Last bbox size (for output)
    cv::Size2f last_size_{20.f, 20.f};
};

} // namespace drone
