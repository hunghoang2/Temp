#include "drone_tracker.h"
#include <algorithm>
#include <cmath>

namespace drone {

static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

DroneTracker::DroneTracker(const DroneTrackerConfig& cfg) : cfg_(cfg) {
    reset();
}

void DroneTracker::reset() {
    kf_inited_ = false;
    hits_ = 0;
    misses_ = 0;
    locked_ = false;
    ref_area_ = 50.f;
    ref_contrast_ = 30.f;
    last_size_ = {20.f, 20.f};
}

void DroneTracker::kalmanInit(float x, float y) {
    // 4-state, 2-measure
    kf_ = cv::KalmanFilter(4, 2, 0, CV_32F);

    // state: x y vx vy
    kf_.transitionMatrix = (cv::Mat_<float>(4,4) <<
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1
    );

    kf_.measurementMatrix = (cv::Mat_<float>(2,4) <<
        1,0,0,0,
        0,1,0,0
    );

    cv::setIdentity(kf_.processNoiseCov);
    kf_.processNoiseCov.at<float>(0,0) = cfg_.q_pos;
    kf_.processNoiseCov.at<float>(1,1) = cfg_.q_pos;
    kf_.processNoiseCov.at<float>(2,2) = cfg_.q_vel;
    kf_.processNoiseCov.at<float>(3,3) = cfg_.q_vel;

    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar(cfg_.r_meas));
    cv::setIdentity(kf_.errorCovPost, cv::Scalar(1));

    kf_.statePost = (cv::Mat_<float>(4,1) << x, y, 0, 0);
    kf_inited_ = true;
}

void DroneTracker::kalmanPredict() {
    kf_.predict();
}

void DroneTracker::kalmanCorrect(float x, float y) {
    cv::Mat meas = (cv::Mat_<float>(2,1) << x, y);
    kf_.correct(meas);
}

void DroneTracker::init(const cv::Mat& frame, const cv::Rect2f& bbox) {
    // use bbox center as initial measurement
    float cx = bbox.x + bbox.width * 0.5f;
    float cy = bbox.y + bbox.height * 0.5f;

    kalmanInit(cx, cy);

    // set last size from bbox
    last_size_ = {bbox.width, bbox.height};

    // initialize reference stats from this ROI (contrast)
    cv::Mat gray = preprocessTo8UGray(frame);
    cv::Rect roi = cv::Rect((int)bbox.x, (int)bbox.y, (int)bbox.width, (int)bbox.height) &
                   cv::Rect(0,0,gray.cols, gray.rows);

    if (roi.area() > 0) {
        cv::Mat patch = gray(roi);
        double minV, maxV;
        cv::minMaxLoc(patch, &minV, &maxV);
        // estimate background from a small ring around roi
        int pad = 10;
        cv::Rect outer = (roi + cv::Size(pad*2, pad*2)) - cv::Point(pad, pad);
        outer &= cv::Rect(0,0,gray.cols, gray.rows);

        double bg = 0.0;
        if (outer.area() > roi.area()) {
            cv::Mat outPatch = gray(outer);
            bg = cv::mean(outPatch)[0];
        } else {
            bg = cv::mean(patch)[0];
        }

        ref_area_ = (float)roi.area();
        ref_contrast_ = (float)(maxV - bg);
        ref_contrast_ = std::max(ref_contrast_, 5.0f);
    }

    hits_ = cfg_.min_hits_to_lock;
    misses_ = 0;
    locked_ = true;
}

cv::Mat DroneTracker::preprocessTo8UGray(const cv::Mat& frame) const {
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 1) {
        gray = frame;
    } else {
        // fallback: convert to gray
        cv::Mat tmp;
        frame.convertTo(tmp, CV_8U);
        gray = tmp;
    }

    if (gray.type() != CV_8U) {
        cv::Mat g8;
        double minV, maxV;
        cv::minMaxLoc(gray, &minV, &maxV);
        if (maxV > minV) {
            gray.convertTo(g8, CV_8U, 255.0/(maxV-minV), -minV*255.0/(maxV-minV));
        } else {
            gray.convertTo(g8, CV_8U);
        }
        gray = g8;
    }

    if (cfg_.median_ksize >= 3) {
        cv::medianBlur(gray, gray, cfg_.median_ksize);
    }

    if (cfg_.use_clahe) {
        auto clahe = cv::createCLAHE(cfg_.clahe_clip, cfg_.clahe_tile);
        cv::Mat eq;
        clahe->apply(gray, eq);
        gray = eq;
    }

    if (cfg_.use_tophat) {
        int k = std::max(3, cfg_.tophat_kernel | 1);
        cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k,k));
        cv::Mat tophat;
        cv::morphologyEx(gray, tophat, cv::MORPH_TOPHAT, se);
        gray = tophat;
    }

    return gray;
}

std::vector<DroneTracker::Candidate> DroneTracker::detectCandidates(const cv::Mat& gray8) const {
    // Dynamic threshold: mean + k*std
    cv::Scalar m, s;
    cv::meanStdDev(gray8, m, s);
    double thr = m[0] + cfg_.thr_k_sigma * s[0];
    thr = clampf((float)thr, 5.f, 250.f);

    cv::Mat bin;
    cv::threshold(gray8, bin, thr, 255, cv::THRESH_BINARY);

    // Morphology to remove speckles and connect tiny blobs
    cv::Mat se3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, se3);
    cv::dilate(bin, bin, se3);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Candidate> out;
    out.reserve(contours.size());

    for (auto& c : contours) {
        int area = (int)cv::contourArea(c);
        if (area < cfg_.min_area || area > cfg_.max_area) continue;

        cv::Rect bb = cv::boundingRect(c);
        float aspect = (bb.height > 0) ? (float)bb.width / (float)bb.height : 999.f;
        aspect = std::max(aspect, 1.f/aspect);
        if (aspect > cfg_.max_aspect) continue;

        // peak intensity in bbox
        cv::Mat patch = gray8(bb);
        double minV, maxV;
        cv::Point minP, maxP;
        cv::minMaxLoc(patch, &minV, &maxV, &minP, &maxP);

        // background mean in an outer ring around bbox (cheap approx)
        int pad = 10;
        cv::Rect outer = (bb + cv::Size(pad*2, pad*2)) - cv::Point(pad, pad);
        outer &= cv::Rect(0,0,gray8.cols, gray8.rows);

        double bg = cv::mean(gray8(outer))[0];
        float contrast = (float)(maxV - bg);

        // centroid by moments (on contour)
        cv::Moments mu = cv::moments(c);
        cv::Point2f cent;
        if (mu.m00 != 0.0) {
            cent = {(float)(mu.m10/mu.m00), (float)(mu.m01/mu.m00)};
        } else {
            cent = {(float)(bb.x + bb.width*0.5f), (float)(bb.y + bb.height*0.5f)};
        }

        Candidate cand;
        cand.c = cent;
        cand.bbox = bb;
        cand.area = area;
        cand.peak = (float)maxV;
        cand.bg = (float)bg;
        cand.contrast = contrast;
        cand.aspect = aspect;

        out.push_back(cand);
    }

    return out;
}

std::optional<int> DroneTracker::associateCandidate(
    const std::vector<Candidate>& cands,
    const cv::Point2f& pred,
    float gate_radius,
    float ref_area,
    float ref_contrast
) const {
    if (cands.empty()) return std::nullopt;

    float bestCost = 1e30f;
    int bestIdx = -1;

    // normalize references
    ref_area = std::max(ref_area, 1.0f);
    ref_contrast = std::max(ref_contrast, 1.0f);

    for (int i = 0; i < (int)cands.size(); ++i) {
        const auto& ca = cands[i];
        float dx = ca.c.x - pred.x;
        float dy = ca.c.y - pred.y;
        float d = std::sqrt(dx*dx + dy*dy);

        if (d > gate_radius) continue;

        // cost terms
        float cd = (d / std::max(1.0f, gate_radius));
        float c_area = std::fabs((float)ca.area - ref_area) / ref_area;

        // contrast similarity (use peak-bg)
        float c_con = std::fabs(ca.contrast - ref_contrast) / ref_contrast;

        float cost =
            cfg_.w_dist * (cd * cd) +
            cfg_.w_area * c_area +
            cfg_.w_contrast * c_con;

        // prefer higher contrast slightly (tie-break)
        cost -= 0.05f * (ca.contrast / (ref_contrast + 1e-3f));

        if (cost < bestCost) {
            bestCost = cost;
            bestIdx = i;
        }
    }

    if (bestIdx < 0) return std::nullopt;
    return bestIdx;
}

TrackResult DroneTracker::update(const cv::Mat& frame) {
    TrackResult res;
    if (frame.empty()) return res;

    cv::Mat gray = preprocessTo8UGray(frame);

    // Predict (or init by fullscan)
    cv::Point2f pred;
    cv::Point2f vel{0,0};

    if (kf_inited_) {
        kalmanPredict();
        pred = {(float)kf_.statePre.at<float>(0), (float)kf_.statePre.at<float>(1)};
        vel  = {(float)kf_.statePre.at<float>(2), (float)kf_.statePre.at<float>(3)};
    } else {
        pred = {(float)gray.cols*0.5f, (float)gray.rows*0.5f};
        vel = {0,0};
    }

    float speed = std::sqrt(vel.x*vel.x + vel.y*vel.y);
    float gate = cfg_.gate_base + cfg_.gate_vel_gain * speed;
    gate = clampf(gate, cfg_.gate_base, cfg_.gate_max);

    // Detect candidates
    auto cands = detectCandidates(gray);

    // Optionally fullscan on misses (not necessary usually)
    bool doGate = true;
    if (cfg_.reacquire_fullscan_every > 0 && misses_ > 0) {
        if ((misses_ % cfg_.reacquire_fullscan_every) == 0) doGate = false;
    }

    cv::Point2f assocPred = pred;
    float assocGate = doGate ? gate : 1e9f;

    // Associate
    auto bestIdxOpt = associateCandidate(cands, assocPred, assocGate, ref_area_, ref_contrast_);

    if (bestIdxOpt.has_value()) {
        const auto& best = cands[*bestIdxOpt];

        // init or correct kalman
        if (!kf_inited_) {
            kalmanInit(best.c.x, best.c.y);
        } else {
            kalmanCorrect(best.c.x, best.c.y);
        }

        // update stats slowly to adapt but avoid drift
        float lr = clampf(cfg_.stats_lr, 0.0f, 0.5f);
        ref_area_ = (1.0f - lr) * ref_area_ + lr * (float)best.area;
        ref_contrast_ = (1.0f - lr) * ref_contrast_ + lr * std::max(best.contrast, 1.0f);

        // update size (smooth)
        cv::Size2f sz((float)best.bbox.width, (float)best.bbox.height);
        last_size_.width  = 0.8f * last_size_.width  + 0.2f * sz.width;
        last_size_.height = 0.8f * last_size_.height + 0.2f * sz.height;

        // update track management
        hits_++;
        misses_ = 0;
        locked_ = (hits_ >= cfg_.min_hits_to_lock);

        // output bbox around measurement bbox (or around center with smoothed size)
        cv::Point2f c = best.c;
        res.center = c;
        res.hits = hits_;
        res.misses = misses_;
        res.locked = locked_;

        float w = std::max(6.0f, last_size_.width);
        float h = std::max(6.0f, last_size_.height);

        cv::Rect2f bb(c.x - w*0.5f, c.y - h*0.5f, w, h);
        bb &= cv::Rect2f(0,0,(float)gray.cols,(float)gray.rows);

        res.bbox = bb;
        return res;
    }

    // No candidate matched -> miss
    misses_++;
    hits_ = std::max(0, hits_ - 1);
    locked_ = locked_ && (misses_ <= cfg_.max_misses);

    // If we have Kalman, we can still output predicted bbox for short gaps
    if (kf_inited_ && misses_ <= cfg_.max_misses) {
        cv::Point2f c = pred;
        res.center = c;
        res.hits = hits_;
        res.misses = misses_;
        res.locked = locked_;

        float w = std::max(6.0f, last_size_.width);
        float h = std::max(6.0f, last_size_.height);
        cv::Rect2f bb(c.x - w*0.5f, c.y - h*0.5f, w, h);
        bb &= cv::Rect2f(0,0,(float)gray.cols,(float)gray.rows);
        res.bbox = bb;
        return res;
    }

    // Lost
    if (misses_ > cfg_.max_misses) {
        kf_inited_ = false;
        locked_ = false;
        // keep refs; or reset them if you prefer:
        // ref_area_ = 50.f; ref_contrast_ = 30.f;
    }

    res.center = pred;
    res.hits = hits_;
    res.misses = misses_;
    res.locked = locked_;
    res.bbox = std::nullopt;
    return res;
}

} // namespace drone
