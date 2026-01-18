// TargetTracker.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <optional>
#include <cmath>
#include <algorithm>

class TargetTracker {
public:
    enum class State { IDLE, TRACKING, LOST };

    struct Config {
        double fps = 25.0;
        double dtOverride = 0.0; // nếu >0 thì dùng dtOverride thay vì 1/fps

        // ROI dynamic
        int roiHalfMin = 120;
        int roiHalfMax = 240;
        double roiK = 2.5;
        int roiMargin = 80;

        // LOST gating
        double errThreshPx = 120.0;
        int errBadNeeded = 3;

        // Template check / reacquire
        int templateSize = 64;          // nếu init bbox 50x50 có thể dùng 48
        double nccLostThresh = 0.60;
        int nccBadNeeded = 3;

        double reacquireRoiScale = 2.0;
        double reacquireNccThresh = 0.65;

        // Kalman noise
        double qPos = 2e-4;
        double qVel = 5e-3;
        double rMeas = 2e-3;
    };

    struct Output {
        State state = State::IDLE;

        // Draw overlay
        cv::Rect2d bboxFull{0,0,0,0};           // bbox hiện tại (full frame)
        cv::Point2f measCenterPx{NAN, NAN};     // tâm bbox hiện tại (để vẽ crosshair)
        cv::Rect roiUsed{0,0,0,0};              // ROI tracking/reacquire (để debug)

        // Pan/Tilt control (pixel predicted, full frame)
        cv::Point2f predCenterPx{NAN, NAN};     // predicted 1-step ahead (đổi sang góc)

        // Diagnostics
        double residualErrPx = NAN;
        double ncc = NAN;
        bool measurementAccepted = false;
        bool reacquiredThisFrame = false;
    };

public:
    explicit TargetTracker(Config cfg = {}) : cfg_(cfg) {
        dt_ = (cfg_.dtOverride > 0.0) ? cfg_.dtOverride : (1.0 / std::max(1.0, cfg_.fps));
    }

    void reset() {
        state_ = State::IDLE;
        tracker_.release();
        templateImg_.release();
        kalmanInitialized_ = false;
        badErrCount_ = 0;
        badNccCount_ = 0;
        bbox_ = {};
    }

    bool init(const cv::Mat& frameBgr, const cv::Rect2d& initBboxFull) {
        if (frameBgr.empty()) return false;
        reset();

        bbox_ = clampRect(initBboxFull, frameBgr.size());
        if (bbox_.width <= 2 || bbox_.height <= 2) return false;

        tracker_ = cv::TrackerCSRT::create();
        if (!tracker_->init(frameBgr, bbox_)) return false;

        // template around center
        const cv::Point2f c = rectCenter(bbox_);
        templateImg_ = extractTemplateGray(frameBgr, c, cfg_.templateSize);
        if (templateImg_.empty()) return false;

        initKalman(c);

        state_ = State::TRACKING;
        return true;
    }

    Output update(const cv::Mat& frameBgr) {
        Output out;
        out.state = state_;
        if (frameBgr.empty()) return out;
        if (state_ == State::IDLE) return out;

        // ---------- 1) Kalman predict ONCE per frame ----------
        const cv::Point2f predCenter = kalmanPredictOnce(); // 1-step ahead
        out.predCenterPx = predCenter;

        // Build ROI around predicted center (for debug + reacquire window)
        const int roiHalf = computeDynamicRoiHalf();
        const cv::Rect roi = makeCenteredRoi(predCenter, roiHalf, frameBgr.size());
        out.roiUsed = roi;

        // ---------- 2) CSRT update (measurement) ----------
        bool ok = false;
        cv::Rect2d newBox = bbox_;

        if (tracker_) {
            ok = tracker_->update(frameBgr, newBox);
        }
        if (ok) {
            newBox = clampRect(newBox, frameBgr.size());
            if (newBox.width <= 2 || newBox.height <= 2) ok = false;
        }

        if (!ok) {
            // CSRT failed (hiếm) -> LOST
            state_ = State::LOST;
        } else {
            bbox_ = newBox;
        }

        // Current bbox for drawing
        out.bboxFull = bbox_;
        const cv::Point2f measCenter = rectCenter(bbox_);
        out.measCenterPx = measCenter;

        // ---------- 3) Validation ----------
        const double errPx = l2(measCenter - predCenter);
        out.residualErrPx = errPx;

        double ncc = NAN;
        if (!templateImg_.empty()) {
            cv::Mat patch = extractTemplateGray(frameBgr, measCenter, cfg_.templateSize);
            if (!patch.empty() && patch.size() == templateImg_.size()) {
                ncc = computeNcc(templateImg_, patch);
            }
        }
        out.ncc = ncc;

        const bool goodErr = (errPx <= cfg_.errThreshPx);
        const bool goodNcc = (std::isnan(ncc) ? true : (ncc >= cfg_.nccLostThresh));

        if (state_ == State::TRACKING) {
            if (!goodErr) badErrCount_++; else badErrCount_ = 0;
            if (!goodNcc) badNccCount_++; else badNccCount_ = 0;

            if (badErrCount_ >= cfg_.errBadNeeded || badNccCount_ >= cfg_.nccBadNeeded) {
                state_ = State::LOST;
            } else {
                // Accept measurement -> kalman correct
                kalmanCorrect(measCenter);
                out.measurementAccepted = true;
            }
        }

        // ---------- 4) LOST -> reacquire ----------
        if (state_ == State::LOST) {
            const bool reacq = tryReacquire(frameBgr, predCenter, roiHalf, out);
            if (reacq) {
                out.reacquiredThisFrame = true;
            }
        }

        out.state = state_;
        return out;
    }

    State state() const { return state_; }

private:
    // ----------------- Kalman CV core -----------------
    void initKalman(const cv::Point2f& centerPx) {
        kf_ = cv::KalmanFilter(4, 2, 0, CV_32F);

        kf_.transitionMatrix = (cv::Mat_<float>(4,4) <<
            1, 0, (float)dt_, 0,
            0, 1, 0, (float)dt_,
            0, 0, 1, 0,
            0, 0, 0, 1
        );

        kf_.measurementMatrix = (cv::Mat_<float>(2,4) <<
            1, 0, 0, 0,
            0, 1, 0, 0
        );

        kf_.processNoiseCov = cv::Mat::zeros(4,4, CV_32F);
        kf_.processNoiseCov.at<float>(0,0) = (float)cfg_.qPos;
        kf_.processNoiseCov.at<float>(1,1) = (float)cfg_.qPos;
        kf_.processNoiseCov.at<float>(2,2) = (float)cfg_.qVel;
        kf_.processNoiseCov.at<float>(3,3) = (float)cfg_.qVel;

        kf_.measurementNoiseCov = cv::Mat::eye(2,2, CV_32F) * (float)cfg_.rMeas;
        kf_.errorCovPost = cv::Mat::eye(4,4, CV_32F) * 1.0f;

        kf_.statePost = (cv::Mat_<float>(4,1) <<
            centerPx.x, centerPx.y, 0.0f, 0.0f
        );

        kalmanInitialized_ = true;
    }

    // Predict exactly once per frame (1-step ahead)
    cv::Point2f kalmanPredictOnce() {
        if (!kalmanInitialized_) {
            // fallback: center of last bbox
            return rectCenter(bbox_);
        }
        const cv::Mat pred = kf_.predict();
        return { pred.at<float>(0), pred.at<float>(1) };
    }

    void kalmanCorrect(const cv::Point2f& measCenterPx) {
        if (!kalmanInitialized_) {
            initKalman(measCenterPx);
            return;
        }
        cv::Mat z(2,1,CV_32F);
        z.at<float>(0) = measCenterPx.x;
        z.at<float>(1) = measCenterPx.y;
        kf_.correct(z);
    }

    int computeDynamicRoiHalf() const {
        if (!kalmanInitialized_) return cfg_.roiHalfMin;

        // statePost is after last correction (or init)
        const float vx = kf_.statePost.at<float>(2); // px/s
        const float vy = kf_.statePost.at<float>(3);
        const float vpx = std::max(std::abs(vx), std::abs(vy));
        const float vPerFrame = vpx * (float)dt_;

        int half = (int)std::round(cfg_.roiK * vPerFrame + cfg_.roiMargin);
        half = std::clamp(half, cfg_.roiHalfMin, cfg_.roiHalfMax);
        return half;
    }

    // ----------------- reacquire -----------------
    bool tryReacquire(const cv::Mat& frameBgr,
                      const cv::Point2f& predCenter,
                      int trackingRoiHalf,
                      Output& out) {
        if (templateImg_.empty()) return false;

        const int half = (int)std::round(trackingRoiHalf * cfg_.reacquireRoiScale);
        const cv::Rect searchRoi = makeCenteredRoi(predCenter, half, frameBgr.size());
        if (searchRoi.width < cfg_.templateSize + 2 || searchRoi.height < cfg_.templateSize + 2)
            return false;

        cv::Mat gray;
        cv::cvtColor(frameBgr, gray, cv::COLOR_BGR2GRAY);
        cv::Mat roiGray = gray(searchRoi);

        cv::Mat result;
        cv::matchTemplate(roiGray, templateImg_, result, cv::TM_CCOEFF_NORMED);

        double minVal=0, maxVal=0;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        if (maxVal < cfg_.reacquireNccThresh) return false;

        cv::Point2f newCenter;
        newCenter.x = (float)(searchRoi.x + maxLoc.x + templateImg_.cols * 0.5);
        newCenter.y = (float)(searchRoi.y + maxLoc.y + templateImg_.rows * 0.5);

        // rebuild bbox around center (keep previous size)
        cv::Rect2d newBox;
        newBox.width  = bbox_.width;
        newBox.height = bbox_.height;
        newBox.x = newCenter.x - newBox.width * 0.5;
        newBox.y = newCenter.y - newBox.height * 0.5;
        newBox = clampRect(newBox, frameBgr.size());

        tracker_.release();
        tracker_ = cv::TrackerCSRT::create();
        if (!tracker_->init(frameBgr, newBox)) return false;

        bbox_ = newBox;
        kalmanCorrect(newCenter);

        badErrCount_ = 0;
        badNccCount_ = 0;
        state_ = State::TRACKING;

        // update output so bạn vẽ đúng ngay frame này
        out.bboxFull = bbox_;
        out.measCenterPx = newCenter;
        out.ncc = maxVal;

        return true;
    }

    // ----------------- helpers -----------------
    static cv::Point2f rectCenter(const cv::Rect2d& r) {
        return {(float)(r.x + r.width * 0.5), (float)(r.y + r.height * 0.5)};
    }

    static cv::Rect2d clampRect(const cv::Rect2d& r, const cv::Size& sz) {
        double x = std::clamp(r.x, 0.0, (double)sz.width  - 1.0);
        double y = std::clamp(r.y, 0.0, (double)sz.height - 1.0);
        double w = std::clamp(r.width,  1.0, (double)sz.width  - x);
        double h = std::clamp(r.height, 1.0, (double)sz.height - y);
        return {x,y,w,h};
    }

    static cv::Rect makeCenteredRoi(const cv::Point2f& c, int half, const cv::Size& sz) {
        int x = (int)std::round(c.x) - half;
        int y = (int)std::round(c.y) - half;
        int w = 2 * half;
        int h = 2 * half;

        x = std::clamp(x, 0, std::max(0, sz.width  - 1));
        y = std::clamp(y, 0, std::max(0, sz.height - 1));
        w = std::clamp(w, 1, sz.width  - x);
        h = std::clamp(h, 1, sz.height - y);
        return {x,y,w,h};
    }

    cv::Mat extractTemplateGray(const cv::Mat& frameBgr, const cv::Point2f& center, int size) const {
        if (frameBgr.empty() || size <= 8) return {};
        cv::Rect roi = makeCenteredRoi(center, size/2, frameBgr.size());
        if (roi.width != size || roi.height != size) return {};
        cv::Mat gray;
        cv::cvtColor(frameBgr, gray, cv::COLOR_BGR2GRAY);
        return gray(roi).clone();
    }

    static double computeNcc(const cv::Mat& a, const cv::Mat& b) {
        CV_Assert(a.size() == b.size() && a.type() == CV_8U && b.type() == CV_8U);
        cv::Mat af, bf;
        a.convertTo(af, CV_32F);
        b.convertTo(bf, CV_32F);
        af -= cv::mean(af)[0];
        bf -= cv::mean(bf)[0];
        const double num = af.dot(bf);
        const double den = std::sqrt(af.dot(af) * bf.dot(bf)) + 1e-9;
        return num / den;
    }

    static double l2(const cv::Point2f& p) {
        return std::sqrt((double)p.x*p.x + (double)p.y*p.y);
    }

private:
    Config cfg_;
    double dt_ = 0.04;

    State state_ = State::IDLE;

    cv::Ptr<cv::Tracker> tracker_;
    cv::Rect2d bbox_{0,0,0,0};

    // template
    cv::Mat templateImg_;

    // kalman
    cv::KalmanFilter kf_;
    bool kalmanInitialized_ = false;

    // counters
    int badErrCount_ = 0;
    int badNccCount_ = 0;
};
