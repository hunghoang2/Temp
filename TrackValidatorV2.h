#pragma once
#include <opencv2/opencv.hpp>

class TrackValidatorLv2 {
public:
    enum TrackState { TRACK_VALID, TRACK_SUSPECT, TRACK_LOST };

    void init(const cv::Mat& frame, const cv::Rect& initBox) {
        baseArea   = initBox.area();
        baseAspect = aspect(initBox);

        prevCenter = center(initBox);
        prevVelocity = {0,0};

        initKalman(prevCenter);

        cv::resize(frame(initBox), templateImg, {32,32});
        templateImg.convertTo(templateImg, CV_32F);

        updateContrastRef(frame, initBox);
    }

    TrackState update(const cv::Mat& frame,
                      const cv::Rect& box,
                      float dt,
                      float& confidence,
                      cv::Point2f& kalmanPredOut)
    {
        // ===== Kalman Predict =====
        setDT(dt);
        cv::Mat pred = kf.predict();
        kalmanPredOut = {pred.at<float>(0), pred.at<float>(1)};

        int fail = 0;
        float score = 1.0f;

        cv::Point2f c = center(box);

        // ===== 1. Shape =====
        float areaRatio = float(box.area()) / baseArea;
        float arRatio   = aspect(box) / baseAspect;
        if(areaRatio > 2.0f || areaRatio < 0.4f) { fail++; score -= 0.2f; }
        if(arRatio > 1.7f || arRatio < 0.6f)     { fail++; score -= 0.2f; }

        // ===== 2. Thermal Contrast (adaptive) =====
        float contrast = computeContrast(frame, box);
        if(contrast < contrastRef * 0.5f) { fail++; score -= 0.25f; }

        // ===== 3. Kalman Mahalanobis Gate =====
        cv::Mat meas = (cv::Mat_<float>(2,1) << c.x, c.y);
        cv::Mat diff = meas - H*kf.statePre;
        cv::Mat S = H*kf.errorCovPre*H.t() + R;
        float maha = diff.dot(S.inv()*diff);
        if(maha > 9.21f) { fail++; score -= 0.25f; } // ~3 sigma

        // ===== 4. Velocity physics =====
        cv::Point2f vel = (c - prevCenter)/dt;
        if(cv::norm(prevVelocity) > 1e-3) {
            float cosang = vel.dot(prevVelocity) /
                           (cv::norm(vel)*cv::norm(prevVelocity)+1e-6f);
            if(cosang < 0.2f) { fail++; score -= 0.15f; }
        }

        // ===== 5. Template similarity =====
        float sim = templateSimilarity(frame, box);
        if(sim < 0.35f) { fail++; score -= 0.3f; }

        confidence = std::max(0.f, score);

        // ===== State decision =====
        TrackState state;
        if(fail <= 1) state = TRACK_VALID;
        else if(fail <= 2) state = TRACK_SUSPECT;
        else state = TRACK_LOST;

        // ===== Kalman Update only if not LOST =====
        if(state != TRACK_LOST) {
            kf.correct(meas);
            prevCenter = c;
            prevVelocity = vel;

            // Update reference models only if confident
            if(state == TRACK_VALID && confidence > 0.75f) {
                updateTemplate(frame, box);
                updateContrastRef(frame, box);
            }
        }

        return state;
    }

private:
    // ================= Kalman =================
    cv::KalmanFilter kf = cv::KalmanFilter(4,2,0);
    cv::Mat H = (cv::Mat_<float>(2,4) << 1,0,0,0, 0,1,0,0);
    cv::Mat R = cv::Mat::eye(2,2,CV_32F) * 4;

    void initKalman(cv::Point2f p) {
        kf.transitionMatrix = cv::Mat::eye(4,4,CV_32F);
        kf.measurementMatrix = H;
        setDT(0.02f);
        setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
        setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1));
        setIdentity(kf.errorCovPost, cv::Scalar::all(1));

        kf.statePost = (cv::Mat_<float>(4,1) << p.x, p.y, 0, 0);
    }

    void setDT(float dt){
        kf.transitionMatrix.at<float>(0,2) = dt;
        kf.transitionMatrix.at<float>(1,3) = dt;
    }

    // ================= Tracking Memory =================
    cv::Mat templateImg;
    float contrastRef = 10.f;
    float baseArea, baseAspect;
    cv::Point2f prevCenter, prevVelocity;

    // ================= Helper =================
    inline float aspect(const cv::Rect& r){ return float(r.w)/r.h; }
    inline cv::Point2f center(const cv::Rect& r){
        return {r.x+r.w*0.5f, r.y+r.h*0.5f};
    }

    float computeContrast(const cv::Mat& f, const cv::Rect& b){
        cv::Rect in = b & cv::Rect(0,0,f.cols,f.rows);
        cv::Rect out(b.x-b.w/2, b.y-b.h/2, b.w*2, b.h*2);
        out &= cv::Rect(0,0,f.cols,f.rows);
        return cv::mean(f(in))[0] - cv::mean(f(out))[0];
    }

    void updateContrastRef(const cv::Mat& f, const cv::Rect& b){
        contrastRef = 0.9f*contrastRef + 0.1f*computeContrast(f,b);
    }

    float templateSimilarity(const cv::Mat& f, const cv::Rect& b){
        cv::Mat roi, small, res;
        cv::resize(f(b), small, {32,32});
        small.convertTo(small, CV_32F);
        cv::matchTemplate(small, templateImg, res, cv::TM_CCOEFF_NORMED);
        return res.at<float>(0,0);
    }

    void updateTemplate(const cv::Mat& f, const cv::Rect& b){
        cv::Mat small;
        cv::resize(f(b), small, {32,32});
        small.convertTo(small, CV_32F);
        templateImg = 0.9f*templateImg + 0.1f*small;
    }
};





#pragma once
#include <opencv2/opencv.hpp>

class TrackValidatorLv2 {
public:
    enum TrackState { TRACK_VALID, TRACK_SUSPECT, TRACK_LOST };

    void init(const cv::Mat& frame, const cv::Rect& initBox)
    {
        cv::Rect roi;
        if(!safeRect(initBox, frame.size(), roi)) return;

        baseArea   = roi.area();
        baseAspect = aspect(roi);

        prevCenter = center(roi);
        prevVelocity = {0,0};

        initKalman(prevCenter);

        cv::Mat tmp = frame(roi);
        cv::resize(tmp, templateImg, {32,32});
        templateImg.convertTo(templateImg, CV_32F);

        contrastRef = computeContrast(frame, roi);
        inited = true;
    }

    TrackState update(const cv::Mat& frame,
                      const cv::Rect& box,
                      float dt,
                      float& confidence,
                      cv::Point2f& kalmanPredOut)
    {
        confidence = 0.0f;
        if(!inited) return TRACK_LOST;

        dt = std::max(dt, 1e-3f);

        // ---------- Kalman predict ----------
        setDT(dt);
        cv::Mat pred = kf.predict();
        kalmanPredOut = {pred.at<float>(0), pred.at<float>(1)};

        cv::Rect roi;
        if(!safeRect(box, frame.size(), roi))
            return TRACK_LOST;

        cv::Point2f c = center(roi);

        int fail = 0;
        float score = 1.0f;

        // ===== Shape =====
        float areaRatio = float(roi.area()) / std::max(baseArea,1.f);
        float arRatio   = aspect(roi) / std::max(baseAspect,0.1f);
        if(areaRatio > 2.5f || areaRatio < 0.3f){ fail++; score-=0.2f; }
        if(arRatio > 2.0f || arRatio < 0.5f){ fail++; score-=0.2f; }

        // ===== Thermal contrast =====
        float contrast = computeContrast(frame, roi);
        if(contrast < contrastRef*0.5f){ fail++; score-=0.25f; }

        // ===== Kalman gate =====
        cv::Mat meas = (cv::Mat_<float>(2,1) << c.x, c.y);
        cv::Mat diff = meas - H*kf.statePre;
        cv::Mat S = H*kf.errorCovPre*H.t() + R;
        cv::Mat Sinv;
        cv::invert(S, Sinv, cv::DECOMP_SVD);
        float maha = diff.dot(Sinv*diff);
        if(maha > 9.21f){ fail++; score-=0.25f; }

        // ===== Velocity physics =====
        cv::Point2f vel = (c - prevCenter)/dt;
        if(cv::norm(prevVelocity) > 1e-3f){
            float cosang = vel.dot(prevVelocity) /
                (cv::norm(vel)*cv::norm(prevVelocity)+1e-6f);
            if(cosang < 0.2f){ fail++; score-=0.15f; }
        }

        // ===== Template similarity =====
        float sim = templateSimilarity(frame, roi);
        if(sim < 0.3f){ fail++; score-=0.25f; }

        confidence = std::max(0.f, score);

        TrackState state;
        if(fail <= 1) state = TRACK_VALID;
        else if(fail == 2) state = TRACK_SUSPECT;
        else state = TRACK_LOST;

        if(state != TRACK_LOST){
            kf.correct(meas);
            prevCenter = c;
            prevVelocity = vel;

            if(state == TRACK_VALID && confidence > 0.75f){
                updateTemplate(frame, roi);
                contrastRef = 0.9f*contrastRef + 0.1f*contrast;
            }
        }

        return state;
    }

private:
    bool inited=false;
    float baseArea=1, baseAspect=1;
    float contrastRef=10;
    cv::Mat templateImg;
    cv::Point2f prevCenter, prevVelocity;

    // ========== Kalman ==========
    cv::KalmanFilter kf = cv::KalmanFilter(4,2,0);
    cv::Mat H = (cv::Mat_<float>(2,4) << 1,0,0,0, 0,1,0,0);
    cv::Mat R = cv::Mat::eye(2,2,CV_32F)*4;

    void initKalman(cv::Point2f p){
        kf.transitionMatrix = cv::Mat::eye(4,4,CV_32F);
        kf.measurementMatrix = H;
        setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
        setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1));
        setIdentity(kf.errorCovPost, cv::Scalar::all(1));
        kf.statePost = (cv::Mat_<float>(4,1)<<p.x,p.y,0,0);
    }

    void setDT(float dt){
        kf.transitionMatrix.at<float>(0,2)=dt;
        kf.transitionMatrix.at<float>(1,3)=dt;
    }

    // ========== SAFE HELPERS ==========
    inline bool safeRect(const cv::Rect& r,const cv::Size& sz,cv::Rect& out){
        out = r & cv::Rect(0,0,sz.width,sz.height);
        return out.area() > 16;
    }

    inline float aspect(const cv::Rect& r){
        return float(r.width)/std::max(1,r.height);
    }

    inline cv::Point2f center(const cv::Rect& r){
        return {r.x+r.width*0.5f, r.y+r.height*0.5f};
    }

    float computeContrast(const cv::Mat& f,const cv::Rect& b){
        cv::Rect outer(b.x-b.width/2,b.y-b.height/2,b.width*2,b.height*2);
        cv::Rect o; if(!safeRect(outer,f.size(),o)) return 0;
        return cv::mean(f(b))[0] - cv::mean(f(o))[0];
    }

    float templateSimilarity(const cv::Mat& f,const cv::Rect& b){
        if(templateImg.empty()) return 0;
        cv::Mat small,res;
        cv::resize(f(b),small,{32,32});
        small.convertTo(small,CV_32F);
        cv::matchTemplate(small,templateImg,res,cv::TM_CCOEFF_NORMED);
        return res.at<float>(0,0);
    }

    void updateTemplate(const cv::Mat& f,const cv::Rect& b){
        if(templateImg.empty()) return;
        cv::Mat small;
        cv::resize(f(b),small,{32,32});
        small.convertTo(small,CV_32F);
        if(small.size()==templateImg.size())
            templateImg = 0.9f*templateImg + 0.1f*small;
    }
};
