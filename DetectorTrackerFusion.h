class DetectorTrackerFusion {
public:
    void init(const cv::Mat& frame, const cv::Rect& initBox) {
        tracker = cv::TrackerCSRT::create();
        tracker->init(frame, initBox);
        validator.init(frame, initBox);
        lastBox = initBox;
        tracking = true;
    }

    void update(const cv::Mat& frame,
                float dt,
                bool yoloValid,
                const cv::Rect& yoloBox)
    {
        cv::Rect trackBox;
        bool ok = tracking && tracker->update(frame, trackBox);

        float conf;
        cv::Point2f kpred;
        auto state = validator.update(frame, trackBox, dt, conf, kpred);

        // ================= CASE 1: TRACK GOOD =================
        if(state == TrackValidatorLv2::TRACK_VALID) {
            lastBox = trackBox;
            return;
        }

        // ================= CASE 2: YOLO AVAILABLE =================
        if(yoloValid) {
            float iouVal = iou(trackBox, yoloBox);
            float centerDist = cv::norm(center(trackBox) - center(yoloBox));

            // ---- Tracker hơi nghi ngờ nhưng vẫn gần YOLO → correction ----
            if(state == TrackValidatorLv2::TRACK_SUSPECT &&
               (iouVal > 0.3f || centerDist < 40.0f))
            {
                reinit(frame, yoloBox);  // snap về detection
                return;
            }

            // ---- Tracker LOST → YOLO cứu ----
            if(state == TrackValidatorLv2::TRACK_LOST) {
                reinit(frame, yoloBox);
                return;
            }
        }

        // ================= CASE 3: NO YOLO =================
        if(state == TrackValidatorLv2::TRACK_LOST) {
            tracking = false;
        }
    }

    bool isTracking() const { return tracking; }
    cv::Rect getBox() const { return lastBox; }

private:
    cv::Ptr<cv::Tracker> tracker;
    TrackValidatorLv2 validator;
    cv::Rect lastBox;
    bool tracking = false;

    void reinit(const cv::Mat& frame, const cv::Rect& box) {
        tracker = cv::TrackerCSRT::create();
        tracker->init(frame, box);
        validator.init(frame, box);
        lastBox = box;
        tracking = true;
    }

    float iou(const cv::Rect& a, const cv::Rect& b) {
        float inter = (a & b).area();
        float uni = a.area() + b.area() - inter;
        return inter / uni;
    }

    inline cv::Point2f center(const cv::Rect& r) {
        return {r.x+r.width*0.5f, r.y+r.height*0.5f};
    }
};
