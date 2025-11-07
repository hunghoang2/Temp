// processing_thread.h
#pragma once
#include <QThread>
#include <QMutex>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

class ProcessingThread : public QThread
{
    Q_OBJECT
public:
    explicit ProcessingThread(QObject *parent = nullptr);
    ~ProcessingThread();

    void setFrame(const cv::Mat &frame);
    void setROI(const QRect &roi);
    void stop();

signals:
    void frameProcessed(const QImage &img);

protected:
    void run() override;

private:
    QMutex mutex;
    cv::Mat currentFrame;
    bool newFrameAvailable = false;
    bool running = false;

    bool tracking = false;
    cv::Ptr<cv::Tracker> tracker;
};
