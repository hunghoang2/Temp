// camera_thread.h
#pragma once
#include <QThread>
#include <QMutex>
#include <opencv2/opencv.hpp>

class CameraThread : public QThread
{
    Q_OBJECT
public:
    explicit CameraThread(QObject *parent = nullptr);
    ~CameraThread();

    bool openCamera(int index = 0);
    void stop();
    cv::Mat getLatestFrame();

signals:
    void newFrameAvailable();

protected:
    void run() override;

private:
    cv::VideoCapture cap;
    cv::Mat latestFrame;
    QMutex mutex;
    bool running = false;
};
