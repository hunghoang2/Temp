// camera_thread.cpp
#include "camera_thread.h"

CameraThread::CameraThread(QObject *parent) : QThread(parent) {}
CameraThread::~CameraThread() { stop(); }

bool CameraThread::openCamera(int index)
{
    if (cap.isOpened()) cap.release();
    bool ok = cap.open(index);
    return ok;
}

void CameraThread::stop()
{
    QMutexLocker locker(&mutex);
    running = false;
}

cv::Mat CameraThread::getLatestFrame()
{
    QMutexLocker locker(&mutex);
    return latestFrame.clone();
}

void CameraThread::run()
{
    {
        QMutexLocker locker(&mutex);
        running = true;
    }

    cv::Mat frame;
    while (true)
    {
        {
            QMutexLocker locker(&mutex);
            if (!running) break;
        }

        cap >> frame;
        if (frame.empty()) continue;

        {
            QMutexLocker locker(&mutex);
            latestFrame = frame.clone();
        }
        emit newFrameAvailable();
        msleep(10); // khoảng 100 fps
    }
}
