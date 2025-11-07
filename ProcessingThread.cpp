// processing_thread.cpp
#include "processing_thread.h"

ProcessingThread::ProcessingThread(QObject *parent)
    : QThread(parent)
{
}

ProcessingThread::~ProcessingThread()
{
    stop();
}

void ProcessingThread::stop()
{
    QMutexLocker locker(&mutex);
    running = false;
}

void ProcessingThread::setFrame(const cv::Mat &frame)
{
    QMutexLocker locker(&mutex);
    currentFrame = frame.clone();
    newFrameAvailable = true;
}

void ProcessingThread::setROI(const QRect &roi)
{
    QMutexLocker locker(&mutex);
    if (!currentFrame.empty())
    {
        tracker = cv::TrackerCSRT::create();
        cv::Rect2d cvroi(roi.x(), roi.y(), roi.width(), roi.height());
        tracker->init(currentFrame, cvroi);
        tracking = true;
    }
}

void ProcessingThread::run()
{
    {
        QMutexLocker locker(&mutex);
        running = true;
    }

    while (true)
    {
        {
            QMutexLocker locker(&mutex);
            if (!running) break;
        }

        cv::Mat frame;
        {
            QMutexLocker locker(&mutex);
            if (!newFrameAvailable) {
                msleep(5);
                continue;
            }
            frame = currentFrame.clone();
            newFrameAvailable = false;
        }

        if (tracking && tracker)
        {
            cv::Rect2d box;
            bool ok = tracker->update(frame, box);
            if (ok)
                cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        QImage img((uchar*)frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
        emit frameProcessed(img.copy());
    }
}
