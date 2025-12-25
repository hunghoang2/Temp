#pragma once

#include <QObject>
#include <QThread>
#include <QTimer>
#include <QString>
#include <QByteArray>
#include <QAtomicInt>

#include <zmq.hpp>

class ZmqClientWorker : public QObject
{
    Q_OBJECT
public:
    explicit ZmqClientWorker(QString endpoint,
                             QString routingId = {},
                             QObject* parent = nullptr)
        : QObject(parent),
          endpoint_(std::move(endpoint)),
          routingId_(std::move(routingId)),
          ctx_(1),
          dealer_(ctx_, zmq::socket_type::dealer)
    {}

public slots:
    void start()
    {
        if (running_.loadAcquire()) return;
        running_.storeRelease(true);

        // Socket options
        dealer_.set(zmq::sockopt::linger, 0);
        dealer_.set(zmq::sockopt::rcvtimeo, 50);
        dealer_.set(zmq::sockopt::sndtimeo, 50);

        if (!routingId_.isEmpty()) {
            dealer_.set(zmq::sockopt::routing_id, routingId_.toStdString());
        }

        dealer_.connect(endpoint_.toStdString());

        // Polling via Qt timer (keeps everything in this thread)
        pollTimer_ = new QTimer(this);
        pollTimer_->setInterval(10);
        connect(pollTimer_, &QTimer::timeout, this, &ZmqClientWorker::pollOnce);
        pollTimer_->start();

        emit connected(endpoint_);
    }

    void stop()
    {
        if (!running_.loadAcquire()) return;
        running_.storeRelease(false);

        if (pollTimer_) {
            pollTimer_->stop();
            pollTimer_->deleteLater();
            pollTimer_ = nullptr;
        }

        // dealer_ will be closed when worker destroyed; linger=0 prevents hang
        emit disconnected();
    }

    // Thread-safe from caller perspective because it's queued to this worker thread.
    void sendMessage(const QByteArray& data)
    {
        if (!running_.loadAcquire()) return;
        if (data.isEmpty()) return;

        // IMPORTANT: This runs in the worker thread, same as recv.
        dealer_.send(zmq::buffer(data.constData(), data.size()), zmq::send_flags::none);
    }

private slots:
    void pollOnce()
    {
        if (!running_.loadAcquire()) return;

        // Drain all pending messages (non-block due to rcvtimeo)
        while (running_.loadAcquire()) {
            zmq::message_t msg;
            auto ok = dealer_.recv(msg, zmq::recv_flags::dontwait);
            if (!ok) break;

            QByteArray data(static_cast<const char*>(msg.data()),
                            static_cast<int>(msg.size()));
            emit messageReceived(data);
        }
    }

signals:
    void connected(const QString& endpoint);
    void disconnected();
    void messageReceived(const QByteArray& data);
    void errorOccurred(const QString& err);

private:
    QString endpoint_;
    QString routingId_;

    zmq::context_t ctx_;
    zmq::socket_t dealer_;

    QTimer* pollTimer_ = nullptr;
    QAtomicInt running_{false};
};


class ZmqClient : public QObject
{
    Q_OBJECT
public:
    explicit ZmqClient(QString endpoint,
                       QString routingId = {},
                       QObject* parent = nullptr)
        : QObject(parent),
          endpoint_(std::move(endpoint)),
          routingId_(std::move(routingId))
    {
        worker_ = new ZmqClientWorker(endpoint_, routingId_);
        worker_->moveToThread(&thread_);

        // Start/stop worker in its own thread
        connect(&thread_, &QThread::started, worker_, &ZmqClientWorker::start);
        connect(&thread_, &QThread::finished, worker_, &QObject::deleteLater);

        // Public signals passthrough
        connect(worker_, &ZmqClientWorker::connected, this, &ZmqClient::connected);
        connect(worker_, &ZmqClientWorker::disconnected, this, &ZmqClient::disconnected);
        connect(worker_, &ZmqClientWorker::messageReceived, this, &ZmqClient::messageReceived);
        connect(worker_, &ZmqClientWorker::errorOccurred, this, &ZmqClient::errorOccurred);

        // Route send requests to worker thread safely
        connect(this, &ZmqClient::sendRequested,
                worker_, &ZmqClientWorker::sendMessage,
                Qt::QueuedConnection);

        // Route stop to worker thread
        connect(this, &ZmqClient::stopRequested,
                worker_, &ZmqClientWorker::stop,
                Qt::QueuedConnection);
    }

    ~ZmqClient() override { stop(); }

    void start()
    {
        if (thread_.isRunning()) return;
        thread_.start();
    }

    void stop()
    {
        if (!thread_.isRunning()) return;
        emit stopRequested();
        thread_.quit();
        thread_.wait();
    }

    // Can be called from ANY thread.
    Q_INVOKABLE void sendMessage(const QByteArray& data)
    {
        emit sendRequested(data);
    }

signals:
    void connected(const QString& endpoint);
    void disconnected();
    void messageReceived(const QByteArray& data);
    void errorOccurred(const QString& err);

private signals:
    void sendRequested(const QByteArray& data);
    void stopRequested();

private:
    QString endpoint_;
    QString routingId_;

    QThread thread_;
    ZmqClientWorker* worker_ = nullptr;
};





#include <zmq.hpp>
#include <iostream>
#include <thread>

int main() {
    zmq::context_t ctx(1);
    zmq::socket_t dealer(ctx, zmq::socket_type::dealer);

    // optional set identity (otherwise ZMQ generates one)
    dealer.set(zmq::sockopt::routing_id, "clientA");

    dealer.connect("tcp://127.0.0.1:5555");

    dealer.send(zmq::buffer("ping"), zmq::send_flags::none);

    zmq::message_t reply;
    dealer.recv(reply);
    std::cout << "Reply: " << reply.to_string() << "\n";
}
