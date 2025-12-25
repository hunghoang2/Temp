#pragma once
#include <QThread>
#include <QByteArray>
#include <QMutex>
#include <QWaitCondition>
#include <QList>
#include <atomic>
#include <memory>
#include <string>
#include <zmq.hpp>

class ZmqClient : public QThread
{
    Q_OBJECT
public:
    struct Config {
        std::string subEndpoint;
        std::string pushEndpoint;
        std::string subTopic = "";
        int pollTimeoutMs = 2;     // poll nhỏ để latency send thấp
        int rcvHwm = 1000;
        int sndHwm = 1000;
    };

    explicit ZmqClient(QObject* parent=nullptr);
    ~ZmqClient() override;

    void setConfig(const Config& cfg);
    void stop();

    // Gọi từ bất kỳ thread nào để gửi lên server (tối đa 100/s ok)
    bool sendMessage(const QByteArray& payload);

signals:
    void rxMessage(const QByteArray& topic, const QByteArray& payload);
    void errorOccurred(const QString& err);

protected:
    void run() override;

private:
    void ioLoop();
    void closeSockets();

private:
    Config m_cfg;

    std::unique_ptr<zmq::context_t> m_ctx;
    std::unique_ptr<zmq::socket_t>  m_sub;
    std::unique_ptr<zmq::socket_t>  m_push;

    std::atomic<bool> m_running{false};

    // TX queue (thread-safe)
    QMutex m_txMutex;
    QWaitCondition m_txCv;        // wake IO thread nhanh hơn thay vì đợi poll timeout
    QList<QByteArray> m_txQueue;
};
