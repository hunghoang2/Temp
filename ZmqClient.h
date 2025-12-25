#pragma once

#include <QObject>
#include <QByteArray>
#include <QMutex>
#include <QThread>
#include <atomic>
#include <memory>
#include <string>

// cppzmq
#include <zmq.hpp>

class ZmqClient : public QThread
{
    Q_OBJECT
public:
    struct Config {
        std::string subEndpoint;   // vd: "tcp://127.0.0.1:5556"
        std::string pushEndpoint;  // vd: "tcp://127.0.0.1:5557"
        std::string subTopic = ""; // "" => subscribe all
        int recvHz = 30;           // xử lý nhận tối đa 30/s
        int sendHz = 100;          // gửi 100/s
        int pollTimeoutMs = 1;     // poll nhỏ để có độ mịn tốt
        int rcvHwm = 10;           // tránh backlog lớn
        int sndHwm = 10;
        bool conflateSub = true;   // true: chỉ giữ message mới nhất (nếu libzmq hỗ trợ)
    };

    explicit ZmqClient(QObject* parent = nullptr);
    ~ZmqClient() override;

    void setConfig(const Config& cfg);

    // Cập nhật payload “mới nhất” để IO thread gửi (thread-safe).
    void setLatestTxPayload(const QByteArray& payload);

    // Optional: gửi ngay 1 message (enqueue) - nếu bạn cần burst.
    void enqueueTx(const QByteArray& payload);

    void stop();

signals:
    void connected();
    void disconnected();
    void rxMessage(const QByteArray& topic, const QByteArray& payload);
    void txTick(qint64 monotonicMs); // báo mỗi tick gửi để app cập nhật data nếu muốn
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

    // Latest payload mode (send fixed rate)
    QMutex m_latestMutex;
    QByteArray m_latestPayload;

    // Queue mode (optional)
    QMutex m_queueMutex;
    QList<QByteArray> m_queue;
};
