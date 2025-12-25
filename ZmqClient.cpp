#include "ZmqClient.h"
#include <QDateTime>

#include <chrono>

using namespace std::chrono;

static inline qint64 nowMonotonicMs()
{
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

ZmqClient::ZmqClient(QObject* parent)
    : QThread(parent)
{
}

ZmqClient::~ZmqClient()
{
    stop();
    wait();
}

void ZmqClient::setConfig(const Config& cfg)
{
    m_cfg = cfg;
}

void ZmqClient::setLatestTxPayload(const QByteArray& payload)
{
    QMutexLocker lk(&m_latestMutex);
    m_latestPayload = payload;
}

void ZmqClient::enqueueTx(const QByteArray& payload)
{
    QMutexLocker lk(&m_queueMutex);
    m_queue.push_back(payload);
}

void ZmqClient::stop()
{
    m_running.store(false, std::memory_order_relaxed);
}

void ZmqClient::run()
{
    m_running.store(true, std::memory_order_relaxed);

    try {
        ioLoop();
    } catch (const zmq::error_t& e) {
        emit errorOccurred(QString("ZMQ error: %1").arg(e.what()));
    } catch (const std::exception& e) {
        emit errorOccurred(QString("Exception: %1").arg(e.what()));
    }

    closeSockets();
}

void ZmqClient::ioLoop()
{
    // 1) Context + sockets MUST be created inside the IO thread
    m_ctx  = std::make_unique<zmq::context_t>(1);
    m_sub  = std::make_unique<zmq::socket_t>(*m_ctx, zmq::socket_type::sub);
    m_push = std::make_unique<zmq::socket_t>(*m_ctx, zmq::socket_type::push);

    // 2) Options: HWM to prevent queue explosion
    m_sub->set(zmq::sockopt::rcvhwm, m_cfg.rcvHwm);
    m_push->set(zmq::sockopt::sndhwm, m_cfg.sndHwm);

    // 3) SUB subscribe
    m_sub->set(zmq::sockopt::subscribe, m_cfg.subTopic);

    // 4) (Optional) CONFLATE: keep only last msg (best for “latest state” telemetry)
    // Note: conflate works for certain patterns; if unsupported, it may throw.
    if (m_cfg.conflateSub) {
        try {
            m_sub->set(zmq::sockopt::conflate, 1);
        } catch (...) {
            // ignore if libzmq doesn’t support conflate here
        }
    }

    // 5) Connect
    m_sub->connect(m_cfg.subEndpoint);
    m_push->connect(m_cfg.pushEndpoint);

    emit connected();

    const auto sendPeriod = milliseconds( (m_cfg.sendHz > 0) ? (1000 / m_cfg.sendHz) : 10 );
    const auto recvMinPeriod = milliseconds( (m_cfg.recvHz > 0) ? (1000 / m_cfg.recvHz) : 33 );

    auto nextSend = steady_clock::now();
    auto nextRecvAllow = steady_clock::now();

    zmq::pollitem_t items[] = {
        { static_cast<void*>(*m_sub), 0, ZMQ_POLLIN, 0 },
        // PUSH typically doesn’t need poll-in; send is non-blocking by using dontwait
    };

    while (m_running.load(std::memory_order_relaxed)) {

        // Poll SUB with a tiny timeout to keep timing stable.
        zmq::poll(items, 1, m_cfg.pollTimeoutMs);

        // --- SUB receive: process at most ~30Hz (drop/skip extras) ---
        if (items[0].revents & ZMQ_POLLIN) {

            // If you want strictly 30Hz processing: only “accept” when time allows.
            // But we still drain to keep queue clean (or conflate will keep only latest).
            // Strategy: read all available quickly; keep the last one; emit at most 30Hz.
            zmq::message_t topic;
            zmq::message_t payload;

            bool gotAny = false;
            zmq::message_t lastTopic, lastPayload;

            while (true) {
                // Try recv topic (may be 1-part or 2-part tùy server)
                zmq::recv_result_t r1 = m_sub->recv(topic, zmq::recv_flags::dontwait);
                if (!r1) break;

                // Try recv payload (if multipart)
                bool more = m_sub->get(zmq::sockopt::rcvmore);
                if (more) {
                    zmq::recv_result_t r2 = m_sub->recv(payload, zmq::recv_flags::none);
                    if (!r2) break;
                    lastTopic = std::move(topic);
                    lastPayload = std::move(payload);
                } else {
                    // single-part: treat as payload, topic empty
                    lastTopic = zmq::message_t{};
                    lastPayload = std::move(topic);
                }
                gotAny = true;
            }

            if (gotAny) {
                auto now = steady_clock::now();
                if (now >= nextRecvAllow) {
                    nextRecvAllow = now + recvMinPeriod;

                    QByteArray qTopic(
                        static_cast<const char*>(lastTopic.data()),
                        static_cast<int>(lastTopic.size())
                    );
                    QByteArray qPayload(
                        static_cast<const char*>(lastPayload.data()),
                        static_cast<int>(lastPayload.size())
                    );

                    emit rxMessage(qTopic, qPayload);
                }
            }
        }

        // --- PUSH send: fixed 100Hz ---
        auto now = steady_clock::now();
        if (now >= nextSend) {
            nextSend = now + sendPeriod;

            // optional tick: app can update payload right before send
            emit txTick(nowMonotonicMs());

            // Priority 1: queued messages (burst)
            QByteArray toSend;
            {
                QMutexLocker lk(&m_queueMutex);
                if (!m_queue.isEmpty()) {
                    toSend = m_queue.takeFirst();
                }
            }

            // Priority 2: latest payload (state)
            if (toSend.isEmpty()) {
                QMutexLocker lk(&m_latestMutex);
                toSend = m_latestPayload;
            }

            if (!toSend.isEmpty()) {
                zmq::message_t msg(toSend.size());
                memcpy(msg.data(), toSend.constData(), static_cast<size_t>(toSend.size()));

                // dontwait: tránh block loop; nếu send fail vì HWM, drop (tuỳ bạn)
                auto ok = m_push->send(msg, zmq::send_flags::dontwait);
                (void)ok;
            }
        }
    }

    emit disconnected();
}

void ZmqClient::closeSockets()
{
    try {
        if (m_sub)  { m_sub->close();  m_sub.reset(); }
        if (m_push) { m_push->close(); m_push.reset(); }
        if (m_ctx)  { m_ctx.reset(); }
    } catch (...) {
        // ignore shutdown exceptions
    }
}
