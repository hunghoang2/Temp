#include "ZmqClient.h"

ZmqClient::ZmqClient(QObject* parent) : QThread(parent) {}
ZmqClient::~ZmqClient() { stop(); wait(); }

void ZmqClient::setConfig(const Config& cfg) { m_cfg = cfg; }

void ZmqClient::stop()
{
    m_running.store(false, std::memory_order_relaxed);
    m_txCv.wakeAll(); // đánh thức nếu IO thread đang wait
}

bool ZmqClient::sendMessage(const QByteArray& payload)
{
    if (payload.isEmpty()) return false;

    {
        QMutexLocker lk(&m_txMutex);
        m_txQueue.push_back(payload);
    }
    // Wake IO thread để gửi ngay (latency thấp)
    m_txCv.wakeOne();
    return true;
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
    // Context + sockets tạo trong IO thread
    m_ctx  = std::make_unique<zmq::context_t>(1);
    m_sub  = std::make_unique<zmq::socket_t>(*m_ctx, zmq::socket_type::sub);
    m_push = std::make_unique<zmq::socket_t>(*m_ctx, zmq::socket_type::push);

    m_sub->set(zmq::sockopt::rcvhwm, m_cfg.rcvHwm);
    m_push->set(zmq::sockopt::sndhwm, m_cfg.sndHwm);

    m_sub->set(zmq::sockopt::subscribe, m_cfg.subTopic);

    m_sub->connect(m_cfg.subEndpoint);
    m_push->connect(m_cfg.pushEndpoint);

    zmq::pollitem_t items[] = {
        { static_cast<void*>(*m_sub), 0, ZMQ_POLLIN, 0 }
    };

    while (m_running.load(std::memory_order_relaxed)) {

        // 1) Poll SUB để nhận liên tục
        zmq::poll(items, 1, m_cfg.pollTimeoutMs);

        if (items[0].revents & ZMQ_POLLIN) {
            // Nhận hết những gì đang có để không backlog (tuỳ server publish rate)
            while (true) {
                zmq::message_t part1;
                auto r1 = m_sub->recv(part1, zmq::recv_flags::dontwait);
                if (!r1) break;

                bool more = m_sub->get(zmq::sockopt::rcvmore);

                QByteArray topic, payload;
                if (more) {
                    zmq::message_t part2;
                    auto r2 = m_sub->recv(part2, zmq::recv_flags::none);
                    if (!r2) break;
                    topic   = QByteArray((const char*)part1.data(), (int)part1.size());
                    payload = QByteArray((const char*)part2.data(), (int)part2.size());
                } else {
                    // single-part: topic empty
                    topic.clear();
                    payload = QByteArray((const char*)part1.data(), (int)part1.size());
                }
                emit rxMessage(topic, payload);
            }
        }

        // 2) Drain TX queue: gửi ASAP tất cả message đã được enqueue từ thread khác
        //    Để giảm latency, nếu queue trống thì wait ngắn (không block SUB quá lâu)
        QList<QByteArray> batch;
        {
            QMutexLocker lk(&m_txMutex);
            if (m_txQueue.isEmpty()) {
                // wait rất ngắn để giảm CPU, nhưng vẫn responsive cho SUB vì poll đang chạy
                m_txCv.wait(&m_txMutex, 1); // 1ms
            }
            if (!m_txQueue.isEmpty()) {
                batch.swap(m_txQueue); // lấy hết một lần, giảm lock contention
            }
        }

        for (const auto& b : batch) {
            zmq::message_t msg((size_t)b.size());
            memcpy(msg.data(), b.constData(), (size_t)b.size());

            // dontwait để không block; nếu HWM đầy sẽ drop.
            // Nếu bạn muốn retry, mình sẽ đưa policy riêng.
            (void)m_push->send(msg, zmq::send_flags::dontwait);
        }
    }
}

void ZmqClient::closeSockets()
{
    try {
        if (m_sub)  { m_sub->close();  m_sub.reset(); }
        if (m_push) { m_push->close(); m_push.reset(); }
        m_ctx.reset();
    } catch (...) {}
}
