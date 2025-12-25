// ZmqServer.hpp (single-file example)
// Requires: ZeroMQ + cppzmq (zmq.hpp)
// Build: g++ -std=c++17 main.cpp -lzmq -pthread

#include <zmq.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

class ZmqServer {
public:
    using ReceiveCallback = std::function<void(const std::string& payload)>;

    struct Config {
        // Server receives client data here (client PUSH -> server PULL)
        std::string pull_bind = "tcp://*:5556";

        // Server publishes data to clients here (server PUB -> client SUB)
        std::string pub_bind  = "tcp://*:5555";

        int io_threads = 1;

        // High-water marks (tune if needed)
        int pull_rcv_hwm = 2000;
        int pub_snd_hwm  = 2000;

        // Linger: 0 = drop pending on close (fast shutdown)
        int linger_ms = 0;

        // Outgoing queue cap to avoid memory blow if clients slow/disconnect
        std::size_t max_out_queue = 5000;
        bool drop_oldest_when_full = true;
    };

    explicit ZmqServer(Config cfg, ReceiveCallback onRx = nullptr)
        : cfg_(std::move(cfg)),
          onReceive_(std::move(onRx)),
          ctx_(cfg_.io_threads) {}

    ~ZmqServer() { stop(); }

    ZmqServer(const ZmqServer&) = delete;
    ZmqServer& operator=(const ZmqServer&) = delete;

    void start() {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) return;
        ioThread_ = std::thread(&ZmqServer::ioLoop_, this);
    }

    void stop() {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false)) return;
        outCv_.notify_all();
        if (ioThread_.joinable()) ioThread_.join();
    }

    // Thread-safe: can be called from ANY thread.
    // Publish one message to all SUB clients.
    // Returns false if server is not running or queue is full (when configured to reject).
    bool sendMessage(std::string payload) {
        if (!running_.load()) return false;

        {
            std::lock_guard<std::mutex> lk(outMtx_);

            if (cfg_.max_out_queue > 0 && outQueue_.size() >= cfg_.max_out_queue) {
                if (cfg_.drop_oldest_when_full && !outQueue_.empty()) {
                    outQueue_.pop_front();
                } else {
                    return false; // reject new message
                }
            }

            outQueue_.push_back(std::move(payload));
        }

        outCv_.notify_one();
        return true;
    }

    void setReceiveCallback(ReceiveCallback cb) {
        std::lock_guard<std::mutex> lk(cbMtx_);
        onReceive_ = std::move(cb);
    }

private:
    void ioLoop_() {
        // Sockets live ONLY in this thread (ZMQ sockets are not thread-safe)
        zmq::socket_t pull(ctx_, zmq::socket_type::pull);
        zmq::socket_t pub(ctx_,  zmq::socket_type::pub);

        // Configure sockets
        pull.set(zmq::sockopt::rcvhwm, cfg_.pull_rcv_hwm);
        pub.set(zmq::sockopt::sndhwm, cfg_.pub_snd_hwm);

        pull.set(zmq::sockopt::linger, cfg_.linger_ms);
        pub.set(zmq::sockopt::linger, cfg_.linger_ms);

        try {
            pull.bind(cfg_.pull_bind);
            pub.bind(cfg_.pub_bind);
        } catch (const zmq::error_t&) {
            running_.store(false);
            return;
        }

        // Poll incoming from clients. Outgoing is drained each loop.
        constexpr auto kPollSlice = std::chrono::milliseconds(5);

        while (running_.load()) {
            zmq::pollitem_t items[] = {
                { static_cast<void*>(pull), 0, ZMQ_POLLIN, 0 }
            };

            try {
                zmq::poll(items, 1, kPollSlice);
            } catch (const zmq::error_t&) {
                if (!running_.load()) break;
            }

            // 1) Drain ALL available incoming messages (client can push 100Hz+)
            if (items[0].revents & ZMQ_POLLIN) {
                while (running_.load()) {
                    zmq::message_t msg;
                    auto ok = pull.recv(msg, zmq::recv_flags::dontwait);
                    if (!ok) break;

                    ReceiveCallback cbCopy;
                    {
                        std::lock_guard<std::mutex> lk(cbMtx_);
                        cbCopy = onReceive_;
                    }
                    if (cbCopy) cbCopy(msg.to_string());
                }
            }

            // 2) Drain outgoing queue and publish to clients
            flushOutgoing_(pub);

            // 3) If nothing to do, wait briefly to reduce CPU
            if (isOutEmpty_()) {
                std::unique_lock<std::mutex> lk(outMtx_);
                outCv_.wait_for(lk, std::chrono::milliseconds(1));
            }
        }

        // Best-effort final flush
        flushOutgoing_(pub);

        try { pull.close(); } catch (...) {}
        try { pub.close(); }  catch (...) {}
    }

    bool isOutEmpty_() const {
        std::lock_guard<std::mutex> lk(outMtx_);
        return outQueue_.empty();
    }

    void flushOutgoing_(zmq::socket_t& pub) {
        // Move queue to local batch to minimize lock time
        std::deque<std::string> batch;
        {
            std::lock_guard<std::mutex> lk(outMtx_);
            batch.swap(outQueue_);
        }

        for (auto& payload : batch) {
            try {
                // dontwait: drop if PUB side is under backpressure
                pub.send(zmq::buffer(payload), zmq::send_flags::dontwait);
            } catch (const zmq::error_t&) {
                // drop on error/backpressure
            }
        }
    }

private:
    Config cfg_;

    // RX callback (protected as it may be updated from other thread)
    mutable std::mutex cbMtx_;
    ReceiveCallback onReceive_;

    // ZMQ context
    zmq::context_t ctx_;

    // IO thread
    std::thread ioThread_;
    std::atomic<bool> running_{false};

    // Outgoing queue
    mutable std::mutex outMtx_;
    std::condition_variable outCv_;
    std::deque<std::string> outQueue_;
};
