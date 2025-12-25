#include <zmq.hpp>
#include <chrono>
#include <thread>
#include <iostream>

int main() {
    zmq::context_t ctx(1);

    // Receive from server (PUB -> SUB)
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.set(zmq::sockopt::subscribe, ""); // subscribe all
    sub.connect("tcp://127.0.0.1:5555");

    // Send to server (PUSH -> PULL)
    zmq::socket_t push(ctx, zmq::socket_type::push);
    push.connect("tcp://127.0.0.1:5556");

    auto nextSend = std::chrono::steady_clock::now();
    auto period100hz = std::chrono::milliseconds(10);

    while (true) {
        // 100Hz send
        auto now = std::chrono::steady_clock::now();
        if (now >= nextSend) {
            nextSend += period100hz;
            push.send(zmq::buffer("hello-from-client"), zmq::send_flags::dontwait);
        }

        // non-blocking receive from server
        zmq::message_t msg;
        auto ok = sub.recv(msg, zmq::recv_flags::dontwait);
        if (ok) {
            std::cout << "RX from server: " << msg.to_string() << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
