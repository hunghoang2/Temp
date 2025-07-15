cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX)

# — use at least C++11 (or newer if you like) —
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# — find Eigen (headers only) —
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# — find OpenCV —
find_package(OpenCV REQUIRED)

# — your executable and sources —
add_executable(my_app
    src/main.cpp
    # … other .cpp/.h files …
)

# — link Eigen’s imported target —
target_link_libraries(my_app PRIVATE Eigen3::Eigen)

# — link OpenCV libraries —
# either with the old-style variable:
target_include_directories(my_app PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(   my_app PRIVATE ${OpenCV_LIBS})

# — or, using OpenCV’s modern imported targets (CMake ≥3.16+) —
# find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
# target_link_libraries(my_app PRIVATE
#     OpenCV::core
#     OpenCV::imgproc
#     OpenCV::highgui
# )
