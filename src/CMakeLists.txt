add_executable(${PROJECT_NAME}
        main.cpp
        yolo/yolov8detect.cpp
        yolo/yolov8detect.h
        yolo/boundingbox.h
)

target_link_libraries(${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)