add_executable(${PROJECT_NAME}
        main.cpp
        yolo/boundingbox.h
        yolo/IYoloDetectBase.h
        yolo/IYoloDetectBase.cpp
        yolo/yolov8detect.cpp
        yolo/yolov8detect.h
        yolo/yolov10detect.cpp
        yolo/yolov10detect.h
)

target_link_libraries(${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)