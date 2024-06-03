#include "yolo/yolov8detect.h"
#include "yolo/yolov10detect.h"

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>

#define YOLOV10

int main () {
    using namespace sleepingwithshoes::yolo;

    std::cout << "Hello, Yolo!" << std::endl;

    // set device to cuda if available
    torch::Device device (torch::cuda::is_available () ?
        torch::kCUDA :
        torch::kCPU);
    std::cout << "CUDA " << (device.is_cuda () ? "is" : "is not") <<
    " available" << std::endl;

#ifndef YOLOV10
    YoloV8detect yoloV8detect ("../../model/yolov8/yolov8s640.torchscript",
                               "../../model/classes/coco.names",
                               { 640, 640 },
                               device);
    IYoloDetectBase& yoloDetect = yoloV8detect;
#else
    YoloV10detect yoloV10detect ("../../model/yolov10/yolov10s640.torchscript",
                                 "../../model/classes/coco.names",
                                 { 640, 640 },
                                 device);
    IYoloDetectBase& yoloDetect = yoloV10detect;
#endif
    //cv::VideoCapture cap (0); // webcam laptop
    cv::VideoCapture cap (4); // webcam external
    if (!cap.isOpened ()) {
        throw std::runtime_error ("No video stream detected");
    }
    cap.set (cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set (cv::CAP_PROP_FRAME_HEIGHT, 640);
    auto capWidth  = cap.get (cv::CAP_PROP_FRAME_WIDTH);
    auto capHeight = cap.get (cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "CAP width: " << std::to_string (capWidth) << " height: " <<
    std::to_string (capHeight) << std::endl;


    cv::namedWindow ("My Video", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;
    while (true) {
        auto timestart = static_cast<double> (cv::getTickCount ());

        // Read frame
        cap.read (frame);
        if (frame.empty ()) {
            std::cout << "Read frame failed!" << std::endl;
            break;
        }

        cv::flip (frame, frame, 1);

        // Object detection
        auto boxes = yoloDetect.detect (frame, 0.5f);

        for (auto& box : boxes) {
            cv::rectangle (
                frame, cv::Rect (box.x1, box.y1, (box.width), (box.height)),
                box.color, 2);

            float fontScale = std::min (
                0.7f, std::max (2.0f, static_cast<float> (box.width) / 256.0f));
            std::string text = box.className + ": " + cv::format (
                "%.2f", box.score);
            cv::putText (frame, text, cv::Point (box.x1, box.y1 - 2),
                         cv::FONT_HERSHEY_SIMPLEX, fontScale, box.color, 2);
        }

        double fps = cv::getTickFrequency () / (static_cast<double> (
            cv::getTickCount ()) - timestart);
        std::stringstream stream;
        stream << std::fixed << std::setprecision (2) << fps;
        cv::putText (frame, "FPS :" + stream.str (), cv::Point (20, 20),
                     cv::FONT_HERSHEY_SIMPLEX,
                     0.7f, cv::Scalar (0, 255, 0), 2);

        cv::imshow ("My Video", frame);

        char key = static_cast<char> (cv::waitKey (1));
        if (key == 27) {
            // 'ESC' is entered break the loop
            break;
        }
    }
    cap.release ();
    cv::destroyAllWindows ();
    return 0;
}
