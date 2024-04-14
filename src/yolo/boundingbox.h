//
// Created by sleepingwithshoes
//

#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace sleepingwithshoes::yolo {
    struct BoundingBox {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        int width = 0;
        int height = 0;
        double score = 0.0;
        int classID = -1;
        std::string className = "";
        cv::Scalar color = cv::Scalar(0, 0, 0);
    };
}