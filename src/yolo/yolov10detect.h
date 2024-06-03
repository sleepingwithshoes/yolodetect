//
// Created by steffens on 02.06.24.
//

#ifndef YOLOV10DETECT_H
#define YOLOV10DETECT_H

#include "IYoloDetectBase.h"
#include "boundingbox.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace sleepingwithshoes::yolo {

class YoloV10detect final : public IYoloDetectBase {
    public:
    YoloV10detect (const std::string& modelPath, const std::string& classesPath, const std::pair<int, int>& inputSize, torch::Device device);

    std::vector<BoundingBox> detect (const cv::Mat& image, float score_thresh = 0.5) override;

    private:
    std::vector<torch::Tensor> _getBoxes (const torch::Tensor& preds, float score_thresh, torch::Device device);
};
}


#endif //YOLOV10DETECT_H
