//
// Created by sleepingwithshpes
//

#pragma once

#include "IYoloDetectBase.h"
#include "boundingbox.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace sleepingwithshoes::yolo {

    class YoloV8detect final : public IYoloDetectBase {
    public:
        YoloV8detect(const std::string& modelPath, const std::string& classesPath,const std::pair<int, int>& inputSize, torch::Device device, float iou_thresh = 0.4);

        std::vector<BoundingBox> detect(const cv::Mat& image, float score_thresh) override;
    private:
        std::vector<torch::Tensor> _nonMaxSupression(const torch::Tensor& preds, float score_thresh, float iou_thresh, torch::Device device);
        float _iou_thresh;
    };
}



