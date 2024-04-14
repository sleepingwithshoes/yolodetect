//
// Created by sleepingwithshpes
//

#pragma once

#include "boundingbox.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace sleepingwithshoes::yolo {

    class YoloV8detect {
    public:
        YoloV8detect(const std::string& modelPath, const std::string& classesPath,const std::pair<int, int>& inputSize, torch::Device device);
        ~YoloV8detect() = default;

        std::vector<BoundingBox> detect(const cv::Mat& image, float score_thresh = 0.5, float iou_thresh = 0.4);

    private:
        void _loadModel(const std::filesystem::path& modelAbsPath);
        void _loadClasses(const std::filesystem::path& classesAbsPath);
        void _generateColors(int classCounts);
        std::vector<torch::Tensor> _nonMaxSupression(const torch::Tensor& preds, float score_thresh, float iou_thresh, torch::Device device);

        torch::Device _device;
        std::pair<int, int> _inputSize;
        torch::jit::script::Module _model;
        std::vector<std::string> _classnames = {};
        std::vector<cv::Scalar> _colors = {};
    };
}



