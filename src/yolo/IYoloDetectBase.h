//
// Created by steffens on 02.06.24.
//

#pragma once

#include "boundingbox.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace sleepingwithshoes::yolo {

class IYoloDetectBase {
    public:
    IYoloDetectBase (const std::string& modelPath, const std::string& classesPath, const std::pair<int, int>& inputSize, torch::Device device);

    ~IYoloDetectBase () = default;

    virtual std::vector<BoundingBox> detect (const cv::Mat& image, float score_thresh) = 0;

    bool operator== (const IYoloDetectBase&) const = default;

    protected:
    torch::Device getDevice () { return _device; }
    std::pair<int, int> getInputSize () { return _inputSize; }
    torch::jit::script::Module getModel () { return _model; }
    std::vector<std::string> getClassnames () { return _classnames; }
    std::vector<cv::Scalar> getColors () { return _colors; }

    private:
    void _loadModel (const std::filesystem::path& modelAbsPath);
    void _loadClasses (const std::filesystem::path& classesAbsPath);
    void _generateColors (int classCounts);

    torch::Device _device;
    std::pair<int, int> _inputSize;
    torch::jit::script::Module _model;
    std::vector<std::string> _classnames = {};
    std::vector<cv::Scalar> _colors      = {};

};
}


