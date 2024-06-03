//
// Created by sleepingwithshoes on 06.04.24.
//

#include "yolov10detect.h"


#include <torch/script.h>
#include <torch/torch.h>
#include <fstream>
#include <random>

namespace sleepingwithshoes::yolo {

YoloV10detect::YoloV10detect (const std::string& modelPath, const std::string& classesPath, const std::pair<int, int>& inputSize, torch::Device device)
    : IYoloDetectBase (modelPath, classesPath, inputSize, device) {
}

std::vector<BoundingBox> YoloV10detect::detect (const cv::Mat& image, float score_thresh) {
    cv::Mat resizedImage;

    // Preprocessing
    cv::resize (image, resizedImage, cv::Size (getInputSize ().first, getInputSize ().second));
    cv::cvtColor (resizedImage, resizedImage, cv::COLOR_BGR2RGB);

    // Preparing input tensor
    torch::Tensor imgTensor = torch::from_blob (resizedImage.data, { resizedImage.rows, resizedImage.cols, 3 }, torch::kByte);
    imgTensor               = imgTensor.permute ({ 2, 0, 1 }).toType (torch::kFloat).div (255).unsqueeze (0).to (getDevice ());

    auto output = getModel ().forward ({ imgTensor.to (getDevice ()) });

    if (output.isNone ()) {
        throw std::runtime_error ("model output is None");
    }
    auto detectionsOutput = _getBoxes (output.toTensor ().to (getDevice ()), score_thresh, getDevice ());

    const float scale_x = static_cast<float> (image.cols) / static_cast<float> (getInputSize ().first);
    const float scale_y = static_cast<float> (image.rows) / static_cast<float> (getInputSize ().second);

    std::vector<BoundingBox> boxes;
    for (auto& detection : detectionsOutput) {
        BoundingBox box;
        box.x1        = static_cast<int> (detection[0].item<float> () * scale_x);
        box.y1        = static_cast<int> (detection[1].item<float> () * scale_y);
        box.x2        = static_cast<int> (detection[2].item<float> () * scale_x);
        box.y2        = static_cast<int> (detection[3].item<float> () * scale_y);
        box.score     = detection[4].item<double> ();
        box.classID   = detection[5].item<int> ();
        box.width     = box.x2 - box.x1;
        box.height    = box.y2 - box.y1;
        box.className = getClassnames ()[box.classID];
        box.color     = getColors ()[box.classID];
        boxes.push_back (box);
    }

    return boxes;
}

std::vector<torch::Tensor> YoloV10detect::_getBoxes (const torch::Tensor& preds, float score_thresh, torch::Device device) {
    std::vector<torch::Tensor> detectionOutputs;

    // preds [1, 300, 6]

    // remove the batch dimension
    torch::Tensor batch = preds.squeeze (0).to (device); // [300, 6]

    // extract objectness score
    torch::Tensor objectness_score = batch.select (1, 4).to (device); // select the 5th colum (index 4)

    // Get the indices of tensor with objectness > score_thresh
    torch::Tensor valid_indices = torch::nonzero (objectness_score >= score_thresh).squeeze (1).to (device);

    // Extract the valid tensors using indexing
    torch::Tensor valid_tensors = torch::index_select (batch, 0, valid_indices).to (device);

    for (int i = 0; i < valid_tensors.sizes ()[0]; ++i) {
        detectionOutputs.push_back (valid_tensors[i].to (device));
    }

    return detectionOutputs;

}

}
