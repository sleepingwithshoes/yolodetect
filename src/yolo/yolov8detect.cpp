//
// Created by sleepingwithshoes on 06.04.24.
//

#include "yolov8detect.h"


#include <torch/script.h>
#include <torch/torch.h>
#include <random>

namespace sleepingwithshoes::yolo {

YoloV8detect::YoloV8detect (const std::string& modelPath, const std::string& classesPath, const std::pair<int, int>& inputSize, torch::Device device, float iou_thresh)
    : IYoloDetectBase (modelPath, classesPath, inputSize, device)
      , _iou_thresh (iou_thresh) {
}

std::vector<BoundingBox> YoloV8detect::detect (const cv::Mat& image, float score_thresh) {
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
    auto detectionsOutput = _nonMaxSupression (output.toTensor ().to (getDevice ()), score_thresh, _iou_thresh, getDevice ());

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


std::vector<torch::Tensor> YoloV8detect::_nonMaxSupression (const torch::Tensor& preds, float score_thresh, float iou_thresh, torch::Device device) {
    std::vector<torch::Tensor> detectionOutputs;

    // preds [1, 84, 8400]
    // remove batch dimension and transpose to [8400, 84]
    torch::Tensor batch = preds[0].squeeze (0).transpose (0, 1).to (device);

    auto maxScores = std::get<0> (torch::max (batch.slice (1, 4, batch.sizes ()[1]), 1)).to (device);
    batch          = torch::index_select (batch, 0, torch::nonzero (maxScores > score_thresh).select (1, 0)).to (device);
    if (batch.sizes ()[0] == 0) {
        return detectionOutputs;
    }
    // compute center_x, center_y, w, h -> x1, y1, x2, y2
    torch::Tensor x1    = batch.select (1, 0) - (batch.select (1, 2) / 2).to (device); // x - w/2
    torch::Tensor y1    = batch.select (1, 1) - (batch.select (1, 3) / 2).to (device); // y - h/2
    torch::Tensor x2    = batch.select (1, 0) + (batch.select (1, 2) / 2).to (device); // x + w/2
    torch::Tensor y2    = batch.select (1, 1) + (batch.select (1, 3) / 2).to (device); // y + h/2
    batch.select (1, 0) = x1.to (device);
    batch.select (1, 1) = y1.to (device);
    batch.select (1, 2) = x2.to (device);
    batch.select (1, 3) = y2.to (device);

    auto maxScoresIncludeIndex = torch::max (batch.slice (1, 4, batch.sizes ()[1]).to (device), 1);
    batch.select (1, 4)        = std::get<0> (maxScoresIncludeIndex).to (device); // score
    batch.select (1, 5)        = std::get<1> (maxScoresIncludeIndex).to (device); // classID
    torch::Tensor detections   = batch.slice (1, 0, 6).to (device);               //detection results

    torch::Tensor keep                                     = torch::empty ({ detections.sizes ()[0] }).to (device);
    torch::Tensor areas                                    = (detections.select (1, 3) - detections.select (1, 1)) * (detections.select (1, 2) - detections.select (1, 0)).to (device);
    std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort (detections.select (1, 4).to (device), 0, 1);
    torch::Tensor indexes                                  = std::get<1> (indexes_tuple).to (device);
    int count                                              = 0;
    while (indexes.sizes ()[0] > 0) {
        keep[count] = (indexes[0].item ().toInt ());
        count += 1;

        // Computing overlaps
        torch::Tensor lefts   = torch::empty (indexes.sizes ()[0] - 1).to (device);
        torch::Tensor tops    = torch::empty (indexes.sizes ()[0] - 1).to (device);
        torch::Tensor rights  = torch::empty (indexes.sizes ()[0] - 1).to (device);
        torch::Tensor bottoms = torch::empty (indexes.sizes ()[0] - 1).to (device);
        torch::Tensor widths  = torch::empty (indexes.sizes ()[0] - 1).to (device);
        torch::Tensor heights = torch::empty (indexes.sizes ()[0] - 1).to (device);
        for (int idx = 0; idx < indexes.sizes ()[0] - 1; ++idx) {
            lefts[idx]   = std::max (detections[indexes[0]][0].item ().toFloat (), detections[indexes[idx + 1]][0].item ().toFloat ());
            tops[idx]    = std::max (detections[indexes[0]][1].item ().toFloat (), detections[indexes[idx + 1]][1].item ().toFloat ());
            rights[idx]  = std::min (detections[indexes[0]][2].item ().toFloat (), detections[indexes[idx + 1]][2].item ().toFloat ());
            bottoms[idx] = std::min (detections[indexes[0]][3].item ().toFloat (), detections[indexes[idx + 1]][3].item ().toFloat ());
            widths[idx]  = std::max (float (0), rights[idx].item ().toFloat () - lefts[idx].item ().toFloat ());
            heights[idx] = std::max (float (0), bottoms[idx].item ().toFloat () - tops[idx].item ().toFloat ());
        }
        torch::Tensor overlaps = (widths * heights).to (device);

        // FIlter by IOUs
        torch::Tensor ious = overlaps / (areas.select (0, indexes[0].item ().toInt ()).to (device) +
            torch::index_select (areas, 0, indexes.slice (0, 1, indexes.sizes ()[0]).to (device)) -
            overlaps).to (device);
        indexes = torch::index_select (indexes, 0, torch::nonzero (ious <= iou_thresh).select (1, 0).to (device) + 1).to (device);
    }
    keep               = keep.toType (torch::kInt64).to (device);
    torch::Tensor test = torch::index_select (detections, 0, keep.slice (0, 0, count).to (device)).to (device);

    for (int j = 0; j < test.size (0); ++j) {
        detectionOutputs.push_back (test.select (0, j));
    }
    return detectionOutputs;
}
}
