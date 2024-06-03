//
// Created by steffens on 02.06.24.
//
#include "IYoloDetectBase.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <fstream>
#include <random>

namespace sleepingwithshoes::yolo {
IYoloDetectBase::IYoloDetectBase(const std::string& modelPath, const std::string& classesPath, const std::pair<int, int>& inputSize, torch::Device device)
:  _device(device)
, _inputSize(inputSize)
{
    _loadModel(std::filesystem::absolute(modelPath));
    _loadClasses(std::filesystem::absolute(classesPath));
    _generateColors(_classnames.size());
}
void IYoloDetectBase::_loadModel(const std::filesystem::path& modelAbsPath) {
    try {
        _model = torch::jit::load(modelAbsPath, _device);
        std::cout << "Model loaded: " << modelAbsPath.string() << std::endl;
    } catch (const c10::Error &e) {
        throw std::runtime_error("Error loading the model: " + modelAbsPath.string() + "\n" + e.what());
    } catch (const std::exception &e) {
        throw std::runtime_error("Error loading the model: " + modelAbsPath.string() + "\n" + e.what());
    }
}

void IYoloDetectBase::_loadClasses(const std::filesystem::path& classesAbsPath) {
    if(!std::filesystem::exists(classesAbsPath)) {
        throw std::runtime_error("classes doesn't exists: " + classesAbsPath.string());
    }

    std::ifstream file(classesAbsPath);
    std::string name;
    while (std::getline(file, name)) {
        _classnames.push_back(name);
    }
    std::cout << "Classes loaded: " << classesAbsPath.string() << std::endl;
}

void IYoloDetectBase::_generateColors(int classCounts) {
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < classCounts; ++i) {
        std::uniform_int_distribution<> dis(0, 255);
        auto color =  cv::Scalar(dis(gen), dis(gen), dis(gen));
        _colors.push_back(color);
    }
}

}