#pragma once

// std
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "class_keypoints.h"

using namespace nvinfer1;

class _Keypoints
{
public:
    _Keypoints();
    ~_Keypoints();

public:
    bool serialize(const std::string &wts_path_, const std::string &engine_path_);

    bool init(const std::string &engine_path_);
    bool run(const std::vector<cv::Mat> &imgs_, 
             std::vector<std::vector<cv::Point2f> > &vec_coords_,
             std::vector<std::vector<float> > &vec_scores_);
    


private:
    nvinfer1::IRuntime* _runtime;
    nvinfer1::ICudaEngine* _engine;
    nvinfer1::IExecutionContext* _context;
    cudaStream_t _stream;

    void* _buffers[2];


};
