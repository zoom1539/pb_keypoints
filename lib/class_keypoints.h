#pragma once

#include "opencv2/opencv.hpp"

// Result for COCO (17 body parts)
// {0,  "Nose"},
// {1,  "LEye"},
// {2,  "REye"},
// {3,  "LEar"},
// {4,  "REar"},
// {5,  "LShoulder"},
// {6,  "RShoulder"},
// {7,  "LElbow"},
// {8,  "RElbow"},
// {9,  "LWrist"},
// {10, "RWrist"},
// {11, "LHip"},
// {12, "RHip"},
// {13, "LKnee"},
// {14, "Rknee"},
// {15, "LAnkle"},
// {16, "RAnkle"},

class Keypoints
{ 
public:
    explicit Keypoints();
    ~Keypoints();

    bool serialize(const std::string &wts_path_, const std::string &engine_path_);

    // 
    bool init(const std::string &engine_path_);
    bool run(const std::vector<cv::Mat> &imgs_, 
             std::vector<std::vector<cv::Point2f> > &vec_coords_,
             std::vector<std::vector<float> > &vec_scores_);
    
   
    

private:
    Keypoints(const Keypoints &);
    const Keypoints &operator=(const Keypoints &);

    class Impl;
    Impl *_impl;
};
