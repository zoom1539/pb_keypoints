#pragma once

#include "opencv2/opencv.hpp"


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
