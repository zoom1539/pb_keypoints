#include "class_keypoints.h"
#include "class_keypoints_.h"

class Keypoints::Impl
{
public:
    _Keypoints _keypoints;
};

Keypoints::Keypoints() : _impl(new Keypoints::Impl())
{
}

Keypoints::~Keypoints()
{
    delete _impl;
    _impl = NULL;
}

bool Keypoints::serialize(const std::string &wts_path_, const std::string &engine_path_)
{
    return _impl->_keypoints.serialize(wts_path_, engine_path_);
}

bool Keypoints::init(const std::string &engine_path_)
{
    return _impl->_keypoints.init(engine_path_);
}

bool Keypoints::run(const std::vector<cv::Mat> &imgs_, 
             std::vector<std::vector<cv::Point2f> > &vec_coords_,
             std::vector<std::vector<float> > &vec_scores_)
{
    return _impl->_keypoints.run(imgs_, vec_coords_, vec_scores_);
}
