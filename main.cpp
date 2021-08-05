#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <chrono>
#include "class_keypoints.h"

int main()
{
    Keypoints keypoints;

    //
    std::string wts_path = "../lib/simple_res50_256x192.wts";
    std::string engine_path = "../lib/simple_res50_256x192.engine";
#if 0
    bool is_serialized = keypoints.serialize(wts_path, engine_path);
    if(!is_serialized)
    {
        std::cout << "serialize fail\n";
        return 0;
    }
#else

    bool is_init = keypoints.init(engine_path);
    if(!is_init)
    {
        std::cout << "serialize fail\n";
        return 0;
    }

    std::vector<cv::Mat> imgs;
    {
        cv::Mat img = cv::imread("../data/test1.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test2.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test3.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test4.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test1.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test2.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test3.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test4.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test1.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test2.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test3.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test4.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test1.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test2.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test3.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test4.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test1.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test2.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test3.jpg");
        imgs.push_back(img);
    }
    {
        cv::Mat img = cv::imread("../data/test4.jpg");
        imgs.push_back(img);
    }
   
    
    auto start = std::chrono::system_clock::now();
    std::vector<std::vector<cv::Point2f> > vec_coords;
    std::vector<std::vector<float> > vec_scores;
    bool is_run = keypoints.run(imgs, vec_coords, vec_scores);

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " total ms" << std::endl;

    for (int i = 0; i < vec_coords.size(); i++)
    {
        for (int j = 0; j < vec_coords[i].size(); j++)
        {
            // std::cout << vec_coords[i][j] << std::endl;
            if (vec_scores[i][j] > 0.3)
            {
                cv::circle(imgs[i], cv::Point(vec_coords[i][j].x, vec_coords[i][j].y), 3, cv::Scalar(0, 0, 255), 1);
            }
        }

        std::stringstream ss;
        ss << "../data/" << i << ".jpg";
        cv::imwrite(ss.str(), imgs[i]);
    }

    for (int i = 0; i < vec_scores.size(); i++)
    {
        for (int j = 0; j < vec_scores[i].size(); j++)
        {
            std::cout << vec_scores[i][j] << std::endl;
        }
        std::cin.get();

    }

#endif

    std::cin.get();
    return 0;
}

