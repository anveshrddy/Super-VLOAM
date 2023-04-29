#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H



#include <ros/package.h>
#include <ros/ros.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "/home/ubuntu/catkin_ws/src/visual_odometry/include/visual_odometry/SuperPoint.h"
#include <memory.h>

namespace vloam
{
  using KeyPointAndDesc = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

KeyPointAndDesc processOneFrame(const Ort::SuperPoint& osh, const cv::Mat& inputImg, float* dst, int borderRemove = 4,
                                float confidenceThresh = 0.015, bool alignCorners = true, int distThresh = 2);
                                void normalizeDescriptors(cv::Mat* descriptors);




enum class DetectorType
{
  ShiTomasi,
  BRISK,
  FAST,
  ORB,
  AKAZE,
  SIFT,
  SuperPoint
};
static const std::string DetectorType_str[] = { "ShiTomasi", "BRISK", "FAST", "ORB", "AKAZE", "SIFT","SuperPoint" };
enum class DescriptorType
{
  BRISK,
  ORB,
  BRIEF,
  AKAZE,
  FREAK,
  SIFT,
  SuperPoint

};
static const std::string DescriptorType_str[] = { "BRISK", "ORB", "BRIEF", "AKAZE", "FREAK", "SIFT","SuperPoint" };
enum class MatcherType
{
  BF,
  FLANN,
  SuperGlue
};
enum class SelectType
{
  NN,
  KNN
};

class ImageUtil
{
public:
  ImageUtil()
  {
    print_result = false;
    visualize_result = false;
    detector_type = DetectorType::SuperPoint;
    descriptor_type = DescriptorType::ORB;
    matcher_type = MatcherType::BF;
    selector_type = SelectType::NN;

    /*Loading Super point model*/
    
    // dst.push_back(Ort::SuperPoint::IMG_CHANNEL * Ort::SuperPoint::IMG_H * Ort::SuperPoint::IMG_W);
  }

  std::vector<cv::KeyPoint> detKeypoints(cv::Mat &img);
  std::vector<cv::KeyPoint> keyPointsNMS(std::vector<cv::KeyPoint> &&keypoints,
                                         const int bucket_width = 100,  // width for horizontal direction in image plane
                                                                        // => x, col
                                         const int bucket_height = 100,  // height for vertical direction in image plane
                                                                         // => y, row
                                         const int max_total_keypoints = 400);
  void saveKeypointsImage(const std::string file_name);
  cv::Mat descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
  // std::vector<cv::DMatch> matchDescriptors(cv::Mat &desc_source, cv::Mat &desc_ref);
  std::vector<cv::DMatch> matchDescriptors(cv::Mat& descriptors0, cv::Mat& descriptors1,cv::Mat& img0,cv::Mat& img1,std::vector<cv::KeyPoint> kp0,std::vector<cv::KeyPoint> kp1);

  void visualizeMatchesCallBack(int event, int x, int y);
  cv::Mat visualizeMatches(const cv::Mat &image0, const cv::Mat &image1, const std::vector<cv::KeyPoint> &keypoints0,
                           const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::DMatch> &matches);
  std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>, std::vector<uchar>>
  calculateOpticalFlow(const cv::Mat &image0, const cv::Mat &image1, const std::vector<cv::KeyPoint> &keypoints0);
  cv::Mat visualizeOpticalFlow(const cv::Mat &image1, const std::vector<cv::Point2f> &keypoints0_2f,
                               const std::vector<cv::Point2f> &keypoints1_2f,
                               const std::vector<uchar> &optical_flow_status);
  cv::Mat visualizeOpticalFlow(const cv::Mat &image1, const std::vector<cv::KeyPoint> &keypoints0,
                               const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::DMatch> &matches);

  // std::string path_prefix;
  bool print_result;
  bool visualize_result;
  DetectorType detector_type;
  DescriptorType descriptor_type;
  MatcherType matcher_type;
  SelectType selector_type;

  int remove_VO_outlier;
  bool optical_flow_match;


// KeyPointAndDesc results;


// std::shared_ptr<Ort::SuperPoint> osh;
   

private:
  double time;
  cv::Mat img_keypoints;
  const int IMG_HEIGHT = 375;
  const int IMG_WIDTH = 1242;
};
}  // namespace vloam

#endif