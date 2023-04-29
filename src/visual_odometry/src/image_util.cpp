#include </home/ubuntu/catkin_ws/src/visual_odometry/include/visual_odometry/image_util.h>
#include <chrono>
using namespace std::chrono;

// namespace
// {
// using KeyPointAndDesc = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

// KeyPointAndDesc processOneFrame(const Ort::SuperPoint& osh, const cv::Mat& inputImg, float* dst, int borderRemove = 4,
//                                 float confidenceThresh = 0.015, bool alignCorners = true, int distThresh = 2);

// }  // namespace


//const static std::string ONNX_MODEL_PATH = "/home/ubuntu/catkin_ws/src/visual_odometry/super_point.onnx";
//const static std::string ONNX_MODEL_PATH = "/home/ubuntu/onnx_runtime_cpp/super_point.onnx";
const static std::string TORCHSCRIPT_MODEL_PATH = "/home/ubuntu/onnx_runtime_cpp/scripts/superpoint/scripted_superpoint.pt";
static Ort::SuperPoint osh(TORCHSCRIPT_MODEL_PATH, 1,
                        std::vector<std::vector<int64_t>>{
                            {1, Ort::SuperPoint::IMG_CHANNEL, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_W}});;

static const int DUMMY_NUM_KEYPOINTS = 256;
static Ort::OrtSessionHandler superGlueOsh("/home/ubuntu/onnx_runtime_cpp/super_glue.onnx", 1,
                                        std::vector<std::vector<int64_t>>{
                                            {4},
                                            {1, DUMMY_NUM_KEYPOINTS},
                                            {1, DUMMY_NUM_KEYPOINTS, 2},
                                            {1, 256, DUMMY_NUM_KEYPOINTS},
                                            {4},
                                            {1, DUMMY_NUM_KEYPOINTS},
                                            {1, DUMMY_NUM_KEYPOINTS, 2},
                                            {1, 256, DUMMY_NUM_KEYPOINTS},
                                        });

namespace vloam
{

    
std::vector<cv::KeyPoint> ImageUtil::detKeypoints(cv::Mat& img)
{
  std::vector<cv::KeyPoint> keypoints;

  if (print_result)
    time = (double)cv::getTickCount();

  if (detector_type == DetectorType::ShiTomasi)
  {
    int block_size =
        5;  //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    // double max_overlap = 0.0; // max. permissible overlap between two features in %
    // double min_distance = (1.0 - max_overlap) * block_size;
    double min_distance = block_size * 1.5;
    // int maxCorners = img.rows * img.cols / std::max(1.0, min_distance); // max. num. of keypoints
    int maxCorners = 1024;

    double quality_level = 0.03;  // minimal accepted quality of image corners
    double k = 0.04;

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, quality_level, min_distance, cv::Mat(), block_size, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
      cv::KeyPoint new_keypoint;
      new_keypoint.pt = cv::Point2f((*it).x, (*it).y);
      new_keypoint.size = block_size;
      keypoints.push_back(new_keypoint);
    }
  }
  else if (detector_type == DetectorType::FAST)
  {
    int threshold = 100;
    cv::FAST(img, keypoints, threshold, true);
  }
  else
  {
    cv::Ptr<cv::FeatureDetector> detector;
    if (detector_type == DetectorType::BRISK)
      detector = cv::BRISK::create();
    else if (detector_type == DetectorType::ORB)
    {
      int num_features = 2000;
      float scaleFactor = 1.2f;
      int nlevels = 8;
      int edgeThreshold = 31;
      int firstLevel = 0;
      int WTA_K = 2;
      cv::ORB::ScoreType scoreType = cv::ORB::FAST_SCORE;
      int patchSize = 31;
      int fastThreshold = 20;
      detector = cv::ORB::create(num_features, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType,
                                 patchSize, fastThreshold);
    }
    else if (detector_type == DetectorType::AKAZE)
      detector = cv::AKAZE::create();
    // else if (detector_type == DetectorType::SIFT)
    //   detector = cv::SIFT::create();
    else if(detector_type==DetectorType::SuperPoint)
    {

      
 
    // Ort::SuperPoint osh("/home/ubuntu/catkin_ws/src/visual_odometry/super_point.onnx", 0,
    //                     std::vector<std::vector<int64_t>>{
    //                         {1, Ort::SuperPoint::IMG_CHANNEL, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_W}});


// std::fill(dst.begin(),dst.end(),0.0f);
    std::vector<float> dst(Ort::SuperPoint::IMG_CHANNEL * Ort::SuperPoint::IMG_H * Ort::SuperPoint::IMG_W);
 
    KeyPointAndDesc results;
    
    auto start = high_resolution_clock::now();
    results=processOneFrame(osh, img, dst.data()); 
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout<< "Time taken for getting key points: " << duration.count() << std::endl;
    //std::cerr << "Detector is SuperPoint no of key points = " <<results.first.size()<< std::endl;




     
      // results=processOneFrame(osh, img, dst.data());
      return results.first;
    }
    else
    {
      std::cerr << "Detector is not implemented" << std::endl;
      exit(EXIT_FAILURE);
    }

    detector->detect(img, keypoints);
  }

  if (print_result)
  {
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    std::cout << DetectorType_str[static_cast<int>(detector_type)] + "detection with n=" << keypoints.size()
              << " keypoints in " << 1000 * time / 1.0 << " ms" << std::endl;
  }

  if (visualize_result)
  {
    // std::vector<cv::KeyPoint> fake_keypoints;
    // fake_keypoints.push_back(keypoints[0]);
    // std::cout << "fake keypoints 0: " << keypoints[0].pt.x << ", " << keypoints[0].pt.y << std::endl;

    img_keypoints = img.clone();
    // cv::drawKeypoints(img, fake_keypoints, img_keypoints, cv::Scalar::all(-1),
    // cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string window_name = DetectorType_str[static_cast<int>(detector_type)] + " Detector Results";
    cv::namedWindow(window_name, 6);
    cv::imshow(window_name, img_keypoints);
    cv::waitKey(0);
  }

  return keypoints;
}

std::vector<cv::KeyPoint> ImageUtil::keyPointsNMS(  // TODO: check if opencv detector minDistance helps here
    std::vector<cv::KeyPoint>&& keypoints,
    const int bucket_width,   // width for horizontal direction in image plane => x, col
    const int bucket_height,  // height for vertical direction in image plane => y, row
    const int max_total_keypoints)
{
  const int bucket_shape_x = std::ceil(static_cast<float>(IMG_WIDTH) / static_cast<float>(bucket_width));    // 13
  const int bucket_shape_y = std::ceil(static_cast<float>(IMG_HEIGHT) / static_cast<float>(bucket_height));  // 4

  const int max_bucket_keypoints = max_total_keypoints / (bucket_shape_x * bucket_shape_y);  // 7

  std::vector<std::vector<std::vector<cv::KeyPoint>>> bucket(
      bucket_shape_x, std::vector<std::vector<cv::KeyPoint>>(bucket_shape_y, std::vector<cv::KeyPoint>()));

  // put all keypoints into buckets
  for (const auto& keypoint : keypoints)
  {
    bucket[static_cast<int>(keypoint.pt.x / static_cast<float>(bucket_width))]
          [static_cast<int>(keypoint.pt.y / static_cast<float>(bucket_height))]
              .push_back(keypoint);
  }

  std::vector<cv::KeyPoint> keypoints_after_NMS;
  keypoints_after_NMS.reserve(max_total_keypoints);

  auto keypoints_sort = [](const cv::KeyPoint& kp0, const cv::KeyPoint& kp1) { return kp0.response > kp1.response; };

  // iterate all bucket, sort and put keypoints with top response to the return
  int col, row;
  for (col = 0; col < bucket_shape_x; ++col)
  {
    for (row = 0; row < bucket_shape_y; ++row)
    {
      if (bucket[col][row].empty())
        continue;

      if (bucket[col][row].size() <= max_bucket_keypoints)
      {
        std::copy(bucket[col][row].begin(), bucket[col][row].end(), std::back_inserter(keypoints_after_NMS));
        continue;
      }

      std::sort(bucket[col][row].begin(), bucket[col][row].end(),
                keypoints_sort);  // ascending order of keypoint response
      std::copy(bucket[col][row].begin(), bucket[col][row].begin() + max_bucket_keypoints,
                std::back_inserter(keypoints_after_NMS));
    }
  }

  return keypoints_after_NMS;
}

void ImageUtil::saveKeypointsImage(const std::string file_name)
{
  if (!img_keypoints.data)
  {
    printf("No keypoints data \n");
    return;
  }
  cv::imwrite(ros::package::getPath("visual_odometry") + "/figures/" + file_name + ".png", img_keypoints);
}

cv::Mat ImageUtil::descKeypoints(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img)
{
  cv::Mat descriptors;

  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (descriptor_type == DescriptorType::BRISK)
  {
    int threshold = 30;          // FAST/AGAST detection threshold score.
    int octaves = 3;             // detection octaves (use 0 to do single scale)
    float pattern_scale = 1.0f;  // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

    extractor = cv::BRISK::create(threshold, octaves, pattern_scale);
  }
  // else if (descriptor_type == DescriptorType::BRIEF) {
  //     extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
  // }
  else if (descriptor_type == DescriptorType::ORB)
  {
    extractor = cv::ORB::create();
  }
  // else if (descriptor_type == DescriptorType::FREAK) {
  //     extractor = cv::xfeatures2d::FREAK::create();
  // }
  else if (descriptor_type == DescriptorType::AKAZE)
  {
    extractor = cv::AKAZE::create();
  }
  // else if (descriptor_type == DescriptorType::SIFT)
  // {
  //   extractor = cv::SIFT::create();
  // }
  else if(descriptor_type == DescriptorType::SuperPoint)
  {
    //  Ort::SuperPoint osh("/home/ubuntu/catkin_ws/src/visual_odometry/super_point.onnx", 0,
    //                     std::vector<std::vector<int64_t>>{
    //                         {1, Ort::SuperPoint::IMG_CHANNEL, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_W}});


// std::fill(dst.begin(),dst.end(),0.0f);
    std::vector<float> dst(Ort::SuperPoint::IMG_CHANNEL * Ort::SuperPoint::IMG_H * Ort::SuperPoint::IMG_W);
 
    KeyPointAndDesc results;
    
    auto start = high_resolution_clock::now();

    results=processOneFrame(osh, img, dst.data()); 
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout<< "Desc Time taken: " << duration.count() << std::endl;
    
    //std::cerr << "Descriptor is SuperPoint  " << results.second.size()<<std::endl;
     normalizeDescriptors(&results.second);


      return  (results.second);
  }
  else
  {
    std::cerr << "Decscriptor is not implemented" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (print_result)
    time = (double)cv::getTickCount();

  extractor->compute(img, keypoints, descriptors);

  if (print_result)
  {
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    std::cout << DescriptorType_str[static_cast<int>(descriptor_type)] << " descriptor extraction in "
              << 1000 * time / 1.0 << " ms" << std::endl;
  }

  return descriptors;
}

std::vector<cv::DMatch> ImageUtil::matchDescriptors(cv::Mat& descriptors0, cv::Mat& descriptors1,cv::Mat& img0,cv::Mat& img1,std::vector<cv::KeyPoint> kp0,std::vector<cv::KeyPoint> kp1)
{
  std::vector<cv::DMatch> matches;
  bool crossCheck = (selector_type == SelectType::NN);
  cv::Ptr<cv::DescriptorMatcher> matcher;

  // Reference: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
  if (matcher_type == MatcherType::BF)
  {
    int normType;
    if (descriptor_type == DescriptorType::AKAZE or descriptor_type == DescriptorType::BRISK or
        descriptor_type == DescriptorType::ORB )
    {
      normType = cv::NORM_HAMMING;
    }
    else if (descriptor_type == DescriptorType::SIFT  ) //or descriptor_type == DescriptorType::SuperPoint
    {
      normType = cv::NORM_L2;
    }
    else
    {
      std::cerr << "match Decscriptor is not implemented" << std::endl;
    }
    matcher = cv::BFMatcher::create(normType, crossCheck);
  }
  else if (matcher_type == MatcherType::FLANN)
  {
    std::cerr << "FLANN is used" << std::endl;
    if (descriptors0.type() != CV_32F)
    {  // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV
       // implementation
      descriptors0.convertTo(descriptors0, CV_32F);
    }
    if (descriptors1.type() != CV_32F)
    {  // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV
       // implementation
      descriptors1.convertTo(descriptors1, CV_32F);
    }

    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  }
  else if (matcher_type == MatcherType::SuperGlue)
  {

     // superglue
    auto start = high_resolution_clock::now();
    std::vector<cv::Mat> images;
    images.push_back(img0);
     images.push_back(img1);
     std::vector<KeyPointAndDesc> superPointResults;
     superPointResults.push_back(make_pair(kp0,descriptors0));
     superPointResults.push_back(make_pair(kp1,descriptors1));


    // superglue
    // static const int DUMMY_NUM_KEYPOINTS = 256;
    // Ort::OrtSessionHandler superGlueOsh("/home/ubuntu/onnx_runtime_cpp/super_glue.onnx", 0,
    //                                     std::vector<std::vector<int64_t>>{
    //                                         {4},
    //                                         {1, DUMMY_NUM_KEYPOINTS},
    //                                         {1, DUMMY_NUM_KEYPOINTS, 2},
    //                                         {1, 256, DUMMY_NUM_KEYPOINTS},
    //                                         {4},
    //                                         {1, DUMMY_NUM_KEYPOINTS},
    //                                         {1, DUMMY_NUM_KEYPOINTS, 2},
    //                                         {1, 256, DUMMY_NUM_KEYPOINTS},
    //                                     });

    int numKeypoints0 = superPointResults[0].first.size();
    int numKeypoints1 = superPointResults[1].first.size();
    std::vector<std::vector<int64_t>> inputShapes = {
        {4}, {1, numKeypoints0}, {1, numKeypoints0, 2}, {1, 256, numKeypoints0},
        {4}, {1, numKeypoints1}, {1, numKeypoints1, 2}, {1, 256, numKeypoints1},
    };
    superGlueOsh.updateInputShapes(inputShapes);

    std::vector<std::vector<float>> imageShapes(2);
    std::vector<std::vector<float>> scores(2);
    std::vector<std::vector<float>> keypoints(2);
    std::vector<std::vector<float>> descriptors(2);

    cv::Mat buffer;
    for (int i = 0; i < 2; ++i) {
        imageShapes[i] = {1, 1, static_cast<float>(images[0].rows), static_cast<float>(images[0].cols)};
        std::transform(superPointResults[i].first.begin(), superPointResults[i].first.end(),
                       std::back_inserter(scores[i]), [](const cv::KeyPoint& keypoint) { return keypoint.response; });
        for (const auto& k : superPointResults[i].first) {
            keypoints[i].emplace_back(k.pt.y);
            keypoints[i].emplace_back(k.pt.x);
        }

        transposeNDWrapper(superPointResults[i].second, {1, 0}, buffer);
        std::copy(buffer.begin<float>(), buffer.end<float>(), std::back_inserter(descriptors[i]));
        buffer.release();
    }
    std::vector<Ort::OrtSessionHandler::DataOutputType> superGlueOrtOutput =
        superGlueOsh({imageShapes[0].data(), scores[0].data(), keypoints[0].data(), descriptors[0].data(),
                      imageShapes[1].data(), scores[1].data(), keypoints[1].data(), descriptors[1].data()});

    // match keypoints 0 to keypoints 1
    std::vector<int64_t> matchIndices(reinterpret_cast<int64_t*>(superGlueOrtOutput[0].first),
                                      reinterpret_cast<int64_t*>(superGlueOrtOutput[0].first) + numKeypoints0);

    std::vector<cv::DMatch> goodMatches;
    for (std::size_t i = 0; i < matchIndices.size(); ++i) {
        if (matchIndices[i] < 0) {
            continue;
        }
        cv::DMatch match;
        match.imgIdx = 0;
        match.queryIdx = i;
        match.trainIdx = matchIndices[i];
        goodMatches.emplace_back(match);
    }
    //std::cout << " Superglue with n=" << goodMatches.size() << " matches in "<< std::endl;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout<< "Time taken for glue: " << duration.count() << " :: :: " <<" Good Matches "<< goodMatches.size()<<std::endl;
    return goodMatches;


  }

  if (print_result)
    time = (double)cv::getTickCount();

  if (selector_type == SelectType::NN)
  {
    matcher->match(descriptors0, descriptors1, matches);

    if (print_result)
    {
      time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
      std::cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * time / 1.0 << " ms" << std::endl;
    }
  }
  else if (selector_type == SelectType::KNN)
  {  // k nearest neighbors (k=2)
    // double t = (double)cv::getTickCount();

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors0, descriptors1, knn_matches, 2);
    for (const auto& knn_match : knn_matches)
    {
      if (knn_match[0].distance < 0.8 * knn_match[1].distance)
      {
        matches.push_back(knn_match[0]);
      }
    }

    if (print_result)
    {
      time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
      std::cout << " (KNN) with n=" << matches.size() << " matches in " << 1000 * time / 1.0 << " ms" << std::endl;
      std::cout << "KNN matches = " << knn_matches.size() << ", qualified matches = " << matches.size()
                << ", discard ratio = " << (float)(knn_matches.size() - matches.size()) / (float)knn_matches.size()
                << std::endl;
    }
  }

  if (print_result)
    std::cout << "MATCH KEYPOINT DESCRIPTORS done, and the number of matches is " << matches.size() << std::endl;

  return matches;
}

void ImageUtil::visualizeMatchesCallBack(int event, int x, int y)
{
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
  }
}

void visualizeMatchesOnMouse(int ev, int x, int y, int, void* obj)
{
  ImageUtil* iu = static_cast<ImageUtil*>(obj);
  if (iu)
    iu->visualizeMatchesCallBack(ev, x, y);
}

cv::Mat ImageUtil::visualizeMatches(const cv::Mat& image0, const cv::Mat& image1,
                                    const std::vector<cv::KeyPoint>& keypoints0,
                                    const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::DMatch>& matches)
{
  std::vector<cv::DMatch> matches_dnsp;
  const int stride = std::ceil(static_cast<float>(matches.size()) / 100.0f);  // at most 100 points

  int prev_pt_x, prev_pt_y, curr_pt_x, curr_pt_y;
  for (int i = 0; i < matches.size(); i += stride)
  {
    prev_pt_x = keypoints0[matches[i].queryIdx].pt.x;
    prev_pt_y = keypoints0[matches[i].queryIdx].pt.y;
    curr_pt_x = keypoints1[matches[i].trainIdx].pt.x;
    curr_pt_y = keypoints1[matches[i].trainIdx].pt.y;
    if (remove_VO_outlier > 0)
    {
      if (std::pow(prev_pt_x - curr_pt_x, 2) + std::pow(prev_pt_y - curr_pt_y, 2) >
          remove_VO_outlier * remove_VO_outlier)
        continue;
    }
    matches_dnsp.push_back(matches[i]);
  }

  cv::Mat matchImg = image1.clone();
  cv::drawMatches(image0, keypoints0, image1, keypoints1, matches_dnsp, matchImg, cv::Scalar::all(-1),
                  cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);

  // std::string windowName = "Matching keypoints between two camera images";
  // cv::namedWindow(windowName, 7);
  // cv::setMouseCallback(windowName, visualizeMatchesOnMouse, this);
  // cv::imshow(windowName, matchImg);
  // std::cout << "Press key to continue to next image" << std::endl;
  // cv::waitKey(0); // wait for key to be pressed
  // ROS_INFO("image showed");

  return matchImg;
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>, std::vector<uchar>> ImageUtil::calculateOpticalFlow(
    const cv::Mat& image0, const cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints0)
{
  std::vector<cv::Point2f> keypoints0_2f;
  std::vector<cv::Point2f> keypoints1_2f;
  std::vector<uchar> optical_flow_status;

  // transform keypoints to points_2f
  std::transform(keypoints0.cbegin(), keypoints0.cend(), std::back_inserter(keypoints0_2f),
                 [](const cv::KeyPoint& keypoint) { return keypoint.pt; });

  // calculate optical flow
  std::vector<float> err;
  cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
  cv::calcOpticalFlowPyrLK(image0, image1, keypoints0_2f, keypoints1_2f, optical_flow_status, err, cv::Size(15, 15), 2,
                           criteria);

  ROS_INFO("Optical Flow: total candidates = %ld",
           std::count(optical_flow_status.cbegin(), optical_flow_status.cend(), 1));

  return std::make_tuple(std::cref(keypoints0_2f), std::cref(keypoints1_2f), std::cref(optical_flow_status));
}

cv::Mat ImageUtil::visualizeOpticalFlow(const cv::Mat& image1, const std::vector<cv::Point2f>& keypoints0_2f,
                                        const std::vector<cv::Point2f>& keypoints1_2f,
                                        const std::vector<uchar>& optical_flow_status)
{
  // Create some random colors
  cv::Mat image1_color = image1.clone();
  cv::cvtColor(image1_color, image1_color, cv::COLOR_GRAY2BGR);
  cv::RNG rng;
  cv::Scalar color;
  int r, g, b, j;
  for (j = 0; j < keypoints0_2f.size(); ++j)
  {
    // Select good points
    if (optical_flow_status[j] == 1)
    {
      // draw the tracks
      r = rng.uniform(0, 256);
      g = rng.uniform(0, 256);
      b = rng.uniform(0, 256);
      color = cv::Scalar(r, g, b);
      cv::line(image1_color, keypoints1_2f[j], keypoints0_2f[j], color, 2);
      cv::circle(image1_color, keypoints1_2f[j], 3, color, -1);
    }
  }

  return image1_color;
}
cv::Mat ImageUtil::visualizeOpticalFlow(const cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints0,
                                        const std::vector<cv::KeyPoint>& keypoints1,
                                        const std::vector<cv::DMatch>& matches)
{
  // Create some random colors
  cv::Mat image1_color = image1.clone();
  cv::cvtColor(image1_color, image1_color, cv::COLOR_GRAY2BGR);
  cv::RNG rng;
  cv::Scalar color;
  int r, g, b, j;
  for (const auto match : matches)
  {
    // draw the tracks
    r = rng.uniform(0, 256);
    g = rng.uniform(0, 256);
    b = rng.uniform(0, 256);
    color = cv::Scalar(r, g, b);
    cv::line(image1_color, keypoints1[match.trainIdx].pt, keypoints0[match.queryIdx].pt, color, 2);
    cv::circle(image1_color, keypoints1[match.trainIdx].pt, 3, color, -1);
  }

  return image1_color;
}

void normalizeDescriptors(cv::Mat* descriptors)
{
    cv::Mat rsquaredSumMat;
    cv::reduce(descriptors->mul(*descriptors), rsquaredSumMat, 1, cv::REDUCE_SUM);
    cv::sqrt(rsquaredSumMat, rsquaredSumMat);
    for (int i = 0; i < descriptors->rows; ++i) {
        float rsquaredSum = std::max<float>(rsquaredSumMat.ptr<float>()[i], 1e-12);
        descriptors->row(i) /= rsquaredSum;
    }
}

KeyPointAndDesc processOneFrame(const Ort::SuperPoint& osh, const cv::Mat& inputImg, float* dst, int borderRemove,
                                float confidenceThresh, bool alignCorners, int distThresh)
{
    
    int origW = inputImg.cols, origH = inputImg.rows;
    std::cerr << origW<<origH << std::endl;
    cv::Mat scaledImg;
    cv::resize(inputImg, scaledImg, cv::Size(Ort::SuperPoint::IMG_W, Ort::SuperPoint::IMG_H), 0, 0, cv::INTER_CUBIC);
    
    osh.preprocess(dst, scaledImg.data, Ort::SuperPoint::IMG_W, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_CHANNEL);
    
    auto inferenceOutput = osh({dst});
    

    std::vector<cv::KeyPoint> keyPoints = osh.getKeyPoints(inferenceOutput, borderRemove, confidenceThresh);

    std::vector<int> descriptorShape(inferenceOutput[1].second.begin(), inferenceOutput[1].second.end());
    cv::Mat coarseDescriptorMat(descriptorShape.size(), descriptorShape.data(), CV_32F,
                                inferenceOutput[1].first);  // 1 x 256 x H/8 x W/8

    std::vector<int> keepIndices = osh.nmsFast(keyPoints, Ort::SuperPoint::IMG_H, Ort::SuperPoint::IMG_W, distThresh);

    std::vector<cv::KeyPoint> keepKeyPoints;
    keepKeyPoints.reserve(keepIndices.size());
    std::transform(keepIndices.begin(), keepIndices.end(), std::back_inserter(keepKeyPoints),
                   [&keyPoints](int idx) { return keyPoints[idx]; });
    keyPoints = std::move(keepKeyPoints);

    cv::Mat descriptors = osh.getDescriptors(coarseDescriptorMat, keyPoints, Ort::SuperPoint::IMG_H,
                                             Ort::SuperPoint::IMG_W, alignCorners);

    for (auto& keyPoint : keyPoints) {
        keyPoint.pt.x *= static_cast<float>(origW) / Ort::SuperPoint::IMG_W;
        keyPoint.pt.y *= static_cast<float>(origH) / Ort::SuperPoint::IMG_H;
    }

    return {keyPoints, descriptors};
}

KeyPointAndDesc processOneFrame(const Ort::SuperPoint& osh, const cv::Mat& inputImg, float* dst, int borderRemove, float confidenceThresh, bool alignCorners, int distThresh)
{

  
}

}  
