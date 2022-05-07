#include <memory>

#include <fstream>
#include <sstream>
#include <iostream>
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <image_transport/image_transport.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types_conversion.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/normal_3d_omp.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>


using namespace cv;
using namespace dnn;
using namespace std;

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud);
cv::Mat image_processing(const cv::Mat in_image);

std::vector<std::vector<double>> Centers;
std::vector<std::string> classes;
//bool variables that set the detection mode
bool blueball_detected = false; 
bool table_detected = false;




class ComputerVisionSubscriber : public rclcpp::Node
{
  public:
    ComputerVisionSubscriber()
    : Node("opencv_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);

      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/rgb/image_raw", qos, std::bind(&ComputerVisionSubscriber::topic_callback, this, std::placeholders::_1));
    
      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "cv_image", qos);
      
    }

  private:
    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw =  cv_ptr->image;

      // Image processing
      cv::Mat cv_image = image_processing(image_raw);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; // >> message to be sent
      img_bridge.toImageMsg(out_image); // from cv_bridge to sensor_msgs::Image

      // Publish the data
      publisher_ -> publish(out_image);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

class PCLSubscriber : public rclcpp::Node
{
  public:
    PCLSubscriber()
    : Node("pcl_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      subscription_3d_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/head_front_camera/depth_registered/points", qos, std::bind(&PCLSubscriber::topic_callback_3d, this, std::placeholders::_1));
    
      publisher_3d_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pcl_points", qos);
    }

  private:
    void topic_callback_3d(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {    
      // Convert to PCL data type
      pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
      pcl::fromROSMsg(*msg, point_cloud);     

      pcl::PointCloud<pcl::PointXYZRGB> pcl_pointcloud = pcl_processing(point_cloud);
      
      // Convert to ROS data type
      sensor_msgs::msg::PointCloud2 output;
      pcl::toROSMsg(pcl_pointcloud, output);
      output.header = msg->header;

      // Publish the data
      publisher_3d_ -> publish(output);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_3d_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_3d_;
};


// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}
//returns a string vector that contains the names of the detected objects
vector<string> getObjectNames(cv::Mat& frame, const vector<cv::Mat>& outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    float confThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    
    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    vector<string> names;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        names.push_back(classes[classIds[idx]]);
    }
    return names;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const vector<cv::Mat>& outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    float confThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    
    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}
// Get the names of the output layers
vector<String> getOutputsNames(const Net& net) {
    static vector<String> names;
    if (names.empty()) {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

cv::Mat image_processing(const cv::Mat in_image) 
{
  cv::Mat out_image = in_image.clone();
  cv::Mat image_HSV, image_filtered,gray_image, three_channels;
  std::vector<cv::Mat> channels;
  std::vector<cv::Vec3f> balls_contour;
  cv::cvtColor(in_image, image_HSV, cv::COLOR_BGR2HSV);
  cv::inRange(image_HSV, cv::Scalar(70, 190, 55), cv::Scalar(160, 255, 255), image_filtered);
  channels.push_back(image_filtered);
  channels.push_back(image_filtered);
  channels.push_back(image_filtered);
  cv::merge(channels, three_channels);
  cv::cvtColor(three_channels, gray_image, cv::COLOR_BGR2GRAY);
  cv::HoughCircles(gray_image, balls_contour, cv::HOUGH_GRADIENT, 1, 50,100, 15, 10, 300);

  if (balls_contour.size() > 0 && cv::countNonZero(image_filtered) > 600){
    blueball_detected = true;
    table_detected = false;
    //intrinsic matrix values
    double intrinsic_data[9] ={ 522.1910329546544, 0.0, 320.5, 0.0, 522.1910329546544, 240.5, 0.0, 0.0, 1.0};
    cv::Mat K_matrix(3,3, CV_64FC1, intrinsic_data);

    //extrinsic matrix values:
    double roll, pitch, yaw, r11, r12, r13, r21, r22, r23, r31, r32, r33;
    roll = 0.0;
    pitch = 0.0;
    yaw = 0.0;
    r11 = cos(yaw) * cos(pitch);
    r12 = cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll);
    r13 = cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll);
    r21 = sin(yaw) * cos(pitch);
    r22 = sin(yaw) * sin(pitch) * sin(yaw) + cos(yaw) * cos(roll);
    r23 = sin(yaw) * sin(pitch) * cos(yaw) - cos(yaw) * sin(roll);
    r31 = -sin(pitch);
    r32 = cos(pitch) * sin(roll);
    r33 = cos(pitch) * cos(roll);
    double y = 0.98;
    double x = 0.0;
    double z = 0.125;
    double extrinsic_data[12] = { r11, r12, r13, x, r21, r22, r23, y, r31, r32, r33, z};
    cv::Mat Rt_matrix(3,4, CV_64FC1, extrinsic_data);

    //obtain the proyection matrix
    cv::Mat P_matrix = K_matrix * Rt_matrix;
    
    cv::Mat res_1,res_2,res_3;
    cv::Point pt1, pt2, pt3;
    int distance = 8;
    //calculate the points depending on the distance argument

    for(int i = 0; i < distance - 2;i++){

      double point_1[4] = { -1.0, 0.0, 3.0 + i, 1.0};
      
      cv::Mat Point1(4,1, CV_64FC1, point_1);

      double point_2[4] = { 1.0, 0.0, 3.0 + i, 1.0};

      cv::Mat Point2(4,1, CV_64FC1, point_2);

      res_1 = P_matrix * Point1;

      res_2 = P_matrix * Point2;

      pt1.x = res_1.at<double>(0,0) * (1/res_1.at<double>(2,0));
      pt1.y = res_1.at<double>(1,0) * (1/res_1.at<double>(2,0));

      pt2.x = res_2.at<double>(0,0) * (1/res_2.at<double>(2,0));
      pt2.y = res_2.at<double>(1,0) * (1/res_1.at<double>(2,0));

      //Write the text with the distance and draw the line adn points on the output image

      cv::putText(out_image, std::to_string(3+i), pt2, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, i*50, 255-i*50),1.5);
      cv::line(out_image, pt2, pt1,cv::Scalar(0, i*50, 255-i*50), 2, cv::LINE_8);
      cv::circle( out_image, pt1, 5, cv::Scalar(0, i*50, 255-i*50), -1 );
      cv::circle( out_image, pt2, 5, cv::Scalar(0, i*50, 255-i*50), -1 );
      
    }
    
    for(size_t j = 0; j < Centers.size();j++){
      
      double center_point[4] = { Centers.at((int)j).at(0), (Centers.at((int)j).at(1)-y), Centers.at((int)j).at(2), 1.0};
      
      cv::Mat Point3(4,1, CV_64FC1, center_point);

      res_3 = P_matrix * Point3;

      pt3.x = res_3.at<double>(0,0) * (1/res_3.at<double>(2,0));
      pt3.y = res_3.at<double>(1,0) * (1/res_3.at<double>(2,0));
      cv::circle( out_image, pt3, 10, cv::Scalar(255, 0, 255), -1 );
    }
    
    for( size_t i = 0; i < balls_contour.size(); i++ ) {
      cv::Vec3i c = balls_contour[i];
      cv::Point center = cv::Point(c[0], c[1]);
      // circle center
      cv::circle( out_image, center, 1, cv::Scalar(0,0,0), 3, cv::LINE_AA);
      // circle outline
      int radius = c[2];
      cv::circle( out_image, center, radius, cv::Scalar(0,0,255), 3, cv::LINE_AA);
    }
    
      
  
  }else{
    blueball_detected = false;
    
    
    int inpWidth = 416;  // Width of network's input image
    int inpHeight = 416; // Height of network's input image
    
    std::string package_share_directory = ament_index_cpp::get_package_share_directory("computer_vision");
    // Load names of classes
    string classesFile = package_share_directory + "/cfg/coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    String modelConfiguration = package_share_directory + "/cfg/yolov3.cfg";
    
    String modelWeights = package_share_directory + "/cfg/yolov3.weights";
    
    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    
    net.setPreferableBackend(DNN_TARGET_CPU);

    cv::Mat blob;

    // Create a 4D blob from a frame.
    blobFromImage(out_image, blob, 1/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
        
    //Sets the input to the network
    net.setInput(blob);
    
    vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));
    vector<string> objects = getObjectNames(out_image,outs);
        
    if (std::find(objects.begin(), objects.end(), "diningtable") != objects.end())
    {
      table_detected = true;
      // Runs the forward pass to get output of the output layers

      // Remove the bounding boxes with low confidence
      postprocess(out_image, outs);
    }else{
      table_detected = false;
    }
        
  }

  return out_image;
}
//function to draw a cube in a designated point in the point cloud
void draw_cube(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud,float x, float y, float z, int red, int green, int blue){

  int size = 10;
  std::uint8_t r = red, g = green, b = blue;  	
  for (int i = (x*100-size); i < (x*100+size) ; i++){
    for(int j = (y*100-size); j < (y*100+size) ; j++){
      for(int k = (z*100-size); k < (z*100+size) ; k++){
        pcl::PointXYZRGB rgb_point; 
        rgb_point.x = i*0.01;
        rgb_point.y = j*0.01;
        rgb_point.z =	k*0.01;
        rgb_point.r = r;
        rgb_point.g = g;
        rgb_point.b = b;
        pointcloud->push_back(rgb_point);
      }
    }
  }
  return;
}

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
{
  // Create output pointcloud
  
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud = in_pointcloud;
  
  pcl::PointCloud<pcl::PointXYZHSV>::Ptr pointcloud_hsv(new pcl::PointCloud<pcl::PointXYZHSV>);
  pcl::PointCloud<pcl::PointXYZHSV>::Ptr pointcloud_filtered(new pcl::PointCloud<pcl::PointXYZHSV>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

  //change color space
  pcl::PointCloudXYZRGBtoXYZHSV(in_pointcloud, *pointcloud_hsv);
  
  // Build the condition to detect the blue balls
  pcl::ConditionAnd<pcl::PointXYZHSV>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZHSV> ());
  
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("h", pcl::ComparisonOps::LT, 161.0*(360.0/255.0))));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("h", pcl::ComparisonOps::GT, 65.0*(360.0/255.0))));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("s", pcl::ComparisonOps::LT, 255.0/255.0)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("s", pcl::ComparisonOps::GT, 190.0/255.0)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("v", pcl::ComparisonOps::LT, 255.0/255.0)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("v", pcl::ComparisonOps::GT, 55.0/255.0)));

  // Build the filter
  pcl::ConditionalRemoval<pcl::PointXYZHSV> range_filt;
  range_filt.setInputCloud(pointcloud_hsv);
  range_filt.setCondition(range_cond);
  range_filt.filter(*pointcloud_filtered);

  if (blueball_detected){
    //change the color space to normal
    Centers.clear();
    for (auto& point : *pointcloud_filtered){
      pcl::PointXYZRGB color_point;

      pcl::PointXYZHSVtoXYZRGB(point, color_point);
      pointcloud_rgb->push_back(color_point);
    }
    
    //outlier removal
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_inliers(new pcl::PointCloud<pcl::PointXYZRGB>);

    //create and aply the statistical filter
    
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud (pointcloud_rgb);
    sor.setMeanK (20);
    sor.setStddevMulThresh (1.0);
    sor.filter (*pointcloud_inliers);

    //find the spheres with ransac
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    if (pointcloud_inliers->size() > 0){
      pcl::copyPointCloud (*pointcloud_inliers, *aux_cloud);
    }
    
    while (aux_cloud->size() > 0){
      pcl::ExtractIndices<pcl::PointXYZRGB> extract;
      std::vector<int> inliers;
      pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>::Ptr model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZRGB> (aux_cloud));

      pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_s);
      ransac.setDistanceThreshold (0.01);
      if (!(ransac.computeModel())){
        break;
      }
      ransac.getInliers(inliers);

      Eigen::VectorXf coef;
      std::vector<double> circle_center;

      ransac.getModelCoefficients(coef);
      circle_center.push_back(coef[0]);
      circle_center.push_back(coef[1]);
      circle_center.push_back(coef[2]);
      Centers.push_back(circle_center);
      draw_cube(pointcloud_inliers,coef[0],coef[1],coef[2],0,0,255);   
      
    }

    //draw the cubes at the designated distance
    for (int n = 0; n < 6;n++){
      draw_cube(pointcloud_inliers,1,0.88,3+n,255-n*42,n*42,0);
      draw_cube(pointcloud_inliers,-1,0.88,3+n,255-n*42,n*42,0);
    }
    
    out_pointcloud = *pointcloud_inliers;
  }else if (table_detected){
    
    
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr pointcloud_hsv(new pcl::PointCloud<pcl::PointXYZHSV>);
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr pointcloud_filtered(new pcl::PointCloud<pcl::PointXYZHSV>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

    //change color space
    pcl::PointCloudXYZRGBtoXYZHSV(in_pointcloud, *pointcloud_hsv);
    
    // Build the condition to detect the table
    pcl::ConditionAnd<pcl::PointXYZHSV>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZHSV> ());
  
    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("h", pcl::ComparisonOps::LT, 38.0*(360.0/255.0))));
    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("h", pcl::ComparisonOps::GT, 27.0*(360.0/255.0))));
    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("s", pcl::ComparisonOps::LT, 255.0/255.0)));
    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("s", pcl::ComparisonOps::GT, 25.0/255.0)));
    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("v", pcl::ComparisonOps::LT, 255.0/255.0)));
    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("v", pcl::ComparisonOps::GT, 150.0/255.0)));
    // Build the filter
    pcl::ConditionalRemoval<pcl::PointXYZHSV> range_filt;
    range_filt.setInputCloud(pointcloud_hsv);
    range_filt.setCondition(range_cond);
    range_filt.filter(*pointcloud_filtered);
    //outlier removal
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_inliers(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr objects_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pointcloud_filtered->size() > 20){
      //change the color space to normal
      Centers.clear();
      for (auto& point : *pointcloud_filtered){
        pcl::PointXYZRGB color_point;

        pcl::PointXYZHSVtoXYZRGB(point, color_point);
        pointcloud_rgb->push_back(color_point);
      }

      //create and aply the statistical filter
      
      pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
      sor.setInputCloud (pointcloud_rgb);
      sor.setMeanK (20);
      sor.setStddevMulThresh (1.0);
      sor.filter (*pointcloud_inliers);
      
    }
    //find the corner points of the table
    double x_min, x_max, y_min, z_min, z_max;
    x_min = 999.0;
    x_max = -999.0;
    y_min = 999.0;
    z_min = 999.0;
    z_max = -999.0;
    for (auto& point : *pointcloud_inliers){
      if (point.x > x_max){
        x_max = point.x ;
      }
      if (point.x < x_min){
        x_min = point.x ;
      }
      if (point.y < y_min){
        y_min = point.y ;
      }
      if (point.z > z_max){
        z_max = point.z ;
      }
      if (point.z < z_min){
        z_min = point.z ;
      }
    }

    for (auto& object_point : in_pointcloud){
      if (object_point.x > x_min && object_point.x <x_max && object_point.y < y_min && object_point.z > z_min && object_point.z < z_max){
        objects_cloud->push_back(object_point);
      }
      
    }

    pcl::copyPointCloud (*objects_cloud, *aux_cloud);
    //if there are points in the cloud proccess it to find the closest cylinder object
    if (aux_cloud->size() > 0){
      //onjects needed to recognise the cylinder pattern 
      pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);   
      std::vector<int> inliers; 

      // Estimate point normals
      ne.setSearchMethod (tree);
      ne.setInputCloud (aux_cloud);
      ne.setRadiusSearch (0.03);
      ne.compute (*cloud_normals);
      
      //create the model
      pcl::SampleConsensusModelCylinder<pcl::PointXYZRGB,pcl::Normal>::Ptr model_c(new pcl::SampleConsensusModelCylinder<pcl::PointXYZRGB,pcl::Normal> (aux_cloud));
      model_c->setInputNormals(cloud_normals);
      model_c->setNormalDistanceWeight(0.5);
      pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_c);
      ransac.setDistanceThreshold (0.01);
      //if there is a model with the desired characteristics draw a red cube in its position
      if ((ransac.computeModel())){
        ransac.getInliers(inliers);
        Eigen::VectorXf coef;

        ransac.getModelCoefficients(coef);

        draw_cube(objects_cloud, coef[0], coef[1], coef[2], 255, 0, 0);
        
        pcl::PointIndices::Ptr cylinder_inliers(new pcl::PointIndices());      

      }
    }
    out_pointcloud=*objects_cloud;

  }
  
  return out_pointcloud;
}


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
 
  rclcpp::executors::SingleThreadedExecutor exec;

  auto cv_node = std::make_shared<ComputerVisionSubscriber>();
  auto pcl_node = std::make_shared<PCLSubscriber>();
  exec.add_node(cv_node);
  exec.add_node(pcl_node);
  exec.spin();
  
  rclcpp::shutdown();
  return 0;
}