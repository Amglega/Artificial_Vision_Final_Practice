#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <image_transport/image_transport.hpp>

#include <memory>
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types_conversion.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/filters/extract_indices.h>


pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud);

class ComputerVisionSubscriber : public rclcpp::Node
{
  public:
    ComputerVisionSubscriber()
    : Node("opencv_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      subscription_3d_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/head_front_camera/depth_registered/points", qos, std::bind(&ComputerVisionSubscriber::topic_callback_3d, this, std::placeholders::_1));
    
      publisher_3d_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pcl_points", qos);
    }

  private:

    void topic_callback_3d(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {    
      
      // Convert to PCL data type
      pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
      pcl::fromROSMsg(*msg, point_cloud);     

      /*
      BOOST_FOREACH( const pcl::PointXYZRGB& pt, point_cloud.points)
      {          
          std::cout  << "x: " << pt.x <<"\n";
          std::cout  << "y: " << pt.y <<"\n";
          std::cout  << "z: " << pt.z <<"\n";
          std::cout  << "rgb: " << (int)pt.r << "-" << (int)pt.g << "-" << (int)pt.b <<"\n";
          std::cout << "---------" << "\n";
      }
      */

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
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;

  pcl::PointCloud<pcl::PointXYZHSV>::Ptr pointcloud_hsv(new pcl::PointCloud<pcl::PointXYZHSV>);
  pcl::PointCloud<pcl::PointXYZHSV>::Ptr pointcloud_filtered(new pcl::PointCloud<pcl::PointXYZHSV>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);

  //change color space
  pcl::PointCloudXYZRGBtoXYZHSV(in_pointcloud, *pointcloud_hsv);
  
  // Build the condition to detect the blue balls
  pcl::ConditionAnd<pcl::PointXYZHSV>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZHSV> ());
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("h", pcl::ComparisonOps::LT, 161.0*(360.0/255.0))));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("h", pcl::ComparisonOps::GT, 61.0*(360.0/255.0))));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("s", pcl::ComparisonOps::LT, 255.0/255.0)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("s", pcl::ComparisonOps::GT, 190.0/255.0)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("v", pcl::ComparisonOps::LT, 255.0/255.0)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZHSV>::Ptr(new pcl::FieldComparison<pcl::PointXYZHSV>("v", pcl::ComparisonOps::GT, 55.0/255.0)));
  
  // Build the filter
  pcl::ConditionalRemoval<pcl::PointXYZHSV> range_filt;
  range_filt.setInputCloud(pointcloud_hsv);
  range_filt.setCondition(range_cond);
  range_filt.filter(*pointcloud_filtered);
  
  //change the color space to normal
  for (auto& point : *pointcloud_filtered){
    pcl::PointXYZRGB color_point;

    pcl::PointXYZHSVtoXYZRGB(point, color_point);
    pointcloud_rgb->push_back(color_point);
  }
  
  //outlier removal
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_inliers(new pcl::PointCloud<pcl::PointXYZRGB>);

  //create and aply the statistical filter
  if (pointcloud_filtered->size() > 0){
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud (pointcloud_rgb);
    sor.setMeanK (20);
    sor.setStddevMulThresh (1.0);
    sor.filter (*pointcloud_inliers);
  }

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
    ransac.getModelCoefficients(coef);
    draw_cube(pointcloud_inliers,coef[0],coef[1],coef[2],0,0,255);

    pcl::PointIndices::Ptr sphere_inliers(new pcl::PointIndices());

    for (auto& indice : inliers){
      sphere_inliers->indices.push_back(indice);
    }
  
    extract.setInputCloud(aux_cloud);
    extract.setIndices(sphere_inliers);
    extract.setNegative(true);
    extract.filter(*aux_cloud);
    
    
  }
  
  //draw the cubes at the designated distance
  for (int n = 0; n < 6;n++){
    draw_cube(pointcloud_inliers,1,0.88,3+n,255-n*42,n*42,0);
    draw_cube(pointcloud_inliers,-1,0.88,3+n,255-n*42,n*42,0);
  }
  

  out_pointcloud = *pointcloud_inliers;
  
  return out_pointcloud;
}


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}