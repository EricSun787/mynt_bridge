// Copyright 2018 Slightech Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mynteye/api/api.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Header.h>

#include <mynt_bridge/calcDis.h>
#include <mynt_bridge/reqImage.h>
#include <numeric>

namespace enc = sensor_msgs::image_encodings;

namespace {

class DepthRegion {
 public:
  explicit DepthRegion(std::uint32_t n)
      : n_(std::move(n)), show_(false), selected_(false), point_(0, 0) {}

  ~DepthRegion() = default;

  /**
   * 鼠标事件：默认不选中区域，随鼠标移动而显示。单击后，则会选中区域来显示。你可以再单击已选中区域或双击未选中区域，取消选中。
   */
  void OnMouse(const int &event, const int &x, const int &y, const int &flags) {
    MYNTEYE_UNUSED(flags)
    if (event != CV_EVENT_MOUSEMOVE && event != CV_EVENT_LBUTTONDOWN) {
      return;
    }
    show_ = true;

    if (event == CV_EVENT_MOUSEMOVE) {
      if (!selected_) {
        point_.x = x;
        point_.y = y;
      }
    } else if (event == CV_EVENT_LBUTTONDOWN) {
      if (selected_) {
        if (x >= static_cast<int>(point_.x - n_) &&
            x <= static_cast<int>(point_.x + n_) &&
            y >= static_cast<int>(point_.y - n_) &&
            y <= static_cast<int>(point_.y + n_)) {
          selected_ = false;
        }
      } else {
        selected_ = true;
      }
      point_.x = x;
      point_.y = y;
    }
  }

  template <typename T>
  void ShowElems(
      const cv::Mat &depth,
      std::function<std::string(const T &elem)> elem2string,
      int elem_space = 40,
      std::function<std::string(
          const cv::Mat &depth, const cv::Point &point, const std::uint32_t &n)>
          getinfo = nullptr) {
    if (!show_)
      return;

    int space = std::move(elem_space);
    int n = 2 * n_ + 1;
    cv::Mat im(space * n, space * n, CV_8UC3, cv::Scalar(255, 255, 255));

    int x, y;
    std::string str;
    int baseline = 0;
    for (int i = -n_; i <= n; ++i) {
      x = point_.x + i;
      if (x < 0 || x >= depth.cols)
        continue;
      for (int j = -n_; j <= n; ++j) {
        y = point_.y + j;
        if (y < 0 || y >= depth.rows)
          continue;

        str = elem2string(depth.at<T>(y, x));
        printf("Depth is %d \n",depth.at<T>(y,x));

        cv::Scalar color(0, 0, 0);
        if (i == 0 && j == 0)
          color = cv::Scalar(0, 0, 255);

        cv::Size sz =
            cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        cv::putText(
            im, str, cv::Point(
                         (i + n_) * space + (space - sz.width) / 2,
                         (j + n_) * space + (space + sz.height) / 2),
            cv::FONT_HERSHEY_PLAIN, 1, color, 1);
      }
    }

    if (getinfo) {
      std::string info = getinfo(depth, point_, n_);
      if (!info.empty()) {
        cv::Size sz =
            cv::getTextSize(info, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        cv::putText(
            im, info, cv::Point(5, 5 + sz.height), cv::FONT_HERSHEY_PLAIN, 1,
            cv::Scalar(255, 0, 255), 1);
      }
    }

    cv::imshow("region", im);
  }

  void DrawRect(cv::Mat &image) {  // NOLINT
    if (!show_)
      return;
    std::uint32_t n = (n_ > 1) ? n_ : 1;
    n += 1;  // outside the region
    cv::rectangle(
        image, cv::Point(point_.x - n, point_.y - n),
        cv::Point(point_.x + n, point_.y + n),
        selected_ ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);
  }

 private:
  std::uint32_t n_;
  bool show_;
  bool selected_;
  cv::Point point_;
};

void OnDepthMouseCallback(int event, int x, int y, int flags, void *userdata) {
  DepthRegion *region = reinterpret_cast<DepthRegion *>(userdata);
  region->OnMouse(event, x, y, flags);
}

}  // namespace








MYNTEYE_USE_NAMESPACE
using namespace cv_bridge;
image_transport::CameraPublisher left_pub_;
std::map<Stream, sensor_msgs::CameraInfoPtr> camera_info_ptrs_;
std::map<Stream, std::string> camera_encodings_;
cv::Mat left_p_;
api::StreamData buffer_depth;
api::StreamData buffer_left;
cv::Mat depth_m_;

cv::Point point1;
cv::Point point2;

double codffs[5] = {
  -2.5034765682756088e-01,
      5.0579399202897619e-02,
      -7.0536676161976066e-04,
      -8.5255451307033846e-03,
      0.
};

//pthread_mutex_t mutex_data_;
sensor_msgs::CameraInfoPtr getCameraInfo(const Stream &stream) {
    sensor_msgs::CameraInfo *camera_info = new sensor_msgs::CameraInfo();
    camera_info_ptrs_[stream] = sensor_msgs::CameraInfoPtr(camera_info);
    camera_info->header.frame_id = "left_frame";
    camera_info->width = 640;
    camera_info->height = 400;

      camera_info->K.at(0) = 3.6220059643202876e+02;
      camera_info->K.at(2) = 4.0658699068023441e+02;
      camera_info->K.at(4) = 3.6350065250745848e+02;
      camera_info->K.at(5) = 2.3435161110061483e+02;
      camera_info->K.at(8) = 1;
      cv::Mat p = left_p_;
      for (int i = 0; i < p.rows; i++) {
        for (int j = 0; j < p.cols; j++) {
          camera_info->P.at(i * p.cols + j) = p.at<double>(i, j);
        }
      }

      camera_info->distortion_model = "plumb_bob";
    
      for (int i = 0; i < 5; i++) {
        camera_info->D.push_back(codffs[i]);
      }

    camera_info->R.at(0) = 1.0;
    camera_info->R.at(1) = 0.0;
    camera_info->R.at(2) = 0.0;
    camera_info->R.at(3) = 0.0;
    camera_info->R.at(4) = 1.0;
    camera_info->R.at(5) = 0.0;
    camera_info->R.at(6) = 0.0;
    camera_info->R.at(7) = 0.0;
    camera_info->R.at(8) = 1.0;

    return camera_info_ptrs_[stream];
  }
void publishCamera(
      const Stream &stream, const api::StreamData &data, std::uint32_t seq,
      ros::Time stamp) {
    // if (camera_publishers_[stream].getNumSubscribers() == 0)
    //   return;
    std_msgs::Header header_p;
    //header.seq = seq;
    header_p.stamp = stamp;
    header_p.frame_id = "left_data";
   // pthread_mutex_lock(&mutex_data_);
    cv::Mat img = data.frame;
    auto &&msg =
        cv_bridge::CvImage(header_p, enc::BGR8, img).toImageMsg();

   // pthread_mutex_unlock(&mutex_data_);
    auto &&info = getCameraInfo(stream);
    //auto info = getCameraInfo(stream);
    //sensor_msgs::CameraInfoPtr &&info = {0};
    info->header.stamp = msg->header.stamp;
    left_pub_.publish(msg, info);
  }




template <typename T>
int calcElment(const cv::Mat& depth,const cv::Point& point1,const cv::Point& point2)
{
  std::vector<int> depth_value;
  

  for(int i=point1.x;i<point2.x;i++)
  {
    for(int j=point1.y;j<point2.y;j++)
    { 
      if(depth.at<T>(j,i)>0 && depth.at<T>(j,i) < 5000)
      {
        depth_value.push_back(depth.at<T>(j,i));
      }
    }
  }
  return (std::accumulate(std::begin(depth_value),std::end(depth_value),0.0)/depth_value.size()); 
  
}

bool calcDis(mynt_bridge::calcDis::Request &req, mynt_bridge::calcDis::Response &res)
{
  point1.x = req.p1_x;
  point1.y = req.p1_y;
  point2.x = req.p2_x;
  point2.y = req.p2_y;

  res.avr_dis = calcElment<ushort>(depth_m_,point1,point2);

  ROS_INFO("request point 1 is %d %d",(int)point1.x,(int)point1.y);
  ROS_INFO("request point 2 is %d %d",(int)point2.x,(int)point2.y);

  ROS_INFO("Avr dis is %d",(int)res.avr_dis);

  return true;

}
bool reqImage(mynt_bridge::reqImage::Request &req,mynt_bridge::reqImage::Response &res)
{

    buffer_depth.frame.copyTo(depth_m_);
    publishCamera(Stream::LEFT,buffer_left,0,ros::Time::now());
    ROS_INFO("Published LEFT Image");

    res.result = true;

    return true;
}




int main(int argc, char *argv[]) {
  auto &&api = API::Create(argc, argv);
  if (!api) return 1;

  bool ok;
  auto &&requests = api->GetStreamRequests();
  //auto &&request = api->SelectStreamRequest(&ok);
  //if(!ok) return 1;
  //api->ConfigStreamRequest(request);

  api->SetOptionValue(Option::IR_CONTROL, 80);

  api->EnableStreamData(Stream::DEPTH);
  api->EnableStreamData(Stream::LEFT_RECTIFIED);
  api->EnableStreamData(Stream::DISPARITY_NORMALIZED);

  api->Start(Source::VIDEO_STREAMING);
  cv::namedWindow("depth");

  ros::init(argc,argv,"mynt_bridge");
  ros::NodeHandle nh;

  ros::ServiceServer service_image_req = nh.advertiseService("request_image",reqImage);
  ros::ServiceServer service_calc_depth = nh.advertiseService("calc_depth",calcDis);

  //pthread_mutex_init(&mutex_data_,nullptr);

  image_transport::ImageTransport it_(nh);
  left_pub_  = it_.advertiseCamera("/left_image",1);

  while (ros::ok()) { 
    
    api->WaitForStreams();

    //auto &&left_data = api->GetStreamData(Stream::LEFT);
    auto &&left_data = api->GetStreamData(Stream::LEFT_RECTIFIED);
    //auto &&right_data = api->GetStreamData(Stream::RIGHT);


    auto &&depth_data = api->GetStreamData(Stream::DEPTH);
    auto &&disp_data = api->GetStreamData(Stream::DISPARITY_NORMALIZED);
    
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "depth_image";


    if (!depth_data.frame.empty() && !left_data.frame.empty()) {
       
       buffer_depth = depth_data; 
       buffer_left = left_data;
       //cv::imshow("depth",left_data.frame);
      // printf("Depth is %d \n",buffer_depth.frame.at<ushort>(200,200));

    }
    if(!disp_data.frame.empty())
	{
		//printf("show \n");
		auto &&depth_frame = disp_data.frame;
		cv::applyColorMap(depth_frame,depth_frame,cv::COLORMAP_JET);
    cv::rectangle(depth_frame,point1,point2,cv::Scalar(0,255,0),1);
		cv::imshow("depth",depth_frame);
    cv::waitKey(5);
	}

    ros::spinOnce();
  }
  cv::destroyAllWindows();
  api->Stop(Source::VIDEO_STREAMING);
  return 0;
}
