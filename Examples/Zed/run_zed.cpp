#include <opencv2/opencv.hpp>
#include <zed/Camera.hpp>
#include <iostream>
#include "System.h"

using namespace std;

int main( int argc, char** argv )
{
    if (argc != 3)
    {
        cout<<"Usage: run_zed path_to_vocabulary path_to_config_file"<<endl;
        return 0;
    }
    sl::zed::SENSING_MODE dm_type = sl::zed::RAW;
    sl::zed::Camera* zed;

    zed = new sl::zed::Camera( sl::zed::VGA );

    int width = zed->getImageSize().width;
    int height = zed->getImageSize().height;

    cout<<"image width="<<width<<", height = "<<height<<endl;
    sl::zed::ERRCODE err = zed->init( sl::zed::MODE::PERFORMANCE, 0, true );

    cout<<sl::zed::errcode2str(err)<<endl;

    if (err!=sl::zed::SUCCESS)
    {
        delete zed;
        return 1;
    }

    cv::Mat depth(height, width, CV_8UC4);
    cv::Mat color(height, width, CV_8UC4);
    //cv::Mat confidencemap(height, width, CV_8UC4);

    int confidence = 100;
    // init orb slam
    ORB_SLAM2::System orbslam( argv[1], argv[2], ORB_SLAM2::System::RGBD, true );

    cout<<"start to grab images, press q to quit."<<endl;
    double index = 0;
    while (1)
    {
        zed->setConfidenceThreshold( confidence );
        bool res = zed->grab( dm_type );
        if (!res)
        {
            slMat2cvMat( zed->normalizeMeasure(sl::zed::MEASURE::DEPTH)).copyTo(depth);
            slMat2cvMat(zed->retrieveImage(sl::zed::LEFT)).copyTo(color);

            orbslam.TrackRGBD( color, depth, index );
            //cv::imshow( "color", color );
            //cv::imshow( "depth", depth );

        }
        if (cv::waitKey(5) == 'q')
        {
            break;
        }

        index += 0.1;
    }

    orbslam.Shutdown();
    delete zed;
    return 0;
}

