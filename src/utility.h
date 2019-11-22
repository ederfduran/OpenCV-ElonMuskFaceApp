#ifndef UTILITY_H_
#define UTILITY_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;  
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }      

    r += "C";
    r += (chans+'0');

    return r;
}

void showGlassMasks(const cv::Mat& bgr,const cv::Mat& mask){
    cv::imshow(" glass_bgr ",bgr); 
    cv::imshow(" glass_mask ",mask);

    std::cout << "bgr type : "<< type2str(bgr.type()) << std::endl;
    std::cout << "mask type : "<< type2str( mask.type()) << std::endl;

}

void naiveImplementation(cv::Mat face_with_naive_glasses,cv::Mat glass_bgr,int h, int w){
    //face_with_naive_glasses*= 255;
    face_with_naive_glasses.convertTo(face_with_naive_glasses,CV_8UC3,255,0);
    // Replace the eye region with the sunglass image
    //glass_bgr*=255;
    glass_bgr.convertTo(glass_bgr,CV_8UC3,250,0);

    glass_bgr.copyTo(face_with_naive_glasses(cv::Range(150,150+h),cv::Range(140,140+ w)));

    cv::imshow("Naive Glasses",face_with_naive_glasses);
}

void maskAndBgr(cv::Mat& mask, cv::Mat& bgr,const cv::Mat& png){
    cv::Mat bgra_channel[4];
    cv::Mat bgr_channel[3];
    cv::split(png,bgra_channel);

    for(size_t i=0;i< 3;i++){
        bgr_channel[i]= bgra_channel[i];
    }
    cv::merge(bgr_channel,3,bgr);
    mask= bgra_channel[3];
}



cv::Mat getMaskedRegion(const cv::Mat& img,cv::Mat& mask,int width_start, int height_start,bool use_bitwise=false){
    cv::Mat result;
    int mask_width=mask.size().width;
    int mask_height=mask.size().height;
    // region of interest
    cv::Mat roi = img(cv::Range(height_start,height_start+mask_height),cv::Range(width_start,width_start+ mask_width));

    cv::Mat roi_channels[3];
    split(roi,roi_channels);
    cv::Mat masked_channels[3];

    cv::Mat mask_not;
    if(use_bitwise){
        cv::bitwise_not(mask,mask_not);
    }
    cv::Mat three_mask_c[]={mask,mask,mask};

    for (int i = 0; i < 3; i++)
    {
        // Use the mask to create the masked eye region
        if(use_bitwise){
            cv::bitwise_and(roi_channels[i],mask_not,masked_channels[i]);
        }else{
            cv::multiply(roi_channels[i], (1-three_mask_c[i]), masked_channels[i]);
        }
    }
    cv::merge(masked_channels,3,result);
    return result;
}


#endif