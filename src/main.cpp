#include <iostream>
#include "utility.h"


int main(){

    bool use_bitwise_op=false;
    // read face
    cv::Mat face= cv::imread("../img/musk.jpg");
    face.convertTo(face,CV_32FC3);
    face= face/255;

    // read glasses with alpha channel
    cv::Mat glass_png= cv::imread("../img/Glasses2.png",-1);
    glass_png.convertTo(glass_png,CV_32F);
    glass_png= glass_png/255;
    // resize the image
    cv::resize(glass_png,glass_png,cv::Size(),0.5,0.5);


    // read moustache with png
    cv::Mat moustache= cv::imread("../img/moustache.png",-1);
    moustache.convertTo(moustache,CV_32F);
    moustache= moustache/255;
    cv::resize(moustache,moustache,cv::Size(),0.5,0.5,2);
    int moustache_width= moustache.size().width;
    int moustache_height= moustache.size().height;

    cv::Mat moustache_channel[4];
    cv::Mat moustache_bgr[3];
    cv::split(moustache,moustache_channel);
    cv::Mat moustache_color,moustache_mask;
    for(size_t i=0;i<3;i++){
        moustache_bgr[i]= moustache_channel[i];
    }
    cv::merge(moustache_bgr,3,moustache_color);
    moustache_mask= moustache_channel[3];
    
    std::cout << "moustache Size : " << moustache.size()<< std::endl;

    std::cout << "Glasses Size : " << glass_png.size()<< std::endl;
    std::cout << "Number of Channels: " << glass_png.channels() << std::endl;

    // prepare the glasses Image

    cv::Mat glass_bgr,glass_mask;
    maskAndBgr(glass_mask,glass_bgr,glass_png);
    
    showGlassMasks(glass_bgr,glass_mask);

    int glass_height= glass_mask.size().height;
    int glass_width= glass_mask.size().width;

    naiveImplementation(face.clone(),glass_bgr.clone(),glass_height,glass_width);

    // Make the dimensions of the mask same as the input image.
    // Since Face Image is a 3-channel image, we create a 3 channel image for the mask
    cv::Mat glassMask;
    cv::Mat glass_mask_channels[] = {glass_mask,glass_mask,glass_mask};
    merge(glass_mask_channels,3,glassMask);

    // Make a copy
    cv::Mat faceWithGlassesArithmetic = face.clone();
    
    // Get the eye region from the face image              Heigh                                Width
    cv::Mat eye_roi = faceWithGlassesArithmetic(cv::Range(150,150+glass_height),cv::Range(140,140+ glass_width));
    // Get mouth region from face image
    cv::Mat mouth_roi= faceWithGlassesArithmetic(cv::Range(210,210+moustache_height),cv::Range(160,160+ moustache_width));
    
    cv::Mat mouth_roi_channel[3];
    cv::split(mouth_roi,mouth_roi_channel);
    cv::Mat moustache_t_mask;
    cv::Mat moustache_mask_channel[]={moustache_mask,moustache_mask,moustache_mask};
    cv::merge(moustache_mask_channel,3,moustache_t_mask);

    cv::Mat masked_mouth_channel[3];

    for(size_t i=0;i<3;i++){
        cv::multiply(mouth_roi_channel[i],1-moustache_mask_channel[i],masked_mouth_channel[i]);
    }

    cv::Mat masked_mouth;
    cv::merge(masked_mouth_channel,3,masked_mouth);

    cv::Mat mouth_moustache_final;
    cv::Mat masked_moustache;
    cv::multiply(moustache_color,moustache_t_mask,masked_moustache);
    cv::add(masked_mouth,masked_moustache,mouth_moustache_final);

    cv::Mat eye_roi_channels[3];
    split(eye_roi,eye_roi_channels);
    cv::Mat masked_eye_channels[3];
    
    
    cv::Mat masked_eye = getMaskedRegion(face.clone(),glass_mask,140,150,use_bitwise_op);

    cv::Mat masked_glass;
    cv::Mat eye_roi_final;
    // Use the mask to create the masked sunglass region
    if(use_bitwise_op){
        cv::bitwise_and(glass_bgr, glassMask, masked_glass);
        cv::bitwise_or(masked_eye, masked_glass, eye_roi_final);
    }else{
        cv::multiply(glass_bgr, glassMask, masked_glass);
        cv::add(masked_eye, masked_glass, eye_roi_final);
    }
    cv::multiply(masked_glass,eye_roi,masked_glass);
    
    
    masked_eye.convertTo(masked_eye,CV_8UC3,255,0);
    masked_glass.convertTo(masked_glass,CV_8UC3,255,0);
    eye_roi_final.convertTo(eye_roi_final,CV_8UC3,255,0);
    mouth_moustache_final.convertTo(mouth_moustache_final,CV_8UC3,255,0);
    face.convertTo(face,CV_8UC3,255,0);
    

    cv::imshow("Masked eye region",masked_eye);
    cv::imshow("Masked Sun glass region",masked_glass);
    cv::imshow("Masked eye with glass region",eye_roi_final);
    cv::imshow("Masked mouth",mouth_moustache_final);

    mouth_moustache_final.copyTo(face(cv::Range(210,210+moustache_height),cv::Range(160,160+ moustache_width)));
    eye_roi_final.copyTo(face(cv::Range(150,150+glass_height),cv::Range(140,140+ glass_width)));
    
    //cv::imshow("Moustache",moustache_color);
    
    cv::imshow("Moustache",moustache_color);
    
    cv::imshow("Final Image",face);

    cv::waitKey(0);

    return 0;
}