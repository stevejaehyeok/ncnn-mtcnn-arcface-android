// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <math.h>

// ncnn
#include "net.h"
#include "benchmark.h"

#include "squeezenet_v1.1.id.h"

using namespace std;

struct Bbox {
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    float ppoint[10];
    float regreCoord[4];
};

static unsigned long get_current_time(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (tv.tv_sec*1000000 + tv.tv_usec);
}

bool cmpScore(Bbox lsh, Bbox rsh) {
    if (lsh.score < rsh.score)
        return true;
    else
        return false;
}

bool cmpArea(Bbox lsh, Bbox rsh) {
    if (lsh.area < rsh.area)
        return false;
    else
        return true;
}


class MTCNN {
public:
    MTCNN(const string &model_path);
    MTCNN(const std::vector<std::string> param_files, const std::vector<std::string> bin_files);
    ~MTCNN();

    void SetMinFace(int minSize);
    void SetNumThreads(int numThreads);
    void SetTimeCount(int timeCount);

    void detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox);
    void detectMaxFace(ncnn::Mat& img_, std::vector<Bbox>& finalBbox);
    //  void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
private:
    void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, float scale);
    void nmsTwoBoxs(vector<Bbox> &boundingBox_, vector<Bbox> &previousBox_, const float overlap_threshold, string modelname = "Union");
    void nms(vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname="Union");
    void refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square);
    void extractMaxFace(vector<Bbox> &boundingBox_);

    void PNet(float scale);
    void PNet();
    void RNet();
    void ONet();
    ncnn::Net Pnet, Rnet, Onet;
    ncnn::Mat img;
    const float nms_threshold[3] = {0.5f, 0.7f, 0.7f};

    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
    const int MIN_DET_SIZE = 12;
    std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    std::vector<Bbox> firstPreviousBbox_, secondPreviousBbox_, thirdPrevioussBbox_;
    int img_w, img_h;

private://部分可调参数
    const float threshold[3] = { 0.8f, 0.8f, 0.6f };
    int minsize = 40;
    const float pre_facetor = 0.709f;

    int count = 10;
    int num_threads = 4;
};

MTCNN::MTCNN(const string &model_path) {

    std::vector<std::string> param_files = {
            model_path+"/det1.param",
            model_path+"/det2.param",
            model_path+"/det3.param"
    };

    std::vector<std::string> bin_files = {
            model_path+"/det1.bin",
            model_path+"/det2.bin",
            model_path+"/det3.bin"
    };

    Pnet.load_param(param_files[0].data());
    Pnet.load_model(bin_files[0].data());
    Rnet.load_param(param_files[1].data());
    Rnet.load_model(bin_files[1].data());
    Onet.load_param(param_files[2].data());
    Onet.load_model(bin_files[2].data());
}

MTCNN::MTCNN(const std::vector<std::string> param_files, const std::vector<std::string> bin_files){
    Pnet.load_param(param_files[0].data());
    Pnet.load_model(bin_files[0].data());
    Rnet.load_param(param_files[1].data());
    Rnet.load_model(bin_files[1].data());
    Onet.load_param(param_files[2].data());
    Onet.load_model(bin_files[2].data());
}


MTCNN::~MTCNN(){
    Pnet.clear();
    Rnet.clear();
    Onet.clear();
}

void MTCNN::SetMinFace(int minSize){
    minsize = minSize;
}

void MTCNN::SetNumThreads(int numThreads){
    num_threads = numThreads;
}

void MTCNN::SetTimeCount(int timeCount) {
    count = timeCount;
}


void MTCNN::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, float scale){
    const int stride = 2;
    const int cellsize = 12;
    //score p
    float *p = score.channel(1);//score.data + score.cstep;
    //float *plocal = location.data;
    Bbox bbox;
    float inv_scale = 1.0f/scale;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                bbox.score = *p;
                bbox.x1 = round((stride*col+1)*inv_scale);
                bbox.y1 = round((stride*row+1)*inv_scale);
                bbox.x2 = round((stride*col+1+cellsize)*inv_scale);
                bbox.y2 = round((stride*row+1+cellsize)*inv_scale);
                bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
                const int index = row * score.w + col;
                for(int channel=0;channel<4;channel++){
                    bbox.regreCoord[channel]=location.channel(channel)[index];
                }
                boundingBox_.push_back(bbox);
            }
            p++;
            //plocal++;
        }
    }
}


void MTCNN::nmsTwoBoxs(vector<Bbox>& boundingBox_, vector<Bbox>& previousBox_, const float overlap_threshold, string modelname)
{
    if (boundingBox_.empty()) {
        return;
    }
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    //std::cout << boundingBox_.size() << " ";
    for (std::vector<Bbox>::iterator ity = previousBox_.begin(); ity != previousBox_.end(); ity++) {
        for (std::vector<Bbox>::iterator itx = boundingBox_.begin(); itx != boundingBox_.end();) {
            int i = itx - boundingBox_.begin();
            int j = ity - previousBox_.begin();
            maxX = std::max(boundingBox_.at(i).x1, previousBox_.at(j).x1);
            maxY = std::max(boundingBox_.at(i).y1, previousBox_.at(j).y1);
            minX = std::min(boundingBox_.at(i).x2, previousBox_.at(j).x2);
            minY = std::min(boundingBox_.at(i).y2, previousBox_.at(j).y2);
            //maxX1 and maxY1 reuse
            maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
            //IOU reuse for the area of two bbox
            IOU = maxX * maxY;
            if (!modelname.compare("Union"))
                IOU = IOU / (boundingBox_.at(i).area + previousBox_.at(j).area - IOU);
            else if (!modelname.compare("Min")) {
                IOU = IOU / ((boundingBox_.at(i).area < previousBox_.at(j).area) ? boundingBox_.at(i).area : previousBox_.at(j).area);
            }
            if (IOU > overlap_threshold&&boundingBox_.at(i).score>previousBox_.at(j).score) {
                //if (IOU > overlap_threshold) {
                itx = boundingBox_.erase(itx);
            }
            else {
                itx++;
            }
        }
    }
    //std::cout << boundingBox_.size() << std::endl;
}

void MTCNN::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname){
    if(boundingBox_.empty()){
        return;
    }
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    std::vector<int> vPick;
    int nPick = 0;
    std::multimap<float, int> vScores;
    const int num_boxes = boundingBox_.size();
    vPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i){
        vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
    }
    while(vScores.size() > 0){
        int last = vScores.rbegin()->second;
        vPick[nPick] = last;
        nPick += 1;
        for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();){
            int it_idx = it->second;
            maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //maxX1 and maxY1 reuse
            maxX = ((minX-maxX+1)>0)? (minX-maxX+1) : 0;
            maxY = ((minY-maxY+1)>0)? (minY-maxY+1) : 0;
            //IOU reuse for the area of two bbox
            IOU = maxX * maxY;
            if(!modelname.compare("Union"))
                IOU = IOU/(boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
            else if(!modelname.compare("Min")){
                IOU = IOU/((boundingBox_.at(it_idx).area < boundingBox_.at(last).area)? boundingBox_.at(it_idx).area : boundingBox_.at(last).area);
            }
            if(IOU > overlap_threshold){
                it = vScores.erase(it);
            }else{
                it++;
            }
        }
    }

    vPick.resize(nPick);
    std::vector<Bbox> tmp_;
    tmp_.resize(nPick);
    for(int i = 0; i < nPick; i++){
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}
void MTCNN::refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square){
    if(vecBbox.empty()){
        cout<<"Bbox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        bbw = (*it).x2 - (*it).x1 + 1;
        bbh = (*it).y2 - (*it).y1 + 1;
        x1 = (*it).x1 + (*it).regreCoord[0]*bbw;
        y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
        x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
        y2 = (*it).y2 + (*it).regreCoord[3]*bbh;



        if(square){
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            maxSide = (h>w)?h:w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);
        }

        //boundary check
        if((*it).x1<0)(*it).x1=0;
        if((*it).y1<0)(*it).y1=0;
        if((*it).x2>width)(*it).x2 = width - 1;
        if((*it).y2>height)(*it).y2 = height - 1;

        it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
    }
}

void MTCNN::extractMaxFace(vector<Bbox>& boundingBox_)
{
    if (boundingBox_.empty()) {
        return;
    }
    sort(boundingBox_.begin(), boundingBox_.end(), cmpArea);
    for (std::vector<Bbox>::iterator itx = boundingBox_.begin() + 1; itx != boundingBox_.end();) {
        itx = boundingBox_.erase(itx);
    }
}

void MTCNN::PNet(float scale)
{
    //first stage
    int hs = (int)ceil(img_h*scale);
    int ws = (int)ceil(img_w*scale);
    ncnn::Mat in;
    resize_bilinear(img, in, ws, hs);
    ncnn::Extractor ex = Pnet.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(num_threads);
    ex.input("data", in);
    ncnn::Mat score_, location_;
    ex.extract("prob1", score_);
    ex.extract("conv4-2", location_);
    std::vector<Bbox> boundingBox_;

    generateBbox(score_, location_, boundingBox_, scale);
    nms(boundingBox_, nms_threshold[0]);

    firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
    boundingBox_.clear();
}


void MTCNN::PNet(){
    firstBbox_.clear();
    float minl = img_w < img_h? img_w: img_h;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = pre_facetor;
    vector<float> scales_;
    while(minl>MIN_DET_SIZE){
        scales_.push_back(m);
        minl *= factor;
        m = m*factor;
    }
    for (size_t i = 0; i < scales_.size(); i++) {
        int hs = (int)ceil(img_h*scales_[i]);
        int ws = (int)ceil(img_w*scales_[i]);
        ncnn::Mat in;
        resize_bilinear(img, in, ws, hs);
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_num_threads(num_threads);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
        ex.extract("conv4-2", location_);
        std::vector<Bbox> boundingBox_;
        generateBbox(score_, location_, boundingBox_, scales_[i]);
        nms(boundingBox_, nms_threshold[0]);
        firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
        boundingBox_.clear();
    }
}
void MTCNN::RNet(){
    secondBbox_.clear();
    int count = 0;
    for(vector<Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 24, 24);
        ncnn::Extractor ex = Rnet.create_extractor();
        ex.set_num_threads(num_threads);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox;
        ex.extract("prob1", score);
        ex.extract("conv5-2", bbox);
        if((float)score[1] > threshold[1]){
            for(int channel=0;channel<4;channel++){
                it->regreCoord[channel]=(float)bbox[channel];//*(bbox.data+channel*bbox.cstep);
            }
            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
            it->score = score.channel(1)[0];//*(score.data+score.cstep);
            secondBbox_.push_back(*it);
        }
    }
}
void MTCNN::ONet(){
    thirdBbox_.clear();
    for(vector<Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 48, 48);
        ncnn::Extractor ex = Onet.create_extractor();
        ex.set_num_threads(num_threads);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox, keyPoint;
        ex.extract("prob1", score);
        ex.extract("conv6-2", bbox);
        ex.extract("conv6-3", keyPoint);
        if((float)score[1] > threshold[2]){
            for(int channel = 0; channel < 4; channel++){
                it->regreCoord[channel]=(float)bbox[channel];
            }
            it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
            it->score = score.channel(1)[0];
            for(int num=0;num<5;num++){
                (it->ppoint)[num] = it->x1 + (it->x2 - it->x1) * keyPoint[num];
                (it->ppoint)[num+5] = it->y1 + (it->y2 - it->y1) * keyPoint[num+5];
            }

            thirdBbox_.push_back(*it);
        }
    }
}

#define TIMEOPEN 0 //设置是否开关调试，1为开，其它为关

void MTCNN::detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox_){
    img = img_;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);

#if(TIMEOPEN==1)
    double total_time = 0.;
    double min_time = DBL_MAX;
    double max_time = 0.0;
    double temp_time = 0.0;
    unsigned long time_0, time_1;

    for(int i =0 ;i < count; i++) {
        time_0 = get_current_time();
#endif

        PNet();
        //the first stage's nms
        if(firstBbox_.size() < 1) return;
        nms(firstBbox_, nms_threshold[0]);
        refine(firstBbox_, img_h, img_w, true);
        printf("firstBbox_.size()=%d\n", firstBbox_.size());
        //second stage
        RNet();
        printf("secondBbox_.size()=%d\n", secondBbox_.size());
        if (secondBbox_.size() < 1) return;
        nms(secondBbox_, nms_threshold[1]);
        refine(secondBbox_, img_h, img_w, true);

        //third stage
        ONet();
        printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
        if (thirdBbox_.size() < 1) return;
        refine(thirdBbox_, img_h, img_w, true);
        nms(thirdBbox_, nms_threshold[2], "Min");
        finalBbox_ = thirdBbox_;

#if(TIMEOPEN==1)
        time_1 = get_current_time();
        temp_time = ((time_1 - time_0)/1000.0);
        if(temp_time < min_time)
        {
            min_time = temp_time;
        }
        if(temp_time > max_time)
        {
            max_time = temp_time;
        }
        total_time += temp_time;

        //LOGD("iter %d/%d cost: %.3f ms\n", i+1, count, temp_time);
    }
    //LOGD("Time cost:Max %.2fms,Min %.2fms,Avg %.2fms\n", max_time,min_time,total_time/count);
#endif

}


void MTCNN::detectMaxFace(ncnn::Mat& img_, std::vector<Bbox>& finalBbox) {
    firstPreviousBbox_.clear();
    secondPreviousBbox_.clear();
    thirdPrevioussBbox_.clear();
    firstBbox_.clear();
    secondBbox_.clear();
    thirdBbox_.clear();

    //norm
    img = img_;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);

#if(TIMEOPEN==1)
    double total_time = 0.;
    double min_time = DBL_MAX;
    double max_time = 0.0;
    double temp_time = 0.0;
    unsigned long time_0, time_1;

    for(int i =0 ;i < count; i++) {
        time_0 = get_current_time();
#endif

        //pyramid size
        float minl = img_w < img_h ? img_w : img_h;
        float m = (float)MIN_DET_SIZE / minsize;
        minl *= m;
        float factor = pre_facetor;
        vector<float> scales_;
        while (minl>MIN_DET_SIZE) {
            scales_.push_back(m);
            minl *= factor;
            m = m*factor;
        }
        sort(scales_.begin(), scales_.end());
        //printf("scales_.size()=%d\n", scales_.size());

        //Change the sampling process.
        for (size_t i = 0; i < scales_.size(); i++)
        {
            //first stage
            PNet(scales_[i]);
            nms(firstBbox_, nms_threshold[0]);
            nmsTwoBoxs(firstBbox_, firstPreviousBbox_, nms_threshold[0]);
            if (firstBbox_.size() < 1) {
                firstBbox_.clear();
                continue;
            }
            firstPreviousBbox_.insert(firstPreviousBbox_.end(), firstBbox_.begin(), firstBbox_.end());
            refine(firstBbox_, img_h, img_w, true);
            //printf("firstBbox_.size()=%d\n", firstBbox_.size());

            //second stage
            RNet();
            nms(secondBbox_, nms_threshold[1]);
            nmsTwoBoxs(secondBbox_, secondPreviousBbox_, nms_threshold[0]);
            secondPreviousBbox_.insert(secondPreviousBbox_.end(), secondBbox_.begin(), secondBbox_.end());
            if (secondBbox_.size() < 1) {
                firstBbox_.clear();
                secondBbox_.clear();
                continue;
            }
            refine(secondBbox_, img_h, img_w, true);
            //printf("secondBbox_.size()=%d\n", secondBbox_.size());

            //third stage
            ONet();
            //printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
            if (thirdBbox_.size() < 1) {
                firstBbox_.clear();
                secondBbox_.clear();
                thirdBbox_.clear();
                continue;
            }
            refine(thirdBbox_, img_h, img_w, true);
            nms(thirdBbox_, nms_threshold[2], "Min");

            if (thirdBbox_.size() > 0) {
                extractMaxFace(thirdBbox_);
                finalBbox = thirdBbox_;//if largest face size is similar,.
                break;
            }
        }

        //printf("firstPreviousBbox_.size()=%d\n", firstPreviousBbox_.size());
        //printf("secondPreviousBbox_.size()=%d\n", secondPreviousBbox_.size());

#if(TIMEOPEN==1)
        time_1 = get_current_time();
        temp_time = ((time_1 - time_0)/1000.0);
        if(temp_time < min_time)
        {
            min_time = temp_time;
        }
        if(temp_time > max_time)
        {
            max_time = temp_time;
        }
        total_time += temp_time;

        //LOGD("iter %d/%d cost: %.3f ms\n", i+1, count, temp_time);
    }
    //LOGD("Time cost:Max %.2fms,Min %.2fms,Avg %.2fms\n", max_time,min_time,total_time/count);
#endif
}

static MTCNN *mtcnn;

bool detection_sdk_init_ok = false;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static std::vector<std::string> squeezenet_words;
static ncnn::Net squeezenet;

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

extern "C" {

    JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnLoad");

        ncnn::create_gpu_instance();

        return JNI_VERSION_1_4;
    }

    JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnUnload");

        ncnn::destroy_gpu_instance();
    }

    /*// public native boolean Init(AssetManager mgr);
    JNIEXPORT jboolean JNICALL Java_com_tencent_squeezencnn_SqueezeNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
    {
        ncnn::Option opt;
        opt.lightmode = true;
        opt.num_threads = 4;
        opt.blob_allocator = &g_blob_pool_allocator;
        opt.workspace_allocator = &g_workspace_pool_allocator;

        // use vulkan compute
        if (ncnn::get_gpu_count() != 0)
            opt.use_vulkan_compute = true;

        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

        squeezenet.opt = opt;

        // init param
        {
            int ret = squeezenet.load_param_bin(mgr, "squeezenet_v1.1.param.bin");
            if (ret != 0)
            {
                __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_param_bin failed");
                return JNI_FALSE;
            }
        }

        // init bin
        {
            int ret = squeezenet.load_model(mgr, "squeezenet_v1.1.bin");
            if (ret != 0)
            {
                __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_model failed");
                return JNI_FALSE;
            }
        }

        // init words
        {
            AAsset* asset = AAssetManager_open(mgr, "synset_words.txt", AASSET_MODE_BUFFER);
            if (!asset)
            {
                __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "open synset_words.txt failed");
                return JNI_FALSE;
            }

            int len = AAsset_getLength(asset);

            std::string words_buffer;
            words_buffer.resize(len);
            int ret = AAsset_read(asset, (void*)words_buffer.data(), len);

            AAsset_close(asset);

            if (ret != len)
            {
                __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "read synset_words.txt failed");
                return JNI_FALSE;
            }

            squeezenet_words = split_string(words_buffer, "\n");
        }

        return JNI_TRUE;
    }

    // public native String Detect(Bitmap bitmap, boolean use_gpu);
    JNIEXPORT jstring JNICALL Java_com_tencent_squeezencnn_SqueezeNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
    {
        if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
        {
            return env->NewStringUTF("no vulkan capable gpu");
        }

        double start_time = ncnn::get_current_time();

        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
        int width = info.width;
        int height = info.height;
        if (width != 227 || height != 227)
            return NULL;
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        // ncnn from bitmap
        ncnn::Mat in = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_BGR);

        // squeezenet
        std::vector<float> cls_scores;
        {
            const float mean_vals[3] = {104.f, 117.f, 123.f};
            in.substract_mean_normalize(mean_vals, 0);

            ncnn::Extractor ex = squeezenet.create_extractor();

            ex.set_vulkan_compute(use_gpu);

            ex.input(squeezenet_v1_1_param_id::BLOB_data, in);

            ncnn::Mat out;
            ex.extract(squeezenet_v1_1_param_id::BLOB_prob, out);

            cls_scores.resize(out.w);
            for (int j=0; j<out.w; j++)
            {
                cls_scores[j] = out[j];
            }
        }

        // return top class
        int top_class = 0;
        float max_score = 0.f;
        for (size_t i=0; i<cls_scores.size(); i++)
        {
            float s = cls_scores[i];
    //         __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%d %f", i, s);
            if (s > max_score)
            {
                top_class = i;
                max_score = s;
            }
        }

        const std::string& word = squeezenet_words[top_class];
        char tmp[32];
        sprintf(tmp, "%.3f", max_score);
        std::string result_str = std::string(word.c_str() + 10) + " = " + tmp;

        // +10 to skip leading n03179701
        jstring result = env->NewStringUTF(result_str.c_str());

        double elasped = ncnn::get_current_time() - start_time;
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%.2fms   detect", elasped);

        return result;
    }*/

    JNIEXPORT jstring JNICALL Java_com_tencent_squeezencnn_SqueezeNcnn_Add(JNIEnv* env, jobject thiz)
    {
        return env->NewStringUTF("Hello :)");
    }

    JNIEXPORT jboolean JNICALL
    Java_com_tencent_squeezencnn_MTCNN_FaceDetectionModelInit(JNIEnv *env, jobject instance,
                                                    jstring faceDetectionModelPath_) {
        ////LOGD("JNI开始人脸检测模型初始化");
        //如果已初始化则直接返回
        if (detection_sdk_init_ok) {
            //  //LOGD("人脸检测模型已经导入");
            return true;
        }
        jboolean tRet = false;
        if (NULL == faceDetectionModelPath_) {
            //   //LOGD("导入的人脸检测的目录为空");
            return tRet;
        }

        //获取MTCNN模型的绝对路径的目录（不是/aaa/bbb.bin这样的路径，是/aaa/)
        const char *faceDetectionModelPath = env->GetStringUTFChars(faceDetectionModelPath_, 0);
        if (NULL == faceDetectionModelPath) {
            return tRet;
        }

        string tFaceModelDir = faceDetectionModelPath;
        string tLastChar = tFaceModelDir.substr(tFaceModelDir.length() - 1, 1);
        ////LOGD("init, tFaceModelDir last =%s", tLastChar.c_str());
        //目录补齐/
        if ("\\" == tLastChar) {
            tFaceModelDir = tFaceModelDir.substr(0, tFaceModelDir.length() - 1) + "/";
        } else if (tLastChar != "/") {
            tFaceModelDir += "/";
        }
        ////LOGD("init, tFaceModelDir=%s", tFaceModelDir.c_str());

        //没判断是否正确导入，懒得改了
        mtcnn = new MTCNN(tFaceModelDir);
        mtcnn->SetMinFace(40);

        env->ReleaseStringUTFChars(faceDetectionModelPath_, faceDetectionModelPath);
        detection_sdk_init_ok = true;
        tRet = true;
        return tRet;
    }

    JNIEXPORT jintArray JNICALL
    Java_com_tencent_squeezencnn_MTCNN_FaceDetect(JNIEnv *env, jobject instance, jbyteArray imageDate_,
                                        jint imageWidth, jint imageHeight, jint imageChannel) {
        //  //LOGD("JNI开始检测人脸");
        if(!detection_sdk_init_ok){
            //LOGD("人脸检测MTCNN模型SDK未初始化，直接返回空");
            return NULL;
        }

        int tImageDateLen = env->GetArrayLength(imageDate_);
        if(imageChannel == tImageDateLen / imageWidth / imageHeight){
            //LOGD("数据宽=%d,高=%d,通道=%d",imageWidth,imageHeight,imageChannel);
        }
        else{
            //LOGD("数据长宽高通道不匹配，直接返回空");
            return NULL;
        }

        jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
        if (NULL == imageDate){
            //LOGD("导入数据为空，直接返回空");
            env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
            return NULL;
        }

        if(imageWidth<20||imageHeight<20){
            //LOGD("导入数据的宽和高小于20，直接返回空");
            env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
            return NULL;
        }

        //TODO 通道需测试
        if(3 == imageChannel || 4 == imageChannel){
            //图像通道数只能是3或4；
        }else{
            //LOGD("图像通道数只能是3或4，直接返回空");
            env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
            return NULL;
        }

        //int32_t minFaceSize=40;
        //mtcnn->SetMinFace(minFaceSize);

        unsigned char *faceImageCharDate = (unsigned char*)imageDate;
        ncnn::Mat ncnn_img;
        if(imageChannel==3) {
            ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_BGR2RGB,
                                              imageWidth, imageHeight);
        }else{
            ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_RGBA2RGB, imageWidth, imageHeight);
        }

        std::vector<Bbox> finalBbox;
        mtcnn->detect(ncnn_img, finalBbox);

        int32_t num_face = static_cast<int32_t>(finalBbox.size());
        //LOGD("检测到的人脸数目：%d\n", num_face);

        int out_size = 1+num_face*14;
        //  //LOGD("内部人脸检测完成,开始导出数据");
        int *faceInfo = new int[out_size];
        faceInfo[0] = num_face;
        for(int i=0;i<num_face;i++){
            faceInfo[14*i+1] = finalBbox[i].x1;//left
            faceInfo[14*i+2] = finalBbox[i].y1;//top
            faceInfo[14*i+3] = finalBbox[i].x2;//right
            faceInfo[14*i+4] = finalBbox[i].y2;//bottom
            for (int j =0;j<10;j++){
                faceInfo[14*i+5+j]=static_cast<int>(finalBbox[i].ppoint[j]);
            }
        }

        jintArray tFaceInfo = env->NewIntArray(out_size);
        env->SetIntArrayRegion(tFaceInfo,0,out_size,faceInfo);
        //  //LOGD("内部人脸检测完成,导出数据成功");
        delete[] faceInfo;
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return tFaceInfo;
    }

    JNIEXPORT jintArray JNICALL
    Java_com_tencent_squeezencnn_MTCNN_MaxFaceDetect(JNIEnv *env, jobject instance, jbyteArray imageDate_,
                                           jint imageWidth, jint imageHeight, jint imageChannel) {
        //  //LOGD("JNI开始检测人脸");
        if(!detection_sdk_init_ok){
            //LOGD("人脸检测MTCNN模型SDK未初始化，直接返回空");
            return NULL;
        }

        int tImageDateLen = env->GetArrayLength(imageDate_);
        if(imageChannel == tImageDateLen / imageWidth / imageHeight){
            //LOGD("数据宽=%d,高=%d,通道=%d",imageWidth,imageHeight,imageChannel);
        }
        else{
            //LOGD("数据长宽高通道不匹配，直接返回空");
            return NULL;
        }

        jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
        if (NULL == imageDate){
            //LOGD("导入数据为空，直接返回空");
            env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
            return NULL;
        }

        if(imageWidth<20||imageHeight<20){
            //LOGD("导入数据的宽和高小于20，直接返回空");
            env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
            return NULL;
        }

        //TODO 通道需测试
        if(3 == imageChannel || 4 == imageChannel){
            //图像通道数只能是3或4；
        }else{
            //LOGD("图像通道数只能是3或4，直接返回空");
            env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
            return NULL;
        }

        //int32_t minFaceSize=40;
        //mtcnn->SetMinFace(minFaceSize);

        unsigned char *faceImageCharDate = (unsigned char*)imageDate;
        ncnn::Mat ncnn_img;
        if(imageChannel==3) {
            ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_BGR2RGB,
                                              imageWidth, imageHeight);
        }else{
            ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_RGBA2RGB, imageWidth, imageHeight);
        }

        std::vector<Bbox> finalBbox;
        mtcnn->detectMaxFace(ncnn_img, finalBbox);

        int32_t num_face = static_cast<int32_t>(finalBbox.size());
        //LOGD("检测到的人脸数目：%d\n", num_face);

        int out_size = 1+num_face*14;
        //  //LOGD("内部人脸检测完成,开始导出数据");
        int *faceInfo = new int[out_size];
        faceInfo[0] = num_face;
        for(int i=0;i<num_face;i++){
            faceInfo[14*i+1] = finalBbox[i].x1;//left
            faceInfo[14*i+2] = finalBbox[i].y1;//top
            faceInfo[14*i+3] = finalBbox[i].x2;//right
            faceInfo[14*i+4] = finalBbox[i].y2;//bottom
            for (int j =0;j<10;j++){
                faceInfo[14*i+5+j]=static_cast<int>(finalBbox[i].ppoint[j]);
            }
        }

        jintArray tFaceInfo = env->NewIntArray(out_size);
        env->SetIntArrayRegion(tFaceInfo,0,out_size,faceInfo);
        //  //LOGD("内部人脸检测完成,导出数据成功");
        delete[] faceInfo;
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return tFaceInfo;
    }


    JNIEXPORT jboolean JNICALL
    Java_com_tencent_squeezencnn_MTCNN_FaceDetectionModelUnInit(JNIEnv *env, jobject instance) {
        if(!detection_sdk_init_ok){
            //LOGD("人脸检测MTCNN模型已经释放过或者未初始化");
            return true;
        }
        jboolean tDetectionUnInit = false;
        delete mtcnn;


        detection_sdk_init_ok=false;
        tDetectionUnInit = true;
        //LOGD("人脸检测初始化锁，重新置零");
        return tDetectionUnInit;

    }


    JNIEXPORT jboolean JNICALL
    Java_com_tencent_squeezencnn_MTCNN_SetMinFaceSize(JNIEnv *env, jobject instance, jint minSize) {
        if(!detection_sdk_init_ok){
            //LOGD("人脸检测MTCNN模型SDK未初始化，直接返回");
            return false;
        }

        if(minSize<=20){
            minSize=20;
        }

        mtcnn->SetMinFace(minSize);
        return true;
    }


    JNIEXPORT jboolean JNICALL
    Java_com_tencent_squeezencnn_MTCNN_SetThreadsNumber(JNIEnv *env, jobject instance, jint threadsNumber) {
        if(!detection_sdk_init_ok){
            //LOGD("人脸检测MTCNN模型SDK未初始化，直接返回");
            return false;
        }

        if(threadsNumber!=1&&threadsNumber!=2&&threadsNumber!=4&&threadsNumber!=8){
            //LOGD("线程只能设置1，2，4，8");
            return false;
        }

        mtcnn->SetNumThreads(threadsNumber);
        return  true;
    }


    JNIEXPORT jboolean JNICALL
    Java_com_tencent_squeezencnn_MTCNN_SetTimeCount(JNIEnv *env, jobject instance, jint timeCount) {

        if(!detection_sdk_init_ok){
            //LOGD("人脸检测MTCNN模型SDK未初始化，直接返回");
            return false;
        }

        mtcnn->SetTimeCount(timeCount);
        return true;

    }

}
