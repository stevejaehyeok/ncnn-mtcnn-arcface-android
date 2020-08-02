#pragma once
#ifndef BASE_H
#define BASE_H
#include <cmath>
#include <cstring>
#include "net.h"

struct Bbox
{
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	float area;
	float ppoint[10]; // xxxxxyyyyy
	float regreCoord[4];
};

struct FaceInfo {
	float score;
	int x[2];
	int y[2];
	float area;
	float regreCoord[4];
	int landmark[10]; // xyxyxyxyxy
	FaceInfo& operator = (const Bbox& bbox) {
		this->score = bbox.score;
		this->area = bbox.area;
		this->x[0] = bbox.x1, this->x[1] = bbox.x2, this->y[0] = bbox.y1, this->y[1] = bbox.y2;
		for (int i = 0; i < 4; i++)
			this->regreCoord[i] = bbox.regreCoord[i];
		for (int i = 0; i < 5; i++)
			this->landmark[2 * i] = bbox.ppoint[i];
		for (int i = 0; i < 5; i++)
			this->landmark[2 * i + 1] = bbox.ppoint[i + 5];
		return *this;
	}
};

ncnn::Mat resize(ncnn::Mat src, int w, int h);

ncnn::Mat bgr2rgb(ncnn::Mat src);

ncnn::Mat rgb2bgr(ncnn::Mat src);

void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M);

void warpAffineMatrix(ncnn::Mat src, ncnn::Mat &dst, float *M, int dst_w, int dst_h);

#include "base.cpp"

#endif