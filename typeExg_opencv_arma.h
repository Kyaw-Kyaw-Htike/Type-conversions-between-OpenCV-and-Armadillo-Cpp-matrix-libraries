#ifndef TYPEEXG_OPENCV_ARMA_H
#define TYPEEXG_OPENCV_ARMA_H

// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

#include "armadillo"
#include "opencv2/opencv.hpp"

namespace hpers_TEOpencvArma
{
	// convert typename and nchannels to opencv mat type such as CV_32FC1
	template <typename T>
	int getOpencvType(int nchannels)
	{
		int depth = cv::DataType<T>::depth;
		return (CV_MAT_DEPTH(depth) + (((nchannels)-1) << CV_CN_SHIFT));
	}
}

template <typename T>
void opencv2arma(const cv::Mat& mat_in, arma::Mat<T>& mat_out)
{
	int nrows = mat_in.rows;
	int ncols = mat_in.cols;

	mat_out.set_size(nrows, ncols);

	T * ptr = mat_out.memptr();
	unsigned long count = 0;	

	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			ptr[count++] = mat_in.at<T>(i, j);			
}

template <typename T>
void arma2opencv(const arma::Mat<T>& mat_in, cv::Mat& mat_out)
{
	int nrows = mat_in.n_rows;
	int ncols = mat_in.n_cols;
	const int nchannels = 1;

	mat_out.create(nrows, ncols, hpers_TEOpencvArma::getOpencvType<T>(nchannels));
				
	for (int j = 0; j < ncols; j++)
		for (int i = 0; i < nrows; i++)
			mat_out.at<T>(i, j) = mat_in.at(i,j);
}


template <typename T, int nchannels>
void opencv2arma(const cv::Mat& mat_in, arma::Cube<T>& mat_out)
{
	int nrows = mat_in.rows;
	int ncols = mat_in.cols;
	//int nchannels = mat_in.channels();

	mat_out.set_size(nrows, ncols, nchannels);

	T * ptr = mat_out.memptr();
	unsigned long count = 0;

	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < ncols; j++)
			for (int i = 0; i < nrows; i++)
				ptr[count++] = mat_in.at<cv::Vec<T, nchannels>>(i, j)[k];			
				
}

template <typename T, int nchannels>
void arma2opencv(const arma::Cube<T>& mat_in, cv::Mat& mat_out)
{	
	int nrows = mat_in.n_rows;
	int ncols = mat_in.n_cols;
	//int nchannels = mat_in.n_slices;

	mat_out.create(nrows, ncols, hpers_TEOpencvArma::getOpencvType<T>(nchannels));

	for (int k = 0; k < nchannels; k++)
		for (int j = 0; j < ncols; j++)
			for (int i = 0; i < nrows; i++)
				mat_out.at<cv::Vec<T, nchannels>>(i, j)[k] = mat_in.at(i, j, k);				
}

#endif