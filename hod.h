#ifndef HOD_H
#define HOD_H


#define PosSamNO 128 
#define NegSamNO 1330
#define TRAIN true 
#define DescriptorDim 192 

#include "COpenNi.h"
#include "HeadFiles.h"
#include "Data3DHelper.h"
#include "MatHelper.h"




class hod
{


private:


	cv::Mat Pos; //正样本;
	cv::Mat Ign;//忽略样本;
	int TotalNum;//总共候选点的个数;
	//候选点的总数 = 正样本 + 负样本 +忽略样本;
	int PosNum;//正样本个数;
	int IgnNum;//负样本个数;

	CvSVM cvsvm;



public:


	hod();
	~hod();

	/*
	void getrectimg(cv::Mat& m_oriBgrResultImage_8UC3,cv::Mat& m_preproResultImage_16UC1);
	*/

	void getHod(cv::Mat& m_oriBgrResultImage_8UC3, cv::Mat& m_preproResultImage_16UC1);


};



#endif