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


	cv::Mat Pos; //������;
	cv::Mat Ign;//��������;
	int TotalNum;//�ܹ���ѡ��ĸ���;
	//��ѡ������� = ������ + ������ +��������;
	int PosNum;//����������;
	int IgnNum;//����������;

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