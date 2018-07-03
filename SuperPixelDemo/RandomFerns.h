#include"headfile.h"
#include<io.h>
#include"detector_classes.h"
#include"detector_functions.h"
class RandomFerns{
public:
	clsParameters prms;
	clsRFs RFs;
	clsClassifierSet clfrs;
	Mat Mc;
	Mat Mq;
	Mat thetas;
	int numMaxClfrs;
	float xi;
	int imgWidth;
	int imgHeight;

	void init();
	void update(clsParameters* prms, clsClassifierSet* clfrs, clsRFs* RFs, cv::Mat img, clsDetectionSet* detSet, cv::Mat labels, cv::Mat& Mc, cv::Mat& Mq,int label);
	int predict(Mat img);
};