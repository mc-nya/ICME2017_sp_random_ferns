#include"headfile.h"
#include<io.h>
#include"detector_classes.h"
#include"detector_functions.h"
#include"RandomFerns.h"
#include"ImageLoader.h"
void RandomFerns::init(){
	// program parameters
	//prms = clsParameters();
	prms.fun_load();
	prms.fun_print();

	// random ferns
	//RFs = clsRFs();
	RFs.fun_compute(prms.fun_get_pool_size(), prms.fun_get_num_features(), prms.fun_get_fern_size(), prms.fun_get_num_image_channels());
	RFs.fun_print();

	// object classifiers
	//clfrs = clsClassifierSet();

	// num. max. classifiers
	numMaxClfrs = prms.fun_get_num_classifiers();

	// sensitivity learning rate (xi)
	xi = prms.fun_get_learning_rate();

	// incremental learning rates
	Mc = Mat(1, numMaxClfrs, CV_32FC1, Scalar::all(0.0));
	Mq = Mat(1, numMaxClfrs, CV_32FC1, Scalar::all(0.0));

	// uncertainty region thresholds (thetas)
	thetas = Mat(1, numMaxClfrs, CV_32FC1, Scalar::all(1.0));

	// image size
	imgWidth = prms.fun_get_image_width();
	imgHeight = prms.fun_get_image_height();

	//ImageLoader imLoaderP = ImageLoader("E:\\lab\\analysis\\分类集\\p_1+8+11\\", "P_", "_raw_Depth.png", "P_",
	//	"_raw_RGB.png", 1, 650);

	//ImageLoader imLoaderN = ImageLoader("E:\\lab\\analysis\\分类集\\n_1+8+11\\", "N_", "_raw_Depth.png", "N_",
	//	"_raw_RGB.png", 1, 2000);

	//ImageLoader imLoaderP = ImageLoader("E:\\lab\\analysis\\分类集\\p_10+18\\", "P_", "_raw_Depth.png", "P_",
	//	"_raw_RGB.png", 1, 300);

	//ImageLoader imLoaderN = ImageLoader("E:\\lab\\analysis\\分类集\\n_10+18\\", "N_", "_raw_Depth.png", "N_",
	//	"_raw_RGB.png", 1, 1000);
	//ImageLoader imLoaderP = ImageLoader("E:\\lab\\ICME2017\\分类\\positive\\", "positive_", "_Depth.png", "positive_", "_RGB.png", 1, 395);

	//ImageLoader imLoaderN = ImageLoader("E:\\lab\\ICME2017\\分类\\negative\\", "negative_", "_Depth.png", "negative_", "_RGB.png", 1, 1760);

	ImageLoader imLoaderP = ImageLoader("E:\\lab\\analysis\\分类集\\p_38+47\\", "P_", "_raw_Depth.png", "P_",
		"_raw_RGB.png", 1, 910);

	ImageLoader imLoaderN = ImageLoader("E:\\lab\\analysis\\分类集\\n_38+47\\", "N_", "_raw_Depth.png", "N_",
		"_raw_RGB.png", 1, 3000);
	int flag_train = 30;
	while(imLoaderP.next()){
		Mat detLabels;
		clsDetectionSet detSet;
		Mat img = imLoaderP.getColorImage();
		resize(img, img, cvSize(imgWidth, imgHeight));
		CvRect rec = cvRect(0, 0, imgWidth - 1, imgHeight - 1);

		// train step
		if (flag_train){
			flag_train --;
			fun_train_step(&prms, &clfrs, &RFs, img, rec);
		}
		// detection step
		fun_test_step(&prms, &clfrs, &RFs, img, &detSet);
		detLabels = fun_detection_labels(&prms, &detSet, thetas);
		update(&prms, &clfrs, &RFs, img, &detSet, detLabels, Mc, Mq, 1);
		fun_update_uncertainty_thresholds(thetas, Mc, Mq, numMaxClfrs, xi);
		//imLoaderP.next();
	}
	while (imLoaderN.next()){
		Mat detLabels;
		clsDetectionSet detSet = clsDetectionSet();
		Mat img = imLoaderN.getColorImage();
		resize(img, img, cvSize(imgWidth, imgHeight));
		CvRect rec = cvRect(1, 1, imgWidth - 1, imgHeight - 1);

		// train step
		if (flag_train){
			flag_train = 0;
			fun_train_step(&prms, &clfrs, &RFs, img, rec);
		}
		// detection step
		fun_test_step(&prms, &clfrs, &RFs, img, &detSet);
		detLabels = fun_detection_labels(&prms, &detSet, thetas);
		update(&prms, &clfrs, &RFs, img, &detSet, detLabels, Mc, Mq, -1);
		fun_update_uncertainty_thresholds(thetas, Mc, Mq, numMaxClfrs, xi);
		//imLoaderN.next();
	}
	//for (int i = 0; i < 700; i++){
	//	Mat detLabels;
	//	clsDetectionSet detSet;
	//	Mat img = imLoaderP.getColorImage();
	//	resize(img, img, cvSize(imgWidth, imgHeight));
	//	CvRect rec = cvRect(1, 1, imgWidth - 2, imgHeight - 2);
	//	//imshow("load", img);
	//	//cvWaitKey(1);
	//	// train step
	//	if (flag_train){
	//		flag_train = 0;
	//		fun_train_step(&prms, &clfrs, &RFs, img, rec);
	//	}
	//	// detection step
	//	fun_test_step(&prms, &clfrs, &RFs, img, &detSet);
	//	detLabels = fun_detection_labels(&prms, &detSet, thetas);
	//	update(&prms, &clfrs, &RFs, img, &detSet, detLabels, Mc, Mq, 1);
	//	fun_update_uncertainty_thresholds(thetas, Mc, Mq, numMaxClfrs, xi);
	//	imLoaderP.next();
	//}
	
	
};

int RandomFerns::predict(Mat img){
	Mat detLabels,detLabelsReal;
	clsDetectionSet detSet;
	CvRect rec;  // image box
	CvScalar color;  // detection color
	cv::Mat copImg;   // copy image
	resize(img, img, cvSize(imgWidth, imgHeight));
	fun_test_step(&prms, &clfrs, &RFs, img, &detSet);
	detLabels = fun_detection_labels(&prms, &detSet, thetas);
	detLabelsReal = fun_detection_labels_real(&prms, &detSet, thetas);
	int recThick = prms.fun_get_rectangle_thickness();
	int u1, v1, u2, v2, k, numDets;
	float score, label, labelReal;
	clsDetection* det;
	numDets = detSet.fun_get_num_detections();
	copImg = img.clone();
	//int updSamps = prms.fun_get_num_update_samples();
	if ((numDets > 0)){
		float* labPtr = detLabels.ptr<float>(0);
		float* labRealPtr = detLabelsReal.ptr<float>(0);
		for (int iter = 0; iter < numDets; iter++){
			det = detSet.fun_get_detection(iter);
			det->fun_get_values(u1, v1, u2, v2, score, k);
			float area = (v2 - v1)*(u2 - u1);
			float totalArea = imgWidth*imgHeight;
			label = *(labPtr + iter);
			labelReal = *(labRealPtr + iter);
			rec = cvRect(v1, u1, v2 - v1, u2 - u1);
			//if (label>0.7){
			//	cv::rectangle(copImg, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(0, 0, 0), recThick + 3);
			//	cv::rectangle(copImg, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(255, 255, 0), recThick);
			//}
			//if (label == 0.5){
			//	cv::rectangle(copImg, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(0, 0, 0), recThick + 3);
			//	cv::rectangle(copImg, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(0, 255, 255), recThick);
			//}
			//if (label < 0.5){
			//	cv::rectangle(copImg, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(0, 0, 0), recThick + 3);
			//	cv::rectangle(copImg, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(0, 255, 0), recThick);
			//}
			//cv::imshow("detection", copImg);
			//cvWaitKey();
			//cout << "ratio: " << area / totalArea << "  score:  " << labelReal << endl;
			if ( labelReal > 0.895){
				return 1;
			}
			if ((area / totalArea)>0.7 && labelReal > 0.96){
				return 1;
			}
		}

	}
	return 0;
};
void RandomFerns::update(clsParameters* prms, clsClassifierSet* clfrs, clsRFs* RFs, cv::Mat img, clsDetectionSet* detSet, cv::Mat labels, cv::Mat& Mc, cv::Mat& Mq,int label_human){
	// parameters
	float beta = prms->fun_get_threshold();  // classifier threshold (beta)
	float varphi = prms->fun_get_image_shift();  // image shift factor (varphi)
	int recThick = prms->fun_get_rectangle_thickness();  // detection rectangle thickness
	int updSamps = prms->fun_get_num_update_samples();  // number of new (updating) samples (Nu)

	// variables
	char key;  // keyword key
	CvRect rec;  // image box
	CvScalar color;  // detection color
	cv::Mat copImg;   // copy image
	float posValue = 1.0;  // positive value (y=+1)
	float negValue = -1.0;  // negative value (y=-1)
	float score, label;  // detection score and sample label (y)
	clsDetection* det;  // detection
	clsClassifier* clfr;  // classifier pointer
	int objHeight, objWidth;  // object size (Bu,Bv)
	int u1, v1, u2, v2, k, numDets;  // image coordinates, classifier index (k), and num. detections
	
	numDets = detSet->fun_get_num_detections();
	img.copyTo(copImg);

	if ((numDets > 0) && (updSamps>0)){

		// pointer
		float* McPtr = Mc.ptr<float>(0);  // num. correct detections (Mc)
		float* MqPtr = Mq.ptr<float>(0);  // num. human assistances (Mq)
		float* labPtr = labels.ptr<float>(0); // labels (y)

		// detections
		for (int iter = 0; iter < numDets; iter++){
			color = cvScalar(255, 255, 255);
			det = detSet->fun_get_detection(iter);
			det->fun_get_values(u1, v1, u2, v2, score, k);
			clfr = clfrs->fun_get_classifier(k);
			clfr->fun_get_object_size(objHeight, objWidth);
			rec = cvRect(v1, u1, v2 - v1, u2 - u1);
			cv::Mat patch = cv::Mat(cvSize(objWidth, objHeight), CV_8UC3);
			cv::resize(img(rec), patch, cvSize(objWidth, objHeight));
			cv::Mat fernMaps = fun_fern_maps(patch, RFs);
			label = *(labPtr + iter);
			cv::rectangle(img, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(0, 0, 0), recThick + 3);
			cv::rectangle(img, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(255, 255, 0), recThick);
			//cv::imshow("detection", img);
			//cvWaitKey();
			if (label_human == -1 && label>0.5){
				//CvRect rec = cvRect(1, 1, imgWidth - 2, imgHeight - 2);
				fun_update_negative_samples_fps(copImg, rec, clfr, RFs, updSamps, varphi);
			}
		}
	}
	if (label_human == 1){
		CvRect rec = cvRect(1, 1, imgWidth - 2, imgHeight - 2);
		fun_update_positive_samples(copImg, rec, clfr, RFs, updSamps, varphi);
	}
	
};


