
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "detector_classes.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

// save image
void fun_save_image(cv::Mat img, long int number){
  // inputs:
  // number: image number

  // parameters
  int numMaxImgs = 10000;  // num. max. images to save

  // variables
  char imgName[50];

  // path and image name
  if (number<10000)
    sprintf(imgName, "../images/image_%ld.jpg",number);
  if (number<1000)
    sprintf(imgName, "../images/image_0%ld.jpg",number);
  if (number<100)
    sprintf(imgName, "../images/image_00%ld.jpg",number);
  if (number<10)
    sprintf(imgName, "../images/image_000%ld.jpg",number);

  // save image
  if (number<numMaxImgs)
    cv::imwrite(imgName, img);

  // message
  if (number>=numMaxImgs)
    cout << "Warning: cannot save images, the number of images is very large" << endl;

};

// draw message
void fun_draw_message(clsParameters* prms, cv::Mat img, char* text, CvPoint location, CvScalar textColor){
  // inputs:
  // prms: program parameters
  // img: input image
  // text: message
  // location: text locaion inside image
  // textColor: text color

  // text font
  float textFont = prms->fun_get_text_font();

  // message
  cv::putText(img, text, location, cv::FONT_HERSHEY_PLAIN, textFont, cvScalar(10,10,10), 8);
  cv::putText(img, text, location, cv::FONT_HERSHEY_PLAIN, textFont, textColor, 3);

};

// draw rectangle
void fun_draw_rectangle(clsParameters* prms, cv::Mat img, CvRect rec, CvScalar recColor){
  // inputs:
  // prms: program parameters
  // img: input image
  // rec: rectangle
  // recColor: rectangle color

  // variables
  CvPoint topLeft = cvPoint(rec.x, rec.y);  // top-left rectangle position
  CvPoint bottomRight = cvPoint(rec.x+rec.width, rec.y+rec.height);  // bottom-right rectangle position

  // rectangle thickness
  int recThick = prms->fun_get_rectangle_thickness();

  // draw rectangle
  cv::rectangle(img, topLeft, bottomRight, recColor, recThick);

};

// show frame
void fun_show_frame(clsParameters* prms, cv::Mat img, int contFrame, double fps){
  // inputs:
  // prms: program parameters
  // img: input image
  // contFrame: frame counter
  // fps: frames per second

  // parameters
  int ua = 30;  // frame location in u
  int va = 160;  // frame location in v
  int ub = 30;  // fps location in u
  int vb = 300;  // fps location in v

  // text font
  float textFont = prms->fun_get_text_font();

  // variables
  char text1[50];  // text
  char text2[50];  // text
  CvPoint uv1,uv2;  // image positions

  // rectangle coordinates
  uv1 = cvPoint(va, ua);
  uv2 = cvPoint(vb, ub);

  // messages
  sprintf(text1, "#%i",contFrame);
  sprintf(text2, "FPS: %.1f",fps);
  cv::putText(img, text1, uv1, cv::FONT_HERSHEY_PLAIN, textFont, cvScalar(10, 10, 10), 8);
  cv::putText(img, text1, uv1, cv::FONT_HERSHEY_PLAIN, textFont, cvScalar(0, 255, 0), 3);
  cv::putText(img, text2, uv2, cv::FONT_HERSHEY_PLAIN, textFont, cvScalar(10, 10, 10), 8);
  cv::putText(img, text2, uv2, cv::FONT_HERSHEY_PLAIN, textFont, cvScalar(0, 255, 0), 3);

};

// detection score
void fun_detection_score(clsParameters* prms, cv::Mat img, clsDetectionSet* detSet){
  // inputs:
  // prms: program parameters
  // img: input image
  // detSet: set of detections

  // parameters
  int ua = 30;  // score location in u (y)
  int va = img.cols -100;  // score location in v (x)

  // variables
  float score;  // detection score
  char text[50];  // text
  int u1,v1,u2,v2,ide;  // detection coordinates and ide

  // max. detection
  clsDetection* maxDet = new clsDetection;

  // maxima detection
  detSet->fun_get_max_detection(maxDet);

  // detection values
  maxDet->fun_get_values(u1, v1, u2, v2, score, ide);

  // detection score
  sprintf(text, "%.2f",score);

  // message
  fun_draw_message(prms, img, text, cvPoint(va, ua), cvScalar(0, 255, 0));

  // release
  delete maxDet;
};

// show image patch
void fun_show_image_patch(clsParameters* prms, cv::Mat img, clsDetectionSet* detSet){
  // inputs:
  // prms: program parameters
  // img: input image
  // detSet: detection set

  // parameters
  int patchSize = 50;  // patch size

  // variables
  float score;  // detection score
  int u1,v1,u2,v2,ide;  // detection coordinates and ide

  // max. detection
  clsDetection* maxDet = new clsDetection;

  // visualization area (top-left corner)
  CvRect area = cvRect(1, 1, patchSize, patchSize);

  // maxima detection
  detSet->fun_get_max_detection(maxDet);

  // detection values
  maxDet->fun_get_values(u1, v1, u2, v2, score, ide);

  // bounding box
  CvRect rec = cvRect(v1, u1, v2-v1, u2-u1);

  // check
  if (v1!=0 && v2!=0 && u1!=0 && u2!=0){

    // patch
    cv::Mat patch = cv::Mat(cvSize(patchSize, patchSize), CV_8UC3);
    cv::resize(img(rec), patch, cvSize(patchSize, patchSize));

    // add patch into image
    addWeighted(img(area), 0.1, patch, 0.9, 0.0, img(area));

  }

  // release
  delete maxDet;
};

// draw object model
void fun_draw_object_model(cv::Mat img, cv::Mat objModel, int objSize, int u, int v, CvScalar recColor){
  // inputs:
  // prms: program parameters
  // objModel: object image (model)
  // objSize: object model size
  // u: image location in u
  // v: image location in v
  // recColor: rectangle color

  // draw object model
  if((v<=img.cols-objSize) && (u<=img.rows-objSize) && (v>=0) && (u>=0)){
    CvRect area = cvRect(v, u, objSize, objSize);  // object model area
    cv::resize(objModel, objModel, cvSize(objSize, objSize));  // resize object model
    addWeighted(img(area), 0.1, objModel, 0.9, 0.0, img(area));  // show object model
    cv::rectangle(img, cvPoint(v, u), cvPoint(v+objSize-1, u+objSize-1), recColor, 2);  // add object bounding box
  }

};

// draw detections
void fun_draw_detections(clsParameters* prms, clsClassifierSet* clfrs, cv::Mat img, clsDetectionSet* detSet, cv::Mat labels){
  // inputs:
  // prms: program parameters
  // clfrs: classifiers
  // img: input image
  // detSet: detection set
  // labels: detection labels (y)

  // variables
  int visuMode;  // visualization mode 
  int numClfrs;  // num. classifiers (K)
  char text1[50];  // text message
  float textFont;  // text font
  clsDetection* det;  // detection
  float score,label;  // detection score and sample label (y)
  int u1,v1,u2,v2,k;  // detection coordinates and classifier index
  clsClassifier* clfr;  // classifier pointer
  int numDets,recThick;  // num. detections and rectangle thickness
  int objSize,tmp1,tmp2;  // image model size and temp. variables  
  CvPoint topLeft,bottomRight;  // image positions
  CvScalar recColor = cvScalar(0, 255, 0);  // default rectangle color

  // rectangle thickness
  recThick = prms->fun_get_rectangle_thickness();

  // text font
  textFont = prms->fun_get_text_font();

  // visualization mode
  visuMode = prms->fun_get_visualization_mode();

  // num. object classifiers
  numClfrs = clfrs->fun_get_num_classifiers();

  // num. detections
  numDets = detSet->fun_get_num_detections();

  // pointer to labels (y)
  float* labPtr = labels.ptr<float>(0);

  // object model size
  objSize = round(img.rows/10); 

  // show all object models (mode 2)
  if (visuMode==2||visuMode==3){
    for (int k=0; k<numClfrs; k++){
      // get classifier k
      clfr = clfrs->fun_get_classifier(k);
      // image coordinates
      if (k<10){tmp1 = k; tmp2 = 1;}
      if (k>=10 && k<20){tmp1 = k-10; tmp2 = 2;}
      if (k>=20){ cout << "Warning: only 20 object models can be shown" << endl; }
      u1 = tmp1*objSize;
      v1 = img.cols - tmp2*objSize;
      // draw object model
      fun_draw_object_model(img, clfr->objModel, objSize, u1, v1, prms->fun_get_color(k));
    }
  }

  // draw each detection
  for (int iter=0; iter<numDets; iter++){

    // sample label (y)
    label = *(labPtr + iter);

    // only positive detections

    // positive sample
    if (label==1.0){

      // current detection
      det = detSet->fun_get_detection(iter);

      // detection values
      det->fun_get_values(u1, v1, u2, v2, score, k);

      // rectangle coordinates
      topLeft = cvPoint(v1, u1);  // top-left position
      bottomRight = cvPoint(v2, u2);  // bottom-right position

      // visualization mode
      switch (visuMode){
        case 1:
          // draw detection score
          sprintf(text1, "%.2f",score);
          cv::putText(img, text1, cvPoint(v1+50, u1-15), cv::FONT_HERSHEY_PLAIN, textFont, cvScalar(0,0,0), 6);
          cv::putText(img, text1, cvPoint(v1+50, u1-15), cv::FONT_HERSHEY_PLAIN, textFont, recColor, 2);
          // draw rectangle
          cv::rectangle(img, topLeft, bottomRight, cvScalar(0,0,0), recThick+3);
          cv::rectangle(img, topLeft, bottomRight, recColor, recThick);
          // draw object model
          clfr = clfrs->fun_get_classifier(k);  // current classifier
          fun_draw_object_model(img, clfr->objModel, objSize, u1-objSize, v1, cvScalar(0, 0, 0));  // draw model
          break;
        case 2:
          // draw detection score
          sprintf(text1, "%.2f",score);
          cv::putText(img, text1, cvPoint(v1+5, u1-10), cv::FONT_HERSHEY_PLAIN, textFont, cvScalar(0,0,0), 6);
          cv::putText(img, text1, cvPoint(v1+5, u1-10), cv::FONT_HERSHEY_PLAIN, textFont, prms->fun_get_color(k), 2);
          // draw rectangle
          cv::rectangle(img, topLeft, bottomRight, cvScalar(0,0,0), recThick+3);
          cv::rectangle(img, topLeft, bottomRight, prms->fun_get_color(k), recThick);
          break;
        case 3:
          // draw rectangle
          cv::rectangle(img, topLeft, bottomRight, cvScalar(0,0,0), recThick+3);
          cv::rectangle(img, topLeft, bottomRight, prms->fun_get_color(k), recThick);
          break;
      }

    }
  }
};

// image equalization
void fun_image_equalization(cv::Mat img){
  // inputs:
  // img: input image

  // temporal image
  cv::Mat imgSpace = img;
  cv::resize(imgSpace, imgSpace, cvSize(img.cols, img.rows));

  // convert color space: YCrCb or Lab
  cvtColor(img,imgSpace,CV_BGR2Lab);

  // split image into color channels
  std::vector<cv::Mat> imgChannels;
  cv::split(imgSpace, imgChannels);

  // equalize histograms
  cv::equalizeHist(imgChannels[0], imgChannels[0]);    // equalize channel 1

  // merge channels
  cv::merge(imgChannels, imgSpace);

  // convert color space: YCrCb or Lab
  cvtColor(imgSpace, img, CV_Lab2BGR);

};

// update uncertainty region thresholds (thetas)
void fun_update_uncertainty_thresholds(cv::Mat& thetas, cv::Mat Mc, cv::Mat Mq, int numMaxClfrs, float xi){
  // inputs:
  // thetas: uncertainty thresholds (thetas)
  // Mc: vector of correct predictions
  // Mq: vector of num. human assistances
  // numMaxClfrs: num. max. classifiers
  // xi: sensitivity learning rate

  // pointers
  float* McPtr = Mc.ptr<float>(0);  // num. correct predictions vector (Mc)
  float* MqPtr = Mq.ptr<float>(0);  // num. human assistance vector (Mq)
  float* thetasPtr = thetas.ptr<float>(0);  // uncertainty thresholds (thetas)

  // adaptive threshold
  for (int k=0; k<numMaxClfrs; k++){
    // update theta k: theta_k = 1 - xi * (M^c_k/M^q_k)
    *(thetasPtr + k) = 1.0 - xi*(*(McPtr + k)/(*(MqPtr + k) + 0.0001));
    // check negative values or small values
    if (*(thetasPtr + k)<=0){*(thetasPtr + k) = 0.001;}
  }

};

// show human-asssted learnning results
void fun_show_learning_results(clsParameters* prms, cv::Mat &img, cv::Mat thetas, cv::Mat Mc, cv::Mat Mq, int k, long int contFrame){
  // inputs:
  // prms: program parameters
  // img: image
  // thetas: uncertainty thresholds
  // Mc: correct predictions vector
  // Mq: num. human assistance vector
  // k: classifier index
  // contFrame: frame counter

  // parameters
  int va = 60;  // text location
  int ua = img.rows - 52;  // text location
  int vb = 255;  // text location
  int ub = img.rows - 52;  // text location
  int vc = 435;  // text location
  int uc = img.rows - 52;  // text location
  int modSize = 50;  // image model size

  // variables
  int visuMode;  // visualization mode
  CvRect area;  // image area
  char text[50];  // text message
  CvScalar textColor;  // text color

  // visualization mode
  visuMode = prms->fun_get_visualization_mode();

  // text color according to visualization mode
  if ((visuMode)==1){textColor = cvScalar(0, 255, 0);}  // default color
  if ((visuMode)!=1){textColor = prms->fun_get_color(k);}  // object colors

   // pointers
  float* McPtr = Mc.ptr<float>(0);  // num. correct predictions vector (Mc)
  float* MqPtr = Mq.ptr<float>(0);  // num. human assistances vector (Mq)
  float* thetasPtr = thetas.ptr<float>(0);  // uncertainty thresholds (thetas)

  // logo images
  cv::Mat asstImg = cv::imread("../logos/hand.png", CV_LOAD_IMAGE_COLOR);  // human assistance
  cv::Mat thetaImg = cv::imread("../logos/theta.png", CV_LOAD_IMAGE_COLOR);  // theta threshold
  cv::Mat lambdaImg = cv::imread("../logos/lambda.png", CV_LOAD_IMAGE_COLOR);  // lambda performance

  // check errors
  if (!asstImg.data){
    cout << "Error: The hands logo cannot be open. Please check the models folder path."  << endl;
    exit(0);
  }
  if (!thetaImg.data){
    cout << "Error: The theta logo cannot be open. Please check the models folder path."  << endl;
    exit(0);
  }
  if (!lambdaImg.data){
    cout << "Error: The lambda logo cannot be open. Please check the models folder path."  << endl;
    exit(0);
  }

  // show logos
  if (visuMode!=3){
    // show human assitance logo
    area = cvRect(va, ua, modSize, modSize);
    cv::resize(asstImg, asstImg, cvSize(modSize, modSize));
    addWeighted(img(area), 0.1, asstImg, 0.9, 0.0, img(area));
    // show theta logo
    area = cvRect(vb, ub, modSize, modSize);
    cv::resize(thetaImg, thetaImg, cvSize(modSize, modSize));
    addWeighted(img(area), 0.1, thetaImg, 0.9, 0.0, img(area));
    // show lambda logo
    area = cvRect(vc, uc, modSize, modSize);
    cv::resize(lambdaImg, lambdaImg, cvSize(modSize, modSize));
    addWeighted(img(area), 0.1, lambdaImg, 0.9, 0.0, img(area));
  }

  // show learning rates
  if (visuMode!=3){
    // show classifier index
    sprintf(text, "[%d]",k+1);
    fun_draw_message(prms, img, text, cvPoint(1, ua+40), textColor);
    // percentage human annotations
    sprintf(text, "%.1f[%%]",100* *(MqPtr + k)/contFrame);
    fun_draw_message(prms, img, text, cvPoint(va+50, ua+40), textColor);
    // uncertainty threshold value (theta)
    sprintf(text, "%.3f",*(thetasPtr + k));
    fun_draw_message(prms, img, text, cvPoint(vb+50, ub+40), textColor);
    // percentage classifier performance (lambda)
    sprintf(text, "%.1f[%%]",100* *(McPtr + k)/ *(MqPtr + k));
    fun_draw_message(prms, img, text, cvPoint(vc+50, uc+40), textColor);
  }
};

// detection labels
cv::Mat fun_detection_labels(clsParameters* prms, clsDetectionSet* detSet, cv::Mat thetas){
  // inputs:
  // prms: program parameters
  // detSet: detection set
  // thetas: uncertainty thresholds

  // variables
  clsDetection* det;  // detection
  float score,beta,theta;  // detection score, classifier threshold (beta), and uncertainty threshold (theta)
  int va,ua,vb,ub,k,numDets;  // image coordinates, classifier index (k), and num. detections

  // classifier threshold (beta)
  beta = prms->fun_get_threshold();

  // num. detections
  numDets = detSet->fun_get_num_detections();

  // create detection labels (y)
  cv::Mat labels = cv::Mat(numDets, 1, CV_32FC1, cv::Scalar::all(0));

  // pointers
  float* thetasPtr = thetas.ptr<float>(0);  // uncertainty thresholds (thetas)
  float* labPtr = labels.ptr<float>(0); // labels (y)

  // detections
  for (int iter=0; iter<numDets; iter++){

    // current detection
    det = detSet->fun_get_detection(iter);

    // detection values
    det->fun_get_values(ua, va, ub, vb, score, k);

    // uncertainty region threshold (theta)
    theta = *(thetasPtr + k);
	//cout << score << endl;
	//cvWaitKey(1);
	//if (score>beta)
	//	*(labPtr + iter) = 1.0;
	//if (score<beta)
	//	*(labPtr + iter) = -1.0;

    // positive detection
    if (score>beta + 0.5*theta)
      *(labPtr + iter) = 1.0;
	
    // negative detection
    if (score<beta- 0.5*theta)
      *(labPtr + iter) = -1.0;

    // require human asssitance (active learning)
    if ((score >= beta - 0.5*theta) && (score <= beta + 0.5*theta))
      *(labPtr + iter) = 0.5;

  }

  // return detection labels
  return labels;

};


cv::Mat fun_detection_labels_real(clsParameters* prms, clsDetectionSet* detSet, cv::Mat thetas){
	// inputs:
	// prms: program parameters
	// detSet: detection set
	// thetas: uncertainty thresholds

	// variables
	clsDetection* det;  // detection
	float score, beta, theta;  // detection score, classifier threshold (beta), and uncertainty threshold (theta)
	int va, ua, vb, ub, k, numDets;  // image coordinates, classifier index (k), and num. detections

	// classifier threshold (beta)
	beta = prms->fun_get_threshold();

	// num. detections
	numDets = detSet->fun_get_num_detections();

	// create detection labels (y)
	cv::Mat labels = cv::Mat(numDets, 1, CV_32FC1, cv::Scalar::all(0));

	// pointers
	float* thetasPtr = thetas.ptr<float>(0);  // uncertainty thresholds (thetas)
	float* labPtr = labels.ptr<float>(0); // labels (y)

	// detections
	for (int iter = 0; iter<numDets; iter++){

		// current detection
		det = detSet->fun_get_detection(iter);

		// detection values
		det->fun_get_values(ua, va, ub, vb, score, k);

		// uncertainty region threshold (theta)
		theta = *(thetasPtr + k);
		*(labPtr + iter) = score;
		// positive detection
		//if (score>beta + 0.5*theta)
		//	*(labPtr + iter) = 1.0;

		//// negative detection
		//if (score<beta - 0.5*theta)
		//	*(labPtr + iter) = -1.0;

		//// require human asssitance (active learning)
		//if ((score >= beta - 0.5*theta) && (score <= beta + 0.5*theta))
		//	*(labPtr + iter) = 0.5;

	}

	// return detection labels
	return labels;

};
// fern maps
cv::Mat fun_fern_maps(cv::Mat img, clsRFs* RFs){
  // inputs:
  // img: image sample x
  // RFs: shared random ferns

  // detector parameters
  int imgChans = img.channels();  // num. image channels (C)
  int imgWidth = img.cols;  // image width (Iv)
  int imgHeight = img.rows;  // image height (Iu)
  int poolSize = RFs->fun_get_pool_size();  // num. fern features parameters (R)
  int fernSize = RFs->fun_get_fern_size();  // fern size (S)
  int numFeats = RFs->fun_get_num_features();  // num. features per fern (M)
  cv::Mat fernsData = RFs->fun_get_data();  // pointer to random ferns

  // create ferns maps for image sample x
  cv::Mat fernMaps = cv::Mat(cvSize(imgWidth, imgHeight), CV_16UC(poolSize));

  // pointers
  unsigned char *imgPtr = (unsigned char*)(img.data);  // image data
  unsigned char *fernsPtr = (unsigned char*)(fernsData.data);  // fern parameters data
  unsigned short *mapsPtr = (unsigned short*)(fernMaps.data);  // fern maps data

  // variables
  float xa,xb; // image pixel values
  int tmp1 = numFeats*6;  // tmp. variable
  int tmp5 = imgWidth*poolSize;  // tmp. variable
  int tmp4 = imgWidth*imgChans;  // tmp. variable
  int tmp2,tmp3;  // tmp. variables
  int vm,um,cm,z;  // feature parameters (u,v,c) and fern output (z)

  // scanning
  for (int u=0; u<imgHeight-fernSize; u++) {
    for (int v=0; v<imgWidth-fernSize; v++) {
      for (int r=0; r<poolSize; r++){

        // temp. variable
        tmp2 = r*tmp1;

        // fern output
        z = 0;

        // fern features
        for (int m=0; m<numFeats; m++){

          // temp. variable
          tmp3 = tmp2 + m*6;

          // point a coordinates (u,v,c)
          um = u + (int)*(fernsPtr + tmp3 + 0);
          vm = v + (int)*(fernsPtr + tmp3 + 1);
          cm =     (int)*(fernsPtr + tmp3 + 2);

          // image pixel value in location a 
          xa = *(imgPtr + um*tmp4 + vm*imgChans + cm);

          // point b coordinates (u,v,c)
          um = u + (int)*(fernsPtr + tmp3 + 3);
          vm = v + (int)*(fernsPtr + tmp3 + 4);
          cm =     (int)*(fernsPtr + tmp3 + 5);

          // image pixel value in location b
          xb = *(imgPtr + um*tmp4 + vm*imgChans + cm);

          // update fern output
          //z += (1 << m) & (0 - (xa > xb));
          z += (1 << m) & (0 - (xa > xb + 0.01));

        }

        // fern output
        *(mapsPtr + u*tmp5 + v*poolSize + r) = z;
      }
    }
  }

  // return fern maps
  return fernMaps;
};

// scanning window
void fun_scanning_window(cv::Mat fernMaps, clsClassifierSet* clfrs, clsDetectionSet* detSet){
  // inputs:
  // fernMaps: fern outputs over a sample x
  // clfrs: classifiers
  // detSet: detection set

  // parameters
  int minJ = 20;  // num. min. tested ferns
  float thr = 0.4;  // default detector threshold
  bool speed = true;  // naive cascade
  float minScore = 0.4;  // min. score (cascade)

  // variables
  float score;  // detection score
  int detCont = 0;  // detection counter
  int uj,vj,wj;  // fern location (uj,vj) and parameters (omega)
  int z,numMaxDets;  // fern output and num. max. detections
  clsClassifier* clfr;  // classifier pointer
  int numBins,numFerns;  // num. histogram bins and ferns
  int objHeight,objWidth;  // object size (Bu,Bv)
  cv::Mat clfrData,ratHstms;  // classifier data and ratio ferns probabilities
  int imgHeight,imgWidth,poolSize;  // image size (Iu,Iv) and pool size (R)
  int numClfrs = clfrs->fun_get_num_classifiers();  // num. classifiers (K)

  // classifiers
  for (int k=0; k<numClfrs; k++){

    // current classifier
    clfr = clfrs->fun_get_classifier(k);

    // detector parameters
    clfrData = clfr->fun_get_data();  // classifier data pointer
    numFerns = clfr->fun_get_num_ferns();  // num. random ferns (J)
    ratHstms = clfr->fun_get_ratHstms();  // classifier distributions
    numBins = ratHstms.cols;  // num. histogram bins

    // object size
    clfr->fun_get_object_size(objHeight, objWidth);

    // num. max. detections
    numMaxDets = detSet->fun_get_num_max_detections();

    // ferm map size
    poolSize = fernMaps.channels();  // num. ferns parameters (R)
    imgWidth = fernMaps.cols;  // image width (Iv)
    imgHeight = fernMaps.rows;  // image height (Iu)

    // pointers to fern maps, classifier data and fern ratio distributions
    float* ratPtr = ratHstms.ptr<float>(0);
    unsigned char *clfrPtr = (unsigned char*)(clfrData.data);
    unsigned short *mapsPtr = (unsigned short*)(fernMaps.data);

    // scanning
    for (int u=0; u<imgHeight-objHeight; u++){
      for (int v=0; v<imgWidth-objWidth; v++){

        // detection score
        score = 0;

        // ferns
        for (int j=0; j<numFerns; j++){

          // fern j parameters
          uj = u + (int)*(clfrPtr + j*3 + 0);  // location u
          vj = v + (int)*(clfrPtr + j*3 + 1);  // location v
          wj =     (int)*(clfrPtr + j*3 + 2);  // fern features parameter (omega)

          // fern output
          z = *(mapsPtr + uj*imgWidth*poolSize + vj*poolSize + wj);

          // update detection score
          score+= *(ratPtr + j*numBins + z);

          // a naive cascade to speed up detection
          if (speed && (j+1)>minJ && (score/(j+1))<minScore)
            break;

        }

        // normalize score
        score = score/numFerns;

        // check amount of detections
        if (detCont>numMaxDets)
          cout << "Warning: there are many object hypotheses" << endl;

        // save detections
        if(score>thr && detCont<numMaxDets){

          // temporal detection
          clsDetection* det = new clsDetection;

          // set detection values
          det->fun_set_values(u+1, v+1, u+objHeight, v+objWidth, score, k);

          // add detection to detection set
          detSet->fun_set_detection(det, detCont);

          // increment the number of detections
          detCont++;

          // save the number of detections
          detSet->fun_set_num_detections(detCont);

          // release
          delete det;
        }
      }
    }
  }
};

// object detection
void fun_detect(clsParameters* prms, clsClassifierSet* clfrs, clsRFs* RFs, cv::Mat img, clsDetectionSet* detSet){
  // inputs:
  // prms: program parameters
  // clfrs: classifiers
  // RFs: fern features parameters
  // img: image (I)
  // detSet: detection set

  // parameters
  int minCell = prms->fun_get_min_cell_size();  // min. cell size
  int maxCell = prms->fun_get_max_cell_size();  // min. cell size
  int poolSize = RFs->fun_get_pool_size();  // num. ferns parameters (R)

  // variables
  clsII II;  // integral image
  int numMaxDets;  // number of detections
  int imgWidth,imgHeight;  // image size (Iv,Iu)
  int objHeight,objWidth;  // object size (Bu,Bv)

  // num. max. output detections
  numMaxDets = detSet->fun_get_num_max_detections();

  // compute integral image
  II.fun_integral_image(img);

  // image levels (scales)
  for (int cellSize=minCell; cellSize<maxCell; cellSize++){

    // temporal detections
    clsDetectionSet* dets = new clsDetectionSet;

    // compute image from II
    II.fun_compute_image(cellSize);
    II.fun_get_image_size(imgWidth, imgHeight);

    // ferns maps (convolve ferns with the image)
    cv::Mat fernMaps = fun_fern_maps(II.fun_get_image(), RFs);

    // test the classifiers
    fun_scanning_window(fernMaps, clfrs, dets);

    // non maxima supression
    dets->fun_non_maxima_supression();

    // scaling detection coordinates
    dets->fun_scaling(cellSize);

    // adding detections
    detSet->fun_add_detections(dets);

    // release image
    II.fun_release_image();

    // release
    delete dets;
  }

  // non maxima supression
  detSet->fun_non_maxima_supression();
};

// update classifier using positive smaples
void fun_update_positive_samples(cv::Mat img, CvRect rec, clsClassifier* clfr, clsRFs* RFs, int updSamps, float varphi){
  // inputs:
  // img: image (I)
  // rec: image box
  // clfr: classifier
  // RFs: fern parameters
  // updSamps: num. new (updating) samples (Nu)
  // varphi: image shift rate

  // parameters
  int minSize = 20;  // min. image size
  float posLabel = 1.0; // sample label (y=+1)

  // variables
  CvRect newRec;  // new bounding box
  int ua,va,ub,vb;  // image coordinates
  int objHeight,objWidth;  // object heigth and width (Bu,Bv)

  // object size
  clfr->fun_get_object_size(objHeight, objWidth);

  // updating positive samples
  for (int iter=0; iter<updSamps; iter++){

    // new rectangle coordinates
    ua = rec.y + round(varphi*rec.height*((float)rand()/RAND_MAX - 0.5));
    va = rec.x + round(varphi*rec.width*((float)rand()/RAND_MAX  - 0.5));
    ub = rec.y + rec.height + round(varphi*rec.height*((float)rand()/RAND_MAX - 0.5));
    vb = rec.x + rec.width + round(varphi*rec.width*((float)rand()/RAND_MAX  - 0.5));

    // check limits
    if (va<0){va = 0;}
    if (ua<0){ua = 0;}
    if (vb>=img.cols){vb = img.cols-1;}
    if (ub>=img.rows){ub = img.rows-1;}

    // check min. size
    if ((vb-va)>minSize && (ub-ua)>minSize){

      // new bounding box
      newRec = cvRect(va, ua, vb-va, ub-ua);

      // image patch using ROI
      cv::Mat patch = cv::Mat(cvSize(objWidth, objHeight), CV_8UC3);
      cv::resize(img(newRec), patch, cvSize(objWidth, objHeight));

      // fern maps (convolve ferns with the image)
      cv::Mat fernMaps = fun_fern_maps(patch, RFs);

      // update classifier
      clfr->fun_update(fernMaps, posLabel);

    }
  }
};

//update classifier using negative samples (random samples)
void fun_update_negative_samples_rnd(cv::Mat img, CvRect rec, clsClassifier* clfr, clsRFs* RFs, int updSamps, float varphi){
  // inputs:
  // img: image (I)
  // rec: image box
  // clfr: classifier
  // RFs: fern parameters
  // updSamps: num. new (updating) samples (Nu)
  // varphi: image shift rate (not used here)

  // parameters
  int minSize = 20;  // min. image size
  float negLabel = -1.0;  // sample label (y=-1)

  // variables
  CvRect newRec;  // image box
  int ua,va,ub,vb;  // image coordinates
  int u1,v1,u2,v2;  // image corrdinates
  int objHeight,objWidth;  // object size (Bu,Bv)

  // object size
  clfr->fun_get_object_size(objHeight, objWidth);

  // updating negative samples
  for (int iter=0; iter<updSamps; iter++){

    // new rectangle coordinates (random location)
    v1 = round(img.cols*((float)rand()/RAND_MAX));
    v2 = round(img.cols*((float)rand()/RAND_MAX));
    u1 = round(img.rows*((float)rand()/RAND_MAX));
    u2 = round(img.rows*((float)rand()/RAND_MAX));
    va = min(v1, v2);
    vb = max(v1, v2);
    ua = min(u1, u2);
    ub = max(u1, u2);

    // check limits
    if (va<0){va = 0;}
    if (ua<0){ua = 0;}
    if (vb>=img.cols){vb = img.cols-1;}
    if (ub>=img.rows){ub = img.rows-1;}

    // check min. size
    if ((vb-va)>minSize && (ub-ua)>minSize){

      // new bounding box
      newRec = cvRect(va, ua, vb-va, ub-ua);

      // image patch using ROI
      cv::Mat patch = cv::Mat(cvSize(objWidth, objHeight), CV_8UC3);
      cv::resize(img(newRec), patch, cvSize(objWidth, objHeight));

      // fern maps (convolve ferns with the image)
      cv::Mat fernMaps = fun_fern_maps(patch, RFs);

      // update classifier
      clfr->fun_update(fernMaps, negLabel);

    }
  }
};

//update classifier using negative samples (false positive samples)
void fun_update_negative_samples_fps(cv::Mat img, CvRect rec, clsClassifier* clfr, clsRFs* RFs, int updSamps, float varphi){
  // inputs:
  // img: image
  // rec: image box
  // clfr: classifier
  // RFs: fern parameters
  // updSamps: num. new (updating) samples (Nu)
  // varphi: image shift rate

  // parameters
  int minSize = 20;  // min. image size
  float negLabel = -1.0;  // sample label (y=-1)

  // variables
  CvRect newRec;  // image box
  int ua,va,ub,vb;  // image coordinates
  int u1,v1,u2,v2;  // image coordinates
  int objHeight,objWidth;  // object size (Bu,Bv)

  // object size
  clfr->fun_get_object_size(objHeight, objWidth);

  // updating positive samples
  for (int iter=0; iter<updSamps; iter++){

    // new rectangle coordinates
    ua = rec.y + round(varphi*rec.height*((float)rand()/RAND_MAX - 0.5));
    va = rec.x + round(varphi*rec.width*((float)rand()/RAND_MAX  - 0.5));
    ub = rec.y + rec.height + round(varphi*rec.height*((float)rand()/RAND_MAX - 0.5));
    vb = rec.x + rec.width + round(varphi*rec.width*((float)rand()/RAND_MAX  - 0.5));

    // check limits
    if (va<0){va = 0;}
    if (ua<0){ua = 0;}
    if (vb>=img.cols){vb = img.cols-1;}
    if (ub>=img.rows){ub = img.rows-1;}

    // check min. size
    if ((vb-va)>minSize && (ub-ua)>minSize){

      // new bounding box
      newRec = cvRect(va, ua, vb-va, ub-ua);

      // image patch using ROI
      cv::Mat patch = cv::Mat(cvSize(objWidth, objHeight), CV_8UC3);
      cv::resize(img(newRec), patch, cvSize(objWidth, objHeight));

      // fern maps (convolve ferns with the image)
      cv::Mat fernMaps = fun_fern_maps(patch, RFs);

      // update classifier
      clfr->fun_update(fernMaps, negLabel);

    }
  }
};

// update classifiers
void fun_update_classifiers(clsParameters* prms, clsClassifierSet* clfrs, clsRFs* RFs, cv::Mat img, clsDetectionSet* detSet, cv::Mat labels, cv::Mat& Mc, cv::Mat& Mq){
  // inputs:
  // prms: program parameters
  // clfr: classifier
  // RFs: fern parameters
  // img: image (I)
  // detSet: detection set
  // labels: samples labels (y)
  // Mc: correct predictions vector
  // Mq: human assistance vector

  // parameters
  float beta = prms->fun_get_threshold();  // classifier threshold (beta)
  float varphi  = prms->fun_get_image_shift();  // image shift factor (varphi)
  int recThick = prms->fun_get_rectangle_thickness();  // detection rectangle thickness
  int updSamps = prms->fun_get_num_update_samples();  // number of new (updating) samples (Nu)

  // variables
  int va = 10;  // human assistance text location in x
  int ua = img.rows - 60;  // human assitance text location in y
  int vb = 10;  // question text location in x
  int ub = 60;  // question text location in y
  char key;  // keyword key
  CvRect rec;  // image box
  char text[50];  // message
  int timePause = 3000;  // time for human answer
  CvScalar color;  // detection color
  cv::Mat copImg;   // copy image
  float posValue = 1.0;  // positive value (y=+1)
  float negValue = -1.0;  // negative value (y=-1)
  char text1[50] = "Human Assistance";  // message
  char text2[50] = "This detection is correct [y/n]?";  // message
  float score,label;  // detection score and sample label (y)
  clsDetection* det;  // detection
  clsClassifier* clfr;  // classifier pointer
  int objHeight,objWidth;  // object size (Bu,Bv)
  int u1,v1,u2,v2,k,numDets;  // image coordinates, classifier index (k), and num. detections

  // text font
  float textFont = prms->fun_get_text_font();

  // num. detections
  numDets = detSet->fun_get_num_detections();

  // image copy: for learning without text messages
  img.copyTo(copImg);

  // check num. detections
  if ((numDets>0) && (updSamps>0)){

    // pointer
    float* McPtr = Mc.ptr<float>(0);  // num. correct detections (Mc)
    float* MqPtr = Mq.ptr<float>(0);  // num. human assistances (Mq)
    float* labPtr = labels.ptr<float>(0); // labels (y)

    // detections
    for (int iter=0; iter<numDets; iter++){

      // detection color
      color = cvScalar(255, 255, 255);

      // current detection
      det = detSet->fun_get_detection(iter);

      // detection values
      det->fun_get_values(u1, v1, u2, v2, score, k);

      // current classifier and object size
      clfr = clfrs->fun_get_classifier(k);
      clfr->fun_get_object_size(objHeight, objWidth);

      // new bounding box
      rec = cvRect(v1, u1, v2-v1, u2-u1);

      // image patch using ROI
      cv::Mat patch = cv::Mat(cvSize(objWidth, objHeight), CV_8UC3);
      cv::resize(img(rec), patch, cvSize(objWidth, objHeight));

      // fern maps (convolve ferns with the image)
      cv::Mat fernMaps = fun_fern_maps(patch, RFs);

      // sample label (y)
      label = *(labPtr + iter);

      // interactive learning: update classifier using human assistance
      if (label==0.5){

        // human assistance
        fun_draw_message(prms, img, text1, cvPoint(va, ua), cvScalar(255, 255, 0));
        fun_draw_message(prms, img, text2, cvPoint(vb, ub), cvScalar(255, 255, 0));

        // detection rectangle
        cv::rectangle(img, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(0, 0, 0), recThick+3);
        cv::rectangle(img, cvPoint(v1, u1), cvPoint(v2, u2), cvScalar(255, 255, 0), recThick);
        cv::imshow("detection", img);

        // user answer
        key = cv::waitKey(timePause);
        switch(key){
          case 'y':
            // set color
            color = cvScalar(0, 255, 0);

            // correct label (y=+1)
            *(labPtr + iter) = 1.0;

            // update classifier with current positive sample
            clfr->fun_update(fernMaps, posValue);

            // update classifier using a set of random positive samples (Nu)
            fun_update_positive_samples(copImg, rec, clfr, RFs, updSamps, varphi);

            // update num. correct detections rate (M^c_k)
            if (score > beta)
              *(McPtr + k)+=1;

            // update num. human assistances (M^q_k)
            *(MqPtr + k)+=1;

            break;

          case 'n':
            // set color
            color = cvScalar(20, 20, 20);

            // correct label (y=-1)
            *(labPtr + iter) = -1.0;

            // update classifier with current negative sample
            clfr->fun_update(fernMaps, negValue);

            // update using a set of negative samples (Nu)
            fun_update_negative_samples_fps(copImg, rec, clfr, RFs, updSamps, varphi);

            // update num. correct detections rate (M^c_k)
            if (score <= beta)
              *(McPtr + k)+=1;

            // update num. human assistances (M^q_k)
            *(MqPtr + k)+=1;

            break;

        }

        // detection rectangle
        cv::rectangle(img, cvPoint(v1, u1), cvPoint(v2, u2), color, recThick);
        cv::imshow("detection", img);
        cv::waitKey(33);

      }
    }
  }
};

// train classifier
void fun_train_classifier(clsParameters* prms, clsClassifier* clfr, clsRFs* RFs, cv::Mat img, CvRect rec){

  // parameters
  float varphi = prms->fun_get_image_shift();  // image shift factor
  int trnSamps = prms->fun_get_num_train_samples();  // num. initial (training) samples (Nt)
  int objModSize = 100;  // object model size

  // variables
  float ratio;  // aspect ratio
  int objHeight,objWidth;  // object size (Bu,Bv)

  // object size
  ratio = (float)rec.width/rec.height;  // aspect ratio
  objHeight = prms->fun_get_object_height();  // standard object height (Bu)
  objWidth = (int)round(ratio*objHeight);  // object width (Bv)

  // set the object size (Bu,Bv)
  clfr->fun_set_object_size(objHeight, objWidth);

  // initialize the classifier using the current object size
  clfr->fun_set_threshold(prms->fun_get_threshold());  // classifier threshold (beta)
  clfr->fun_compute(prms->fun_get_num_ferns(), RFs->fun_get_pool_size(), RFs->fun_get_num_features(), RFs->fun_get_fern_size());  // copute classifier
  clfr->fun_print();  // print classifier parameters

  // object model
  cv::Mat objModel = cv::Mat(cvSize(objModSize, objModSize), CV_8UC3);  // object model
  cv::resize(img(rec), objModel, cvSize(objModSize, objModSize));  // copy object model
  clfr->objModel = objModel;  // save object model in the classifier

  // update classifier with positive samples (Nt)
  fun_update_positive_samples(img, rec, clfr, RFs, trnSamps, varphi);

  // update classifier with negative samples (Nt)
  fun_update_negative_samples_rnd(img, rec, clfr, RFs, trnSamps, varphi);

};

// train step: train classifier
void fun_train_step(clsParameters* prms, clsClassifierSet* clfrs, clsRFs* RFs, cv::Mat img, CvRect rec){

  // variables
  clsClassifier* clfr;  // classifier
  int numClfrs,numMaxClfrs;  // num. classifiers (K)

  // num. classifiers
  numClfrs = clfrs->fun_get_num_classifiers();  // num. trained classifiers (K)
  numMaxClfrs = clfrs->fun_get_num_max_classifiers();  // num. max. classifiers

  // check num. max. classifiers
  if (numClfrs>=numMaxClfrs){
    cout << "Warning: the number maximum of classifiers has been achieved" << endl;;
    return;
  }

  // current classifier
  clfr = clfrs->fun_get_classifier(numClfrs);

  // increment num. classifiers (K)
  numClfrs++;

  // message
  cout << "\n*********************************************************" << endl;
  cout << "new classifier -> " << numClfrs << endl;
  cout << "*********************************************************" << endl;

  // set num. classifiers
  clfrs->fun_set_num_classifiers(numClfrs);

  // train classifier
  fun_train_classifier(prms, clfr, RFs, img, rec);

};

// test step: detection and online learning
void fun_test_step(clsParameters* prms, clsClassifierSet* clfrs, clsRFs* RFs, cv::Mat img, clsDetectionSet* detSet){

  //num. classifiers (K)
  int numClfrs = clfrs->fun_get_num_classifiers();

  // object detections 
  if (numClfrs>0) fun_detect(prms, clfrs, RFs, img, detSet);

};

