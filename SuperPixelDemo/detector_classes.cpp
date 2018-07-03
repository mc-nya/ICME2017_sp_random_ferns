#include"headfile.h"
#include <list>
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "detector_classes.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

// clsParameters
// constructor
clsParameters::clsParameters(){
  // initialize default parameters
  fun_init();
};
// destructor
clsParameters::~clsParameters(){
};
// initialize default parameters
void clsParameters::fun_init(){
  // default values
  this->xi = 1.0;  // sensitivity learning parameter (xi)
  this->beta = 0.58;  // classifier threshold (beta)
  this->varphi = 0.15;  // image shift (varphi)
  this->numClfrs = 1;  // num. classifiers (K)
  this->numFerns = 300;  // num. random ferns (J)
  this->numFeats = 8;  // num. features (M)
  this->poolSize = 10;  // num. ferns parameters (R)
  this->fernSize = 8;  // fern size (S)
  this->trnSamps = 100;  // num. initial (training) samples (N_t)
  this->updSamps = 2;  // num. new (updating) samples (N_u)
  this->recThick = 4;  // rectangle thickness
  this->textFont = 2.0;  // text font
  this->imgEqual = false; // image equalization
  this->imgChans = 3;  // num. image channels (C)
  this->saveImgs = false;  // save images in disk
  this->visuMode = 1;  // visualization mode
  this->imgWidth = 640;  // image width
  this->imgHeight = 480;  // image height
  this->objHeight = 18; // standard object height (B_u)
  this->minCellSize = 5;  // min. image cell size
  this->maxCellSize = 20;  // max. image cell size

  // colors
  //this->colors.push_back(cvScalar(0, 255, 0)); // reserved
  this->colors.push_back(cvScalar(0, 0, 255));
  this->colors.push_back(cvScalar(255, 255, 0));
  this->colors.push_back(cvScalar(255, 0, 255));
  this->colors.push_back(cvScalar(0, 255, 255));
  this->colors.push_back(cvScalar(200, 100, 0));
  this->colors.push_back(cvScalar(200, 0, 100));
  this->colors.push_back(cvScalar(0, 200, 100));
  this->colors.push_back(cvScalar(100, 200, 0));
  this->colors.push_back(cvScalar(100, 0, 200));
  this->colors.push_back(cvScalar(0, 100, 200));
  this->colors.push_back(cvScalar(150, 50, 50));
  this->colors.push_back(cvScalar(50, 150, 50));
  this->colors.push_back(cvScalar(50, 50, 150));
  this->colors.push_back(cvScalar(100, 100, 0));
  this->colors.push_back(cvScalar(0, 100, 100));
  this->colors.push_back(cvScalar(100, 0, 100));
  this->colors.push_back(cvScalar(100, 0, 0));
  this->colors.push_back(cvScalar(0, 100, 0));
  this->colors.push_back(cvScalar(0, 0, 100));
  this->colors.push_back(cvScalar(255, 0, 0));

  // parameter files
  this->filePath = "E:\\lab\\SuperPixelDemo\\SuperPixelDemo\\parameters.txt";

  };
// get sensitivity learning rate (xi)
float clsParameters::fun_get_learning_rate(){
  return this->xi;
};
// get num. random ferns (J)
int clsParameters::fun_get_num_ferns(){
  return this->numFerns;
};
// get num. binary features (M)
int clsParameters::fun_get_num_features(){
  return this->numFeats;
};
// get fern size (S)
int clsParameters::fun_get_fern_size(){
  return this->fernSize;
};
// get num. object classifiers (K)
int clsParameters::fun_get_num_classifiers(){
  return this->numClfrs;
};
// get fern pool size (R)
int clsParameters::fun_get_pool_size(){
  return this->poolSize;
};
// get standard object height (Bu)
int clsParameters::fun_get_object_height(){
  return this->objHeight;
};
// get classifier threshold (beta)
float clsParameters::fun_get_threshold(){
  return this->beta;
};
// get image shift (varphi)
float clsParameters::fun_get_image_shift(){
  return this->varphi;
};
// get thickness
int clsParameters::fun_get_rectangle_thickness(){
  return this->recThick;
};
// get num. image channels (C)
int clsParameters::fun_get_num_image_channels(){
  return this->imgChans;
};
// get image width (Iv)
int clsParameters::fun_get_image_width(){
  return this->imgWidth;
};
// get image height (Iu)
int clsParameters::fun_get_image_height(){
  return this->imgHeight;
};
// get min. cell size
int clsParameters::fun_get_min_cell_size(){
  return this->minCellSize;
};
// get max. cell size
int clsParameters::fun_get_max_cell_size(){
  return this->maxCellSize;
};
// get text font
float clsParameters::fun_get_text_font(){
  return this->textFont;
};
// get flag for save images in disk
bool clsParameters::fun_get_save_images(){
  return this->saveImgs;
}
// get visualization mode
int clsParameters::fun_get_visualization_mode(){
  return this->visuMode;
}
// set visualization mode
void clsParameters::fun_set_visualization_mode(int mode){
  this->visuMode = mode;
}
// get the number of initial (training) samples (Nt)
int clsParameters::fun_get_num_train_samples(){
  return this->trnSamps;
};
// get the number of new (updating) samples (Nu)
int clsParameters::fun_get_num_update_samples(){
  return this->updSamps;
};
// get image equalization
bool clsParameters::fun_image_equalization(){
  return imgEqual;
};
// get specific color
CvScalar clsParameters::fun_get_color(int index){
  // check color index
  if (index>=this->colors.size()) {
    // message
    cout << "Warning: only 20 colors are defined" << endl;
    // default color
    index = 0;
  }
  // return color
  return this->colors.at(index);
}
// load parameters
void clsParameters::fun_load(){

  // variables
  char* tmp;
  char buffer[128];
  FILE *ptr = NULL;
  int fontThick = 3;
  int bool1,bool2,msm;

  // open file
  if ((ptr = fopen(this->filePath, "r"))!= NULL){

    // read parameters
    tmp = fgets(buffer, 128, ptr);
    msm = fscanf(ptr, "%s", buffer);
    msm = fscanf(ptr, "%s %d", buffer, &this->numClfrs);
    msm = fscanf(ptr, "%s %d", buffer, &this->numFerns);
    msm = fscanf(ptr, "%s %d", buffer, &this->numFeats);
    msm = fscanf(ptr, "%s %d", buffer, &this->poolSize);
    msm = fscanf(ptr, "%s %d", buffer, &this->fernSize);
    msm = fscanf(ptr, "%s %d", buffer, &this->imgChans);
    msm = fscanf(ptr, "%s %d", buffer, &this->objHeight);
    msm = fscanf(ptr, "%s %f", buffer, &this->xi);
    msm = fscanf(ptr, "%s %f", buffer, &this->beta);
    msm = fscanf(ptr, "%s %d", buffer, &this->trnSamps);
    msm = fscanf(ptr, "%s %d", buffer, &this->updSamps);
    msm = fscanf(ptr, "%s %d", buffer, &this->imgWidth);
    msm = fscanf(ptr, "%s %d", buffer, &this->imgHeight);
    msm = fscanf(ptr, "%s %f", buffer, &this->varphi);
    msm = fscanf(ptr, "%s %d", buffer, &this->minCellSize);
    msm = fscanf(ptr, "%s %d", buffer, &this->maxCellSize);
    msm = fscanf(ptr, "%s %d", buffer, &bool1);
    msm = fscanf(ptr, "%s %f", buffer, &this->textFont);
    msm = fscanf(ptr, "%s %d", buffer, &this->recThick);
    msm = fscanf(ptr, "%s %d", buffer, &bool2);
    msm = fscanf(ptr, "%s %d", buffer, &this->visuMode);

    // bool variables
    this->imgEqual = bool1;
    this->saveImgs = bool2;

    // close file pointer
    fclose(ptr);

  }
  else {
    cout << "ERROR : not input parameter file" << endl; 
    exit(0);
  }
};
// print parameters
void clsParameters::fun_print(){

  cout << "\n*********************************************************" << endl;
  cout << "*               Online Multi-Object Detection           *" << endl;
  cout << "*                    Michael Villamizar                 *" << endl;
  cout << "*                 mvillami.at.iri.upc.edu               *" << endl;
  cout << "*                          2015                         *" << endl;
  cout << "*********************************************************" << endl;

  cout << "\n*********************************************************" << endl;
  cout << "* Program parameters :" << endl;
  cout << "* image size -> " << this->imgHeight << "x" << this->imgWidth << endl;
  cout << "* object height -> " << this->objHeight << endl;
  cout << "* num. image channels -> " << this->imgChans << endl;
  cout << "* num. object classifiers -> " << this->numClfrs << endl;
  cout << "* num. random ferns -> " << this->numFerns << endl;
  cout << "* num. ferns parameters -> " << this->poolSize << endl;
  cout << "* num. features -> " << this->numFeats << endl;
  cout << "* fern size -> " << this->fernSize << "x" << this->fernSize << endl;
  cout << "* classfier threshold -> " << this->beta << endl;
  cout << "* num. training samples -> " << this->trnSamps << endl;
  cout << "* num. updating samples -> " << this->updSamps << endl;
  cout << "* min. cell size -> " << this->minCellSize << endl;
  cout << "* max. cell size -> " << this->maxCellSize << endl;
  cout << "* image shift -> " << this->varphi << endl;
  cout << "* rec. thickness -> " << this->recThick<< endl;
  cout << "*********************************************************" << endl;
};




// clsRFs
// constructor
clsRFs::clsRFs(){
  // initialize default parameters
  this->poolSize = 5;  // num. ferns parameters (R)
  this->fernSize = 4;  // fern size (S)
  this->numFeats = 5;  // num. binary features per fern (M)
  this->fernDepth = 3;  // fern depth (num. image channels (C))
};
// destructor
clsRFs::~clsRFs(){
};
// get num. ferns parameters (R)
int clsRFs::fun_get_pool_size(){
  return this->poolSize;
};
// get num. binary features (M)
int clsRFs::fun_get_num_features(){
  return this->numFeats;
};
// get fern size (S)
int clsRFs::fun_get_fern_size(){
  return this->fernSize;
};
// get fern depth (image channels (C))
int clsRFs::fun_get_fern_depth(){
  return this->fernDepth;
};
// get ferns parameters data
cv::Mat clsRFs::fun_get_data(){
  return this->data;
};
// compute random ferns
void clsRFs::fun_compute(int R, int M, int S, int C){
  // inputs:
  // R: num. ferns parameters (pool size)
  // M: num. binary features
  // S: fern size
  // C: num. image channels (e.g color channels)

  // variables
  int ua,va,ca,ub,vb,cb;

  // set random ferns values
  this->poolSize = R;  // num. ferns parameters
  this->numFeats = M;  // num. binary features
  this->fernSize = S;  // spatial fern size
  this->fernDepth = C;  // fern depth (num. image channels)

  // allocate data
  this->data = cv::Mat(R, M, CV_8UC(6));

  // pointer to ferns data
  unsigned char *dataPtr = (unsigned char*)(data.data);

  // random ferns parameters
  for (int r = 0; r<R; r++ ){
    for (int m = 0; m<M; m++){

      // feature a coordinates
      ua = floor(S*((double)rand()/RAND_MAX));
      va = floor(S*((double)rand()/RAND_MAX));
      ca = floor(C*((double)rand()/RAND_MAX));

      // feature b coordinates
      ub = floor(S*((double)rand()/RAND_MAX));
      vb = floor(S*((double)rand()/RAND_MAX));
      cb = floor(C*((double)rand()/RAND_MAX));

      // save
      *(dataPtr + r*M*6 + m*6 + 0) = ua;
      *(dataPtr + r*M*6 + m*6 + 1) = va;
      *(dataPtr + r*M*6 + m*6 + 2) = ca;
      *(dataPtr + r*M*6 + m*6 + 3) = ub;
      *(dataPtr + r*M*6 + m*6 + 4) = vb;
      *(dataPtr + r*M*6 + m*6 + 5) = cb;
    }
  }
};
// print random ferns values
void clsRFs::fun_print(){
  cout << "\n*********************************************************" << endl;
  cout << "* Random Fern Parameters : " << endl;
  cout << "* num. ferns parameters -> " << this->poolSize << endl;
  cout << "* num. binary features -> " << this->numFeats << endl;
  cout << "* fern size ->  " << this->fernSize << "x" << this->fernSize << endl;
  cout << "* fern depth ->  " << this->fernDepth << endl;
  cout << "*********************************************************\n" << endl;
};




// clsClassifier
// constructor
clsClassifier::clsClassifier(){
  // initialize default parameters
  this->beta = 0.5;  // classifier threshold (beta)
  this->numBins = 128;  // num. histogram bins
  this->numFerns = 100;  // num. random ferns (J)
  this->objWidth = 24;  // object width (Bv)
  this->objHeight = 24;  // object height (Bu)
};
// destructor
clsClassifier::~clsClassifier(){
};
// get threshold (beta)
float clsClassifier::fun_get_threshold(){
  return this->beta;
};
// set threshold (beta)
void clsClassifier::fun_set_threshold(float thr){
  // inputs:
  // thr: classifier threshold
  this->beta = thr;
};
// set object size (Bu,Bv)
void clsClassifier::fun_set_object_size(int Bu, int Bv){
  // inputs:
  // Bu: object size (image) in u (y)
  // Bv: object size (image) in v (x)
  this->objHeight = Bu;
  this->objWidth = Bv;
};
// get object size (Bu,Bv)
void clsClassifier::fun_get_object_size(int& Bu, int& Bv){
  // inputs:
  // Bu: object size (image) in u (y)
  // Bv: object size (image) in v (x)
  Bu = this->objHeight;
  Bv = this->objWidth;
};
// get num. random ferns (J)
int clsClassifier::fun_get_num_ferns(){
  return this->numFerns;
};
// get classifier data
cv::Mat clsClassifier::fun_get_data(){
  return this->data;
};
// get positive fern distributions
cv::Mat clsClassifier::fun_get_posHstms(){
  return this->posHstms;
};
// get negative fern distributions
cv::Mat clsClassifier::fun_get_negHstms(){
  return this->negHstms;
};
// get ratio of fern distributions
cv::Mat clsClassifier::fun_get_ratHstms(){
  return this->ratHstms;
};
// compute the classifier
void clsClassifier::fun_compute(int J, int R, int M, int S){
  // inputs:
  // J: num. random ferns
  // R: fern pool size
  // M: num. binary features
  // S: fern size
	
  // variables
  int u,v,w;  // fern location (u,v) and fern parameter (omega)
  int Bu = this->objHeight;  // object height
  int Bv = this->objWidth;  // object width

  // set the number of ferns and histogram bins
  this->numFerns = J;
  this->numBins = (int)pow(2, M);
  //cv::Mat temp(J, 3, CV_8UC1);
  //this->data = temp.clone();
  // classifier data (random ferns parameters)
  this->data = cv::Mat(J, 3, CV_8UC1);
  //this->data = cv::Mat::zeros(J, 3, CV_8UC1);
  
  // pointer to data
  unsigned char *dataPtr = (unsigned char*)(this->data.data);
  srand((unsigned)time(NULL));
  // random ferns
  for (int j = 0; j<J; j++){
	  
    // fern parameters
    u = floor((Bu-S)*((float)rand()/RAND_MAX));
    v = floor((Bv-S)*((float)rand()/RAND_MAX));
    w = floor(R*((float)rand()/RAND_MAX));

    // save
    *(dataPtr + j*3 + 0) = u;  // fern location in u
    *(dataPtr + j*3 + 1) = v;  // fern location in v
    *(dataPtr + j*3 + 2) = w;  // features parameters (omega)

  }

  // distributions
  this->posHstms = cv::Mat(this->numFerns, this->numBins, CV_32FC1, cv::Scalar::all(1.0));
  this->negHstms = cv::Mat(this->numFerns, this->numBins, CV_32FC1, cv::Scalar::all(1.0));
  this->ratHstms = cv::Mat(this->numFerns, this->numBins, CV_32FC1, cv::Scalar::all(0.5));

};
// update the classifier
void clsClassifier::fun_update(cv::Mat &fernMaps, float label){
  // inputs:
  // fernMaps: ferns outputs over an image sample x
  // label: class label (y = {+1,-1}) for the sample x

  // variables
  int u,v,w,z;  // fern location (u,v), fern parameters (omega), fern output (z)
  float pos,neg;  // positive and negative values

  // fern map size
  int poolSize = fernMaps.channels();  // pool size (num. fern features parameters)
  int imgWidth = fernMaps.cols;  // image width (Iv)
  int imgHeight = fernMaps.rows;  // image height (Iu)

  // pointer to classifier data, ferns maps, and positive, negative and ratio ferns distributions
  unsigned char *dataPtr = (unsigned char*)(this->data.data);
  unsigned short *mapsPtr = (unsigned short*)(fernMaps.data);
  float* posPtr = this->posHstms.ptr<float>(0);
  float* negPtr = this->negHstms.ptr<float>(0);
  float* ratPtr = this->ratHstms.ptr<float>(0);

  // update random ferns
  for (int j=0; j<this->numFerns; j++){

    // fern parameters
    u = (int)*(dataPtr + j*3 + 0);  // location u
    v = (int)*(dataPtr + j*3 + 1);  // location v
    w = (int)*(dataPtr + j*3 + 2);  // features parameters

    // fern output
    z = (int)*(mapsPtr + u*imgWidth*poolSize + v*poolSize + w);

    // update positive fern distribution
    if (label==1.0)
      *(posPtr + j*this->numBins + z)+= 1.0;

    // update negative fern distribution
    if (label==-1.0)
      *(negPtr + j*this->numBins + z)+= 1.0;

    // ratio fern distribution
    pos = *(posPtr + j*this->numBins + z);
    neg = *(negPtr + j*this->numBins + z);
    *(ratPtr + j*this->numBins + z) = pos/(pos+neg);

  }
};
// print classifier parameters
void clsClassifier::fun_print(){
  cout << "\n*********************************************************" << endl;
  cout << "* Classifier Parameters : " << endl;
  cout << "* object size ->  " << this->objHeight << "x" << this->objWidth << endl;
  cout << "* num. random ferns -> " << this->numFerns << endl;
  cout << "* classifier threshold -> " << this->beta << endl;
  cout << "*********************************************************\n" << endl;
};




// clsClassifierSet
// constructor
clsClassifierSet::clsClassifierSet(){
  // initialize
  fun_init();
};
// Destructor
clsClassifierSet::~clsClassifierSet(){
  // relase memory
  fun_release();
};
// initialize
void clsClassifierSet::fun_init(){

  // default values
  this->numClfrs = 0;  // num. classifiers (K) 
  this->numMaxClfrs = 30;  // num. max. classifiers 

  // array of classifiers
  this->clfrs = new clsClassifier[this->numMaxClfrs];

};
// release
void clsClassifierSet::fun_release(){
  delete[]this->clfrs;
};
// get num. classifiers (K)
int clsClassifierSet::fun_get_num_classifiers(){
  return this->numClfrs;
};
// get num. max. classifiers
int clsClassifierSet::fun_get_num_max_classifiers(){
  return this->numMaxClfrs;
};
// get classifier
clsClassifier* clsClassifierSet::fun_get_classifier(int k){
  // inputs:
  // k: classifier index in the list of all classifiers
  clsClassifier *clfr = &this->clfrs[k];
  return clfr;
};
// set num. classifiers (K)
void clsClassifierSet::fun_set_num_classifiers(int K){
  // inputs:
  // K: num. classifiers
  this->numClfrs = K;
};




// clsDetection
// constructor
clsDetection::clsDetection(){
  // initialize default values
  fun_init();
};
// destructor
clsDetection::~clsDetection(){
};
// initialize with default values
void clsDetection::fun_init(){
  // default parameters
  this->u1 = 0; // detection location
  this->v1 = 0;
  this->u2 = 0;
  this->v2 = 0;
  this->ide = 0;  // detection identifier
  this->score = 0;  // detection score
};
// set values
void clsDetection::fun_set_values(int ua, int va, int ub, int vb, float score, int ide) {
  // inputs:
  // ua,va: top-left detection location
  // ub,vb: bottom-right detection location
  // score: detection score
  // ide: detection ide

  // set values
  this->u1 = ua;  // detection location
  this->v1 = va;
  this->u2 = ub;
  this->v2 = vb;
  this->ide = ide;  // detection identifier
  this->score = score;  // detection score
};
// get values
void clsDetection::fun_get_values (int &ua, int &va, int &ub, int &vb, float &score, int &ide) {
  // inputs:
  // ua,va: top-left detection location
  // ub,vb: bottom-right detection location
  // score: detection score
  // ide: detection ide

  // get values
  ua = this->u1;  // detection location
  va = this->v1;
  ub = this->u2;
  vb = this->v2;
  ide = this->ide;  // detection identifier
  score = this->score;  // detection score
};



// clsDetectionSet
// constructor
clsDetectionSet::clsDetectionSet(){
  // initialize
  fun_init();
};
// Destructor
clsDetectionSet::~clsDetectionSet(){
  // relase memory
  fun_release();
};
// initialize
void clsDetectionSet::fun_init(){

  // default values
  this->numDets = 0;  // num. detections
  this->numMaxDets = 1000;  // num. max. detections

  // array of detections
  this->dets = new clsDetection[this->numMaxDets];
};
// release
void clsDetectionSet::fun_release(){
  delete[]this->dets;
};
// get num. detections
int clsDetectionSet::fun_get_num_detections(){
  return this->numDets;
};
// set num. detections
void clsDetectionSet::fun_set_num_detections(int value){
  // inputs:
  // value: num. detections
  this->numDets = value;
};
// get num. max. detections
int clsDetectionSet::fun_get_num_max_detections(){
  return this->numMaxDets;
};
// get detection
clsDetection* clsDetectionSet::fun_get_detection(int index){
  // inputs:
  // index: detection index in the list of all detections
  clsDetection* det = &this->dets[index];
  return det;
};
// set a new detection
void clsDetectionSet::fun_set_detection(clsDetection* det, int index){
  // inputs:
  // det: detection
  // index: detection index

  // variable
  float score;
  int u1,v1,u2,v2,ide;

  // get input detection values
  det->fun_get_values(u1, v1, u2, v2, score, ide);

  // set detection values
  this->dets[index].fun_set_values(u1, v1, u2, v2, score, ide);
};
// scaling
void clsDetectionSet::fun_scaling(float scaleFactor){
  // inputs:
  // scale factor: scaling factor for detections coordinates

  // variables
  float score;
  int u1,v1,u2,v2,ide;

  // scaling detection coordinates
  for (int iter=0; iter<this->numDets; iter++){

    // scaling spatial coordinates
    this->dets[iter].fun_get_values(u1, v1, u2, v2, score, ide);
    u1 = (int)round(u1*scaleFactor);
    v1 = (int)round(v1*scaleFactor);
    u2 = (int)round(u2*scaleFactor);
    v2 = (int)round(v2*scaleFactor);
    this->dets[iter].fun_set_values(u1, v1, u2, v2, score, ide);
  }
};
// add detections
void clsDetectionSet::fun_add_detections(clsDetectionSet* newDets){
  // inputs:
  // newDets: new detections

  // variables
  int numNewDets;

  // num. new detections
  numNewDets = newDets->fun_get_num_detections();

  // check
  if (this->numDets + numNewDets >= this->numMaxDets){
    cout << "ERROR: num. max. detections"  << endl;
  }
  else{
    // add detections
    for (int iter=0; iter<numNewDets; iter++){
      this->dets[this->numDets+iter] = newDets->dets[iter];
    }
    // updates num. detections 
    this->numDets += numNewDets;
  }
};
// maxima detection
void clsDetectionSet::fun_get_max_detection(clsDetection* maxDet){
  // inputs:
  // maxDet: maxima detection

  // variables
  int u1,v1,u2,v2,ide;
  float score,maxScore = 0;

  // detections
  for (int iter=0; iter<this->numDets; iter++){

    // get detection values
    this->dets[iter].fun_get_values(u1, v1, u2, v2, score, ide);

    // best score
    if (score>maxScore){

      // set max. detection
      maxDet->fun_set_values(u1, v1, u2, v2, score, ide);

      // update max. score
      maxScore = score;
    }
  }
};
// remove detections 
void clsDetectionSet::fun_remove_detections(clsDetection* det){
  // inputs:
  // det: detection

  // parameters
  float thr = 0.01;  // overlapping rate threshold 

  // variables
  int counter = 0;  // counter
  int ua,va,ub,vb;  // overlapping coordinates
  clsDetection empty;  // empty detection
  float score,rscore;  // detection scores
  int u1,v1,u2,v2,ide;  // detection coordinates
  float w,h,area,rarea;  // overlapping variables 
  int ru1,rv1,ru2,rv2,ride;  // reference detection coordinates 

  // num. initial detections
  this->numDets = fun_get_num_detections();

  // get reference detection values
  det->fun_get_values(ru1, rv1, ru2, rv2, rscore, ride);

  // reference area
  rarea = (ru2-ru1)*(rv2-rv1);

  // check each detection
  for (int iter=0; iter<this->numDets; iter++){

    // get detection values
    this->dets[iter].fun_get_values(u1, v1, u2, v2, score, ide);

    // overlapping values
    ua = max(ru1,u1);
    va = max(rv1,v1);
    ub = min(ru2,u2);
    vb = min(rv2,v2);
    w = vb-va;  // width
    h = ub-ua;  // height
    area = w*h;  // area

    // remove detections
    if (w<=0 || h<=0 || (area/rarea)<thr){
    // no intersection
      this->dets[counter].fun_set_values(u1, v1, u2, v2, score, ide);
      counter++;
    }
  }

  // delete detections
  for (int iter=counter; iter<this->numDets; iter++){
    this->dets[iter] = empty;
  }

  // update num. detections
  fun_set_num_detections(counter);

};
// non-maxima suppresion
void clsDetectionSet::fun_non_maxima_supression(){

  // variables
  int counter = 0;   // detection counter

  // temporal detection set
  clsDetectionSet* detSet = new clsDetectionSet;

  // num. initial detections
  this->numDets = fun_get_num_detections();

  // check detections
  if (this->numDets>0){

    // remove detections
    while (this->fun_get_num_detections()!=0){

      // max. detection
      clsDetection* maxDet = new clsDetection;

      // get max. detection
      this->fun_get_max_detection(maxDet);

      // get non-intersection detections
      this->fun_remove_detections(maxDet);

      // add detection: max. detection
      detSet->fun_set_detection(maxDet, counter);

      // update counter
      counter++;

      // set num. detections
      detSet->fun_set_num_detections(counter);

      // release
      delete maxDet;
    }
  }

  // add max. detections: non-maxima suppression
  this->fun_add_detections(detSet);

  // release
  delete detSet;
};



// clsII
// constructor
clsII::clsII(){
  // default values
  this->width = 0;  // integral image width
  this->height = 0;  // integral image height
  this->imgChans = 3;  // num. image feature channels (C)
  this->imgWidth = 0;  // image width (Iv)
  this->imgHeight = 0; // image height (Iu)
  this->cellSize = 1;  // cell size
};
// destructor
clsII::~clsII(){
};
// get integral image
cv::Mat clsII::fun_get_image() {
  return this->img;
};
// get image size
void clsII::fun_get_image_size(int& Iu, int& Iv) {
  // inputs:
  // Iu: image size in u (y)
  // Iv: image size in v (x)
  Iv = this->imgWidth;
  Iu = this->imgHeight;
};
// compute image
void clsII::fun_compute_image(int cellSize){
  // inputs:
  // cellSize: cell size for image pyramid

  // set values
  this->cellSize = cellSize;  // cell size
  this->imgWidth = (int)ceil((float)this->width/this->cellSize);  // image width (Iv)
  this->imgHeight = (int)ceil((float)this->height/this->cellSize);  // image height (Iu)

  // variables
  int W = this->width;  // integral image width
  int H = this->height;  // integral image height
  int C = this->imgChans;  // num. image channels (C)
  int D = this->cellSize;  // cell size
  int Iv = this->imgWidth;  // image width
  int Iu = this->imgHeight;  // image height
  int t,b,l,r;  // top,bottom,left and right values
  double tl,bl,tr,br,val;  // top-left,bottom-left,top-right,bottom-right,value

  // create image
  this->img = cv::Mat(cvSize(Iv,Iu), CV_8UC(C), cv::Scalar::all(0));

  // pointer to image and integral image data
  double* IIPtr = this->II.ptr<double>(0);
  unsigned char *imgPtr = (unsigned char*)(this->img.data);

  // scanning
  for (int u=0; u<Iu-1; u++) {

    // local coordinates
    t = u*D;  // top
    b = t + D;  // bottom

    for (int v=0; v<Iv-1; v++) {

      // local coordinates
      l = v*D;  // left
      r = l + D;  // right

      // check
      if (t<0 || b>=H || l<0 || r>=W)
        cout << "Warning: incorrect corner coordinates" << endl;

      for (int c=0; c<C; c++){

        // integral image values
        tl = *(IIPtr + t*W*C + l*C + c);  // top-left corner
        bl = *(IIPtr + b*W*C + l*C + c);  // bottom-left corner
        tr = *(IIPtr + t*W*C + r*C + c);  // top-right corner
        br = *(IIPtr + b*W*C + r*C + c);  // bottom-right corner

        // check
        if (tl<0 || bl<0 || tr<0 || br<0)
          cout << "Warning: incorrect corner values" << endl;

        // image value
        val = br + tl - tr - bl;
        val = (double)val/(D*D);

        // check (2 -> small error)
        if (val<0 || val>255+2)
          cout << "Warning: incorrect image value" << endl;

        // image
        *(img.data + u*Iv*C + v*C + c) = (int) val;

      }
    }
  }

};
// compute integral image
void clsII::fun_integral_image(cv::Mat image){
  // inputs:
  // image: input image

  // set integral image size
  this->width = image.cols;  // II width
  this->height = image.rows;  // II height
  this->imgChans = image.channels();  // num. image channels (C)

  // variables
  int W = this->width;  // II width
  int H = this->height;  // II height
  int C = this->imgChans;  // num. image channels
  double imgVal,val;  // image pixel value, value

  // create integral image (II)
  this->II = cv::Mat(H, W, CV_64FC(C), cv::Scalar::all(0));

  // pointers to image and integral image data
  double* IIPtr = II.ptr<double>(0);
  unsigned char *imgPtr = (unsigned char*)(image.data);

  // construction
  for (int u=0; u<H; u++){
    for (int v=0; v<W; v++){
      for (int c=0; c<C; c++){

        // image pixel value
        imgVal = *(imgPtr + u*W*C + v*C + c);

        // integral image values
        if (v>0){
          if (u>0){
            val = *(IIPtr + u*W*C + (v-1)*C + c) + *(IIPtr + (u-1)*W*C + v*C + c) - *(IIPtr + (u-1)*W*C + (v-1)*C + c);
          }
          else {
            val = *(IIPtr + u*W*C + (v-1)*C + c);
          }
        }
        else {
          if (u>0){
            val = *(IIPtr + (u-1)*W*C + v*C + c);
          }
          else {
            val = 0;
          }
        }

        // compute current pixel value in II
        *(IIPtr + u*W*C + v*C + c) = val + imgVal;

        // check
        if (val + imgVal > (u+1)*(v+1)*255)
          cout << "Warning incorrect integral image value" << endl;

      }
    }
  }
};
// compute integral image
void clsII::fun_release_image(){
  // release image
  this->img.release();
};








