
#ifndef detector_functions_h
#define detector_functions_h

#include "detector_functions.h"
#include "detector_classes.h"

// save image
void fun_save_image(cv::Mat img, long int number);

// draw message
void fun_draw_message(clsParameters* prms, cv::Mat img, char* text, CvPoint location, CvScalar textColor);

// draw rectangle
void fun_draw_rectangle(clsParameters* prms, cv::Mat img, CvRect rec, CvScalar recColor);

// show frame
void fun_show_frame(clsParameters* prms, cv::Mat img, int contFrame, double fps);

// detection score
void fun_detection_score(clsParameters*prms, cv::Mat img, clsDetectionSet* detSet);

// show image patch
void fun_show_image_patch(clsParameters* prms, cv::Mat img, clsDetectionSet* detSet);

// draw object model
void fun_draw_object_model(cv::Mat img, cv::Mat objModel, int objSize, int u, int v, CvScalar recColor);

// draw detections
void fun_draw_detections(clsParameters* prms, clsClassifierSet* classifiers, cv::Mat img, clsDetectionSet* detSet, cv::Mat labels);

// image equalization
void fun_image_equalization(cv::Mat img);

// update uncertainty thresholds (thetas)
void fun_update_uncertainty_thresholds(cv::Mat& thetas, cv::Mat Mc, cv::Mat Mq, int numMaxClfrs, float xi);

// show human-assisted learning rates
void fun_show_learning_results(clsParameters* prms, cv::Mat &img, cv::Mat thetas, cv::Mat Mc, cv::Mat Mq, int k, long int contFrame);

// detection labels
cv::Mat fun_detection_labels(clsParameters* prms, clsDetectionSet* detSet, cv::Mat thetas);

// fern maps
cv::Mat fun_fern_maps(cv::Mat img, clsRFs* RFs);
cv::Mat fun_detection_labels_real(clsParameters* prms, clsDetectionSet* detSet, cv::Mat thetas);
// scanning window
void fun_scanning_window(cv::Mat fernMaps, clsClassifierSet* classifiers, clsDetectionSet* detSet);

// object detection
void fun_detect(clsParameters* prms, clsClassifierSet* classifiers, clsRFs* RFs, cv::Mat img, clsDetectionSet* detSet);

// update classifier using positive samples
void fun_update_positive_samples(cv::Mat img, CvRect box, clsClassifier* clfr, clsRFs* RFs, int updSamps, float varphi);

// update classifier using negative samples (random samples)
void fun_update_negative_samples_rnd(cv::Mat img, CvRect box, clsClassifier* clfr, clsRFs* RFs, int updSamps, float varphi);

// update classifier using negative samples (false positive samples)
void fun_update_negative_samples_fps(cv::Mat img, CvRect box, clsClassifier* clfr, clsRFs* RFs, int updSamps, float varphi);

// update classifiers
void fun_update_classifiers(clsParameters* prms, clsClassifierSet* clfrs, clsRFs* RFs, cv::Mat img, clsDetectionSet* detSet, cv::Mat labels, cv::Mat& Mc, cv::Mat& Mq);

// train classifier
void fun_train_classifier(clsParameters* prms, clsClassifier* clfr, clsRFs* RFs, cv::Mat img, CvRect box);

// train step: train initial classifier
void fun_train_step(clsParameters* prms, clsClassifierSet* clfrs, clsRFs* RFs, cv::Mat img, CvRect box);

// test step: detection and online learning
void fun_test_step(clsParameters* prms, clsClassifierSet* clfr, clsRFs* RFs, cv::Mat img, clsDetectionSet* detSet);

#endif
