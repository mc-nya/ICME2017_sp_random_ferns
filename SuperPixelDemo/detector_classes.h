#ifndef objectDetector_h
#define objectDetector_h

#include <list>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

// program parameters
class clsParameters{
  private:
    float xi;  // sensitivity learning rate (xi)
    float beta;  // classifier threshold (beta)
    int numClfrs;  // num. object classifiers (K)
    int numFerns;  // num. random ferns (J)
    int numFeats;  // num. binary features per fern (M)
    int poolSize;  // num. random fern paramaters (R)
    int fernSize;  // spatial fern size (S)
    int imgChans;  // num. image channels (C)
    int trnSamps;  // num. initial (training) samples (Nt)
    int updSamps;  // num. new (updating) samples (Nu)
    float varphi;  // image shift (varphi)
    int recThick;  // rectangle thickness
    int visuMode;  // visualization mode
    int imgWidth;  // image width (Iv)
    int imgHeight;  // image height (Iu)
    int objHeight;  // standard object height (Bu)
    bool imgEqual;  // image equalization
    bool saveImgs;  // save images in disk
    float textFont;  // text font
    int minCellSize;  // min. cell size
    int maxCellSize;  // max. cell size
    char const* filePath;  // file parameters path
    vector <CvScalar> colors;  // colors
  public:
    clsParameters(); // constructor
    ~clsParameters(); // destructor
    void fun_init();  // initialize variables
    void fun_load();  // load parameters from file
    void fun_print();  // print program parameters
    int fun_get_pool_size();  // return num. ferns parameters (R)
    int fun_get_num_ferns();  // return the number of random ferns (J)
    int fun_get_fern_size();  // return fern size (S)
    float fun_get_text_font();  // return the text font
    int fun_get_image_width(); // return image width (Iv)
    float fun_get_threshold();  // return the classifier threshold (beta)
    bool fun_get_save_images();  // get save images flag
    int fun_get_image_height();  // return image height (Iu)
    int fun_get_num_features();  // return num. features (M)
    float fun_get_image_shift();  // return the image shift rate (varphi)
    int fun_get_object_height();  // standard object height (Bu)
    int fun_get_min_cell_size();  // get min. cell size (image pyramid)
    int fun_get_max_cell_size();  // get max. cell size (image pyramid)
    bool fun_image_equalization();  // image equalization
    int fun_get_num_classifiers();  // return the number of object classifiers (K)
    float fun_get_learning_rate();  // return the sensitivity learning rate (xi)
    int fun_get_num_train_samples();  // return the number of initial samples (Nt)
    int fun_get_num_update_samples();  // return the number of new samples (Nu)
    int fun_get_num_image_channels();  // return num. image channels (C)
    int fun_get_visualization_mode();  // return visualization mode
    int fun_get_rectangle_thickness();  // return rectangle thickness
    void fun_set_visualization_mode(int mode);  // set visualization mode

    CvScalar fun_get_color(int index);  // return color
};

// random ferns (RFs)
class clsRFs {
  private:
    cv::Mat data;  // ferns parameters data
    int poolSize;  // num. random ferns parameters (R)
    int numFeats;  // num. binary features per fern (M)
    int fernSize;  // spatial fern size (S)
    int fernDepth;  // num. image channels (C)
  public:
    clsRFs();  // constructor
    ~clsRFs();  // destructor
    void fun_print();  // print fern parameters
    cv::Mat fun_get_data();  // get ferns data
    int fun_get_pool_size();  // return num. ferns paramaters (R)
    int fun_get_fern_size();  // return fern size (S)
    int fun_get_fern_depth();  // return fern depth (C)
    int fun_get_num_features();  // return num. binary features (M)
    void fun_compute(int R, int M, int S, int C);  // create random ferns
};

// classifier
class clsClassifier {
  private:
    float beta;  // classifier threshold (beta)
    int numBins;  // number of histogram bins
    cv::Mat data;  // classifier data
    int numFerns;  // number of random ferns (J)
    int objWidth;  // object image width (Bv)
    int objHeight;  // object image height (Bu)
    cv::Mat posHstms;  // positive fern histograms
    cv::Mat negHstms;  // negative fern histograms
    cv::Mat ratHstms;  // ratio of fern histograms
  public:
    clsClassifier();  // constructor
    ~clsClassifier();  // destructor
    void fun_print();  // print classifier parameters
    cv::Mat objModel;  // object model (image)
    cv::Mat fun_get_data();  // pointer to classifier data
    int fun_get_num_ferns();  // get num. random ferns (J)
    float fun_get_threshold();  // return classifier threshold (beta)
    cv::Mat fun_get_posHstms();  // pointer to positive fern distributions
    cv::Mat fun_get_negHstms();  // pointer to negative fern distributions
    cv::Mat fun_get_ratHstms();  // pointer to ratio of fern distributions
    void fun_set_threshold(float beta);  // set detector threshold (beta)
    void fun_update(cv::Mat &mapx, float y);  // update the object classifier
    void fun_set_object_size(int Bu, int Bv);  // set object size (Bu,Bv)
    void fun_get_object_size(int& Bu, int& Bv);  // return object size (Bu,Bv)
    void fun_compute(int J, int R, int M, int S);  // compute the object classifier
};

// set of specific classifiers
class clsClassifierSet {
  private:
    int numClfrs;  // num. classifiers (K)
    int numMaxClfrs;  // num. max. classifiers
    clsClassifier* clfrs;  // array of classifiers
  public:
    clsClassifierSet();  // constructor
    ~clsClassifierSet();  // destructor
    void fun_init();  // initialize
    void fun_release();  // release memory
    int fun_get_num_classifiers();  // return num. classifiers (K)
    int fun_get_num_max_classifiers();  // return num. max. classifiers
    void fun_set_num_classifiers(int K);  // set num. classifiers (K)
    clsClassifier* fun_get_classifier(int k);  // return indexed classifier
};


// one instance detection
class clsDetection {
  private:
    int ide;  // detection identifier
    float score;  // detection score
    int u1,v1,u2,v2;  // detection location
  public:
    clsDetection();  // constructor
    ~clsDetection(); // destructor
    void fun_init();  // initialize
    void fun_set_values(int u1, int v1, int u2, int v2, float score, int ide);  // set detection values
    void fun_get_values(int &u1, int &v1, int &u2, int &v2, float &score, int &ide);  // return detection values
};


// detection set -multiple detections-
class clsDetectionSet {
  private:
    int numDets;  // num. detections
    int numMaxDets; // num. max. detections
    clsDetection* dets; // array of detections
  public:
    clsDetectionSet();  // constructor
    ~clsDetectionSet();  // destructor
    void fun_init();  // initialize
    void fun_release();  // release memory
    int fun_get_num_detections();  // return num. detections
    int fun_get_num_max_detections();  // return num. max. detections
    void fun_non_maxima_supression();  // return maxima detections
    void fun_scaling(float scaleFactor);  // scale detection coordinates
    void fun_set_num_detections(int value);  // set number of detections
    clsDetection* fun_get_detection(int index);  // return indexed detection
    void fun_remove_detections(clsDetection* det);  // remove detections using overlapping measure
    void fun_add_detections(clsDetectionSet* dets);  // add new detections
    void fun_get_max_detection(clsDetection* maxDet);  // return max. detection
    void fun_set_detection(clsDetection* det, int index);  // set indexed detection
};


// integral image
class clsII {
  private:
    int width;  // integral image width
    int height;  // integral image height
    cv::Mat II;  // integral image
    cv::Mat img;  // image (I)
    int imgChans;  // num. image channels (C)
    int cellSize;  // cell size -pixelsxpixels-
    int imgWidth;  // image width (Iv)
    int imgHeight;  // image height (Iu)
  public:
    clsII();  // constructor
    ~clsII();  // destructor
    cv::Mat fun_get_image();  // return the image pointer
    void fun_release_image();  // release image
    void fun_compute_image(int size);  // compute image from II
    void fun_integral_image(cv::Mat img);  // compute integral image
    void fun_get_image_size(int& Iu, int& Iv);  // return image size
};

#endif


