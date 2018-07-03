// superpixel-opencv.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include"headfile.h"
#include "slic.h"
#include <cstdlib>
#include <cstdio>
#include"ImageLoader.h"
#include<time.h>
#include "detector_classes.h"
#include "detector_functions.h"
#include"MathHelper.h"
#include"MatHelper.h"
#include"RandomFerns.h"
#include"DarkImLoader.h"
#define DIFF_T 100
using namespace std;


//��ͼʹ�ñ���
int superPixelMap[400000];	//����ģ���ڽӱ�,superPixelMap�����
int mapNext[400000];	//����ģ���ڽӱ� mapNext����ָ��
int head[4000];		//����ģ���ڽӱ�  head����ͷָ��
int seed[4000][2];	//������ӣ�seed[label][0]=�к�(i)  seed[label][1]=�к�(j)
Point2i topPoint[4000];
int unionSet[4000];   //���鼯��������ϲ�  -1��ʾ������
int mapcount = 0;
int maxLabel = 0;
//ͳ�Ʊ���
int superPixelSum[4000];		//����ҶȺ�
int pixel[4000];	//�������ظ���
int Lmax[4000];		//�Ҷ����ֵ
int Lmin[4000];		//�Ҷ���Сֵ
int unionSum[4000];		//���ϻҶȺ�
int unionPixel[4000];	//�������ظ���
//��ʱ����
int visitUsingInGraph[700][700];		//���ڽ�ͼʱʹ�õ���ʱflag����ջ��д��ȫ��
int qx[200000], qy[200000];				//���ڽ�ͼʱʹ�õĶ��У���ջ��д��ȫ��
int queue[100000];				//���ںϲ�ʱʹ�õĶ��У���ջ��д��ȫ��


//Ѱ�Ҽ�������
int findFather(int label){
	if (unionSet[label] == label)
		return label;
	unionSet[label] = findFather(unionSet[label]);
	return unionSet[label];
}


int main()
{

	cv::Mat img, result;
	//ImageLoader imLoader =ImageLoader("E:\\ʵ����\\dataset\\47\\", "depth\\depth_", ".png", "rgb\\rgb_", ".png", 34889, 53000);
	//ImageLoader imLoader = ImageLoader("E:\\ʵ����\\dataset\\38\\", "depth\\depth_", ".png", "rgb\\rgb_", ".png", 2030, 3000);
	//ImageLoader imLoader = ImageLoader("E:\\ʵ����\\dataset\\18D\\", "depth\\png\\1 (", ")_16UC1.png", "rgb\\1 (", ").png", 1, 4485);
	//ImageLoader imLoader = ImageLoader("E:\\ʵ����\\dataset\\10D\\", "depth\\png\\1 (", ")_16UC1.png", "rgb\\1 (", ").png", 1, 550);
	//ImageLoader imLoader = ImageLoader("E:\\ʵ����\\dataset\\1\\", "depth\\png\\1 (", ")_8UC1_From1File.png", "rgb\\1 (", ").png", 1, 1490);
	//ImageLoader imLoader = ImageLoader("E:\\ʵ����\\dataset\\11\\", "depth\\png\\1 (", ")_16UC1.png", "rgb\\1 (", ").png", 240, 1112);
	//ImageLoader imLoader = ImageLoader("E:\\ʵ����\\dataset\\8\\", "depth\\png\\1 (", ")_8UC1_From8File.png", "rgb\\1 (", ").png", 1, 1430);

	//DarkImLoader imLoader = DarkImLoader("F:\\TMM_2017\\Outdoor\\dark\\56\\", "depth\\depth_", ".png", "rgb\\img_", ".png",0, 1439);
	DarkImLoader imLoader = DarkImLoader("F:\\TMM_2017\\Outdoor\\dark\\31\\", "depth\\depth_", ".png", "rgb\\img_", ".png", 0, 1439);
	//DarkImLoader imLoader = DarkImLoader("F:\\TMM_2017\\Outdoor\\dark\\54\\", "depth\\depth_", ".png", "rgb\\img_", ".png", 0, 1439);

	RandomFerns RF = RandomFerns();
	//RF.init();
	//printf("please input the number of superpixel:");
	//int num; 
	//scanf("%d",&num);
	//int numSuperpixel = num;
	int numSuperpixel = 1000;
	while (imLoader.next()){

		/*if (imLoader.getCount() % 10 != 0){
			continue;
		}
		*/
		SLIC slic;
		img = imLoader.getDepthImage();		//����ͼ��
		////timing**********
		clock_t start, end;
		double dur;
		start = clock();
		////*******************	

		slic.GenerateSuperpixels(img, numSuperpixel);	//����superpixel
		int* label = slic.GetLabel();		//��ȡ���  ��Ÿ�ʽ����i��j��ͨ��label[i*cols+j]����
		//cout << imLoader.getCount() << endl;
		if (img.channels() == 3)
			result = slic.GetImgWithContours(cv::Scalar(0, 0, 255));
		else
			result = slic.GetImgWithContours(cv::Scalar(50000));


		
		
		for (int i = 0; i < 200000; i++){	//��ʼ��
			mapNext[i] = -1;
			superPixelMap[i] = -1;
		}
		for (int i = 0; i < 4000; i++){	//��ʼ��
			Lmin[i] = INT_MAX;
			superPixelSum[i] = 0;
			pixel[i] = 0;
			Lmax[i] = -1;
			seed[i][0] = -1;
			seed[i][1] = -1;
			head[i] = -1;
			unionSet[i] = i;
			topPoint[i].x = -1;
			topPoint[i].y = -1;
		}

		{		//Ϊÿ������趨һ�����ӵ�
			int visit[4000];
			for (int i = 0; i < 4000; i++){		//��ʼ��
				visit[i] = 0;		
				
			}
			for (int i = 5; i < result.rows-5; i++){
				for (int j = 5; j < result.cols-5; j++){
					int lb = label[i*result.cols + j];
					maxLabel = max(maxLabel, lb);
					if (!visit[lb]){		//δ���ʹ����趨���ӵ�
						visit[lb] = 1;
						seed[lb][0] = i;
						seed[lb][1] = j;
					}
				}
			}
		}
		{

			int isAdjoin[4000];
			for (int i = 0; i < 700; i++){
				for (int j = 0; j < 700; j++){
					visitUsingInGraph[i][j] = 0;
				}
			}

			for (int i = 0; i <= maxLabel; i++){		//�����ӵ㿪ʼ�������򣬽������ݵ�ͳ�ƺ��ڽ�������趨		i��ʾ��ǰ����label
				for (int j = 0; j < maxLabel; j++){
					isAdjoin[j] = 0;
				}
				int initx = seed[i][0];
				int inity = seed[i][1];
				if (initx == -1){
					continue;
				}
				int h = 0, t = 0;
				qx[0] = initx; qy[0] = inity;
				visitUsingInGraph[initx][inity] = 1;			//��ʼֵ�趨
				int dx[4] = { -1, -1, 1, 1 };		//��������
				int dy[4] = { -1, 1, -1, 1 };
				while (h <= t){
					//ͳ��
					int pixelDepth = img.at<ushort>(qx[h], qy[h]);
					if (pixelDepth > 800 && pixelDepth < 10000)
					{
						superPixelSum[i] += pixelDepth;
						pixel[i]++;
						Lmax[i] = max(Lmax[i], int(pixelDepth));
						Lmin[i] = min(Lmin[i], int(pixelDepth));
					}
					//��ͼ
					for (int j = 0; j < 4; j++){
						int x = qx[h] + dx[j];
						int y = qy[h] + dy[j];
						if (x < 0) x = 0;
						if (x == result.rows) x = result.rows - 1;
						if (y < 0) y = 0;
						if (y == result.cols)y = result.cols - 1;
						if (label[x*result.cols + y] != i){			//̽�����ڽ�����							
							if (isAdjoin[label[x*result.cols + y]] == 0){		//�жϸñ�֮ǰδ������ͼ��
								mapcount++;
								mapNext[mapcount] = head[i];
								head[i] = mapcount;
								superPixelMap[mapcount] = label[x*result.cols + y];
								isAdjoin[label[x*result.cols + y]] = 1;			//����ͼ�в����
							}
						}
						else
						if (!visitUsingInGraph[x][y]){		//��ͬ����δ���ʹ��ĵ������ʶ���
							t++;
							qx[t] = x;
							qy[t] = y;
							visitUsingInGraph[x][y] = 1;
						}
					}
					h++;
				}
			}
		}

		for (int i = 0; i < maxLabel; i++){
			unionPixel[i] = pixel[i];
			unionSum[i] = superPixelSum[i];
		}



		int colorPrint[5000];

		{
			int h = 0, t = -1;
			int isInQueue[4000];	//���falg
			for (int i = 0; i<maxLabel; i++){
				if (head[i] != -1){
					t++;
					queue[t] = i;
					isInQueue[i] = 1;
				}
				else{
					isInQueue[i] = 0;
				}
			}
			

			while (h <= t){
				int currentLabel = queue[h];
				int currentFather = findFather(currentLabel);
				//ͳ����Ч���ز�Ϊ0
				if (pixel[currentLabel] != 0){			
					int mapPoint = head[currentLabel];
					while (mapPoint != -1){
						int testLabel = superPixelMap[mapPoint];
						int testFather = findFather(testLabel);
						//cout << testLabel << " " << pixel[testLabel] << "��" << testFather << " " << currentFather << endl;
						//ͳ����Ч���ز�Ϊ0
						if (testLabel >= 0 && testLabel<maxLabel){
							int testFather = findFather(testLabel);
							if (pixel[testLabel] != 0 && testFather != currentFather){
								/*double mid1 = double(superPixelSum[currentLabel]) / double(pixel[currentLabel]);
								double mid2 = double(superPixelSum[testLabel]) / double(pixel[testLabel]);
								double diff = abs(mid1 - mid2);
								if (diff < 500){
								unionSet[testFather] = currentFather;
								if (isInQueue[testLabel] == 0){
								t++;
								queue[t] = testLabel;
								isInQueue[testLabel] = 1;
								}
								}*/
								double mid1 = double(unionSum[currentFather]) / double(unionPixel[currentFather]);
								double mid2 = double(unionSum[testFather]) / double(unionPixel[testFather]);
								double diff = abs(mid1 - mid2);
								if (diff < DIFF_T){
									unionSet[testFather] = currentFather;
									unionSum[currentFather] += unionSum[testFather];
									unionPixel[currentFather] += unionPixel[testFather];
									if (isInQueue[testLabel] == 0){
										t++;
										queue[t] = testLabel;
										isInQueue[testLabel] = 1;
									}
								}
							}
						}
						
						

					mapPoint = mapNext[mapPoint];
					}
				}
				
				isInQueue[currentLabel] = 0;
				h++;
			}
		}
		
		srand((unsigned)time(NULL));
		for (int i = 0; i < maxLabel; i++){
			int fatherLabel = findFather(i);
			if (topPoint[fatherLabel].y == -1 || topPoint[fatherLabel].x > seed[i][0]){
				topPoint[fatherLabel].x = seed[i][0];
				topPoint[fatherLabel].y = seed[i][1];
			}
			colorPrint[i] = ((rand() % 65535)*(rand() % 65535)) %65535;
		}
		
		Mat saveImg = img.clone();
		{
			Vec3w colorPrint[4000];
			for (int i = 0; i < maxLabel; i++){
				colorPrint[i][0] = ((rand() % 65535)*(rand() % 65535)) % 65535;
				colorPrint[i][1] = ((rand() % 65535)*(rand() % 65535)) % 65535;
				colorPrint[i][2] = ((rand() % 65535)*(rand() % 65535)) % 65535;
			}
			for (int i = 5; i < saveImg.rows - 5; i++){
				for (int j = 5; j < saveImg.cols - 5; j++){
					saveImg.at<ushort>(i, j) = (50000 - saveImg.at<ushort>(i, j));

				}
			}
			cvtColor(saveImg, saveImg, CV_GRAY2RGB);
			for (int i = 5; i < saveImg.rows - 5; i++){
				for (int j = 5; j < saveImg.cols - 5; j++){
					int father = findFather(label[i*saveImg.cols + j]);
					if (unionPixel[father]>500){
						saveImg.at<cv::Vec3w>(i, j) = colorPrint[father];
					}
					//if (unionPixel[father] * ((unionSum[father] / (unionPixel[father]>0?unionPixel[father]:1)) / 1000)>3000){
					//	saveImg.at<cv::Vec3w>(i, j) = colorPrint[father];
					//}

				}
			}
			for (int i = 0; i < maxLabel; i++){
				int fatherLabel = findFather(i);
				if (unionPixel[fatherLabel]>500){
					circle(saveImg, Point2i(topPoint[fatherLabel].y, topPoint[fatherLabel].x), 4, Scalar(0, 69 * 256, 255 * 255), -1, 8);
				}
				
			}

			
		}
		
		
		for (int i = 5; i < result.rows - 5; i++){
			for (int j = 5; j < result.cols - 5; j++){
				//int father = findFather(label[i*result.cols + j]);
				//result.at<ushort>(i, j) = short(colorPrint[father]);// 55535 - result.at<ushort>(i, j);
				result.at<ushort>(i, j) = 50000 - result.at<ushort>(i, j);
			}
		}
		int headcount = 0;
		Mat imgRGB = imLoader.getColorImage();
		Mat resultRGB = imgRGB.clone();
		
		for (int setLabel = 0; setLabel < maxLabel; setLabel++){
			if (setLabel == findFather(setLabel) && unionPixel[setLabel]>0){
				
				int i = topPoint[setLabel].x;
				int j = topPoint[setLabel].y;

				//if (unionPixel[setLabel]>500)
				//circle(imgRGB, Point(j, i), 3, Scalar(0, 69, 255), -1, 8);
				//circle(result, Point(j, i), 3, Scalar(0, 69, 255), -1, 8);
				//int pixelNum =MathHelper::GetSizeInImageBySizeIn3D(150, img.at<unsigned short>(i, j));
				int pixelNum = MathHelper::GetSizeInImageBySizeIn3D(150,unionSum[setLabel]/unionPixel[setLabel]);
				if (i<40 || i>400)continue;
				//if (int(j - 1.5*pixelNum - 1) < 0 || int(j + 1.5*pixelNum + 1) > 640 || int(i + 3.3*pixelNum + 1) > 480 || int(i - 0.7*pixelNum - 1) < 0)
					//continue;
				Mat RectRGB = Mat(4 * pixelNum + 2, 3 * pixelNum + 2, CV_8UC3);
				MatHelper::GetRectMat(imgRGB, RectRGB,int(j - 1.5*pixelNum - 1), int(i - 0.7*pixelNum - 1), 3 * pixelNum + 2, 4 * pixelNum + 2);

				
				////if ((double)unionPixel[setLabel] * ((double)((double)unionSum[setLabel] / (double)unionPixel[setLabel]) / 10000.0)>200){
				//if ((double)unionPixel[setLabel]>500){
				//	if (RF.predict(RectRGB) == 0)continue;
				////if ((double)unionPixel[setLabel]>pixelNum*pixelNum*1.5){
				//	//headcount++;
				//	//putText(resultRGB, to_string(headcount), Point(j+3, i+3), CV_FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 69, 255));
				//	circle(resultRGB, Point(j, i), 3, Scalar(0, 69, 255), -1, 8);
				//	rectangle(resultRGB, cv::Rect(j - 1.5*pixelNum, i - 0.7*pixelNum, 3 * pixelNum, 4 * pixelNum), cv::Scalar(47, 230, 173), 5, 8);
				//	//circle(result, Point(j, i), 3, Scalar(0, 69, 255), -1, 8);
				//	//rectangle(result, cv::Rect(j - 1.5*pixelNum, i - 0.7*pixelNum, 3 * pixelNum, 4 * pixelNum), cv::Scalar(47, 230, 173), 5, 8);
				//}

			}
		}
		//cout << headcount << endl;
		////timing**********
		end = clock();
		dur = (double)(end - start);
		//dur = dur / 60;
		//cout <<"FPS: "<< (1 / (dur / CLOCKS_PER_SEC)) << endl;
		//cout <<"time: "<< (dur / CLOCKS_PER_SEC) << endl;
		////*******************


		

		imwrite("E:\\lab\\SuperPixelDemo\\output\\seg\\31\\Superpixel_"+to_string(imLoader.getCount())+".png",result);
		imwrite("E:\\lab\\SuperPixelDemo\\output\\seg\\31\\Superpixel_" + to_string(imLoader.getCount()) + "_Union.png", saveImg);
		//imwrite("E:\\lab\\SuperPixelDemo\\output\\47\\x2\\DetectResult_" + to_string(imLoader.getCount()) + ".png", resultRGB);
		//imwrite("E:\\lab\\SuperPixelDemo\\output\\T\\"+to_string(DIFF_T)+"\\DetectResult_" + to_string(imLoader.getCount()) + ".png", resultRGB);
		//fstream file("E:\\lab\\SuperPixelDemo\\output\\T\\" + to_string(DIFF_T) + "\\result.txt", ios::in | ios::app);

		//file << headcount << endl;
		//file.flush();
		//file.close();
		imshow("save", saveImg);
		imshow("depth" , result);
		//imshow("color", resultRGB);
		cvWaitKey(1);
		//slic.SLIC::~SLIC();
		//imLoader.next();
	}
	//cvWaitKey();
	//cv::imwrite("E:\\lab\\SuperPixelDemo\\SuperPixelDemo\\result.png", result);
	//cv::imwrite("E:\\lab\\SuperPixelDemo\\SuperPixelDemo\\result.jpg", result);
	//cout << "SLIC finished!" << endl;
	return 0;
}
