#include "hod.h"


hod::hod(){
	Pos = cv::Mat(40, 7, CV_32FC1);//cv::Mat(25,6,CV_32FC1);
	Ign = cv::Mat(40, 5, CV_32FC1);

	MatHelper::ReadMat("32F", Pos, "F:\\2014project\\jianceren\\jianceren\\Pos.txt");
	MatHelper::ReadMat("32F", Ign, "F:\\2014project\\jianceren\\jianceren\\Ign.txt");
	//cout << Pos << endl;
	//printf("mat read ok\n");
	/////////////////////////////////////
	//     printf("ͳ����������;\n");
	//     TotalNum = 1485;
	//     PosNum = 0;
	//     IgnNum = 0;
	//     
	//     printf("TotalNum = %d\n",TotalNum);
	// 
	//     for(int Posj = 0; Posj < Pos.rows; Posj++ )
	//         for(int Posi = 0; Posi < Pos.cols; Posi++ ){
	//             if( Pos.at<float>(Posj,Posi) != -1)
	//                 PosNum = PosNum + 1 ;
	//         }
	//     printf("PosNum = %d\n",PosNum);
	// 
	//     for(int Ignj = 0; Ignj < Ign.rows; Ignj++ )
	//         for(int Igni = 0; Igni < Ign.cols; Igni++ ){
	//             if ( Ign.at<float>(Ignj,Igni) != -1)
	//                 IgnNum = IgnNum + 1 ;
	//         }
	//     printf("IgnNum = %d\n",IgnNum);
	// 
	//     printf("����������; = %d",TotalNum - PosNum -IgnNum );
	// 
	//     getchar();
	//         
	cvsvm.load("F:\\2014project\\jianceren\\jianceren\\SVM_HOG.xml");//��XML�ļ���ȡѵ���õ�SVMģ�� 


}

hod::~hod(){

}

int PosRow = -1, IgnRow = -1, PosCol, IgnCol, sampleFeatureMatRow = 0, sampleFeatureMatCol;
cv::Mat sampleFeatureMat = cv::Mat::zeros(PosSamNO + NegSamNO, DescriptorDim, CV_32FC1);
//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά�� ;     
cv::Mat sampleLabelMat = cv::Mat::zeros(PosSamNO + NegSamNO, 1, CV_32FC1);
//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����;

int cKey;
int label0 = 0, label1 = 0;

void hod::getHod(cv::Mat& m_oriBgrResultImage_8UC3, cv::Mat& m_preproResultImage_16UC1){

	int t = 0, s = 0, k = 0, li = 0, lj = 0, r, r2, pointnum = 0;
	char buffer[200];
	cv::Mat m_oriBgrResultImage2_8UC3, m_oriBgrResultImage3_8UC3;//copyto;
	m_oriBgrResultImage_8UC3.copyTo(m_oriBgrResultImage2_8UC3);
	m_oriBgrResultImage_8UC3.copyTo(m_oriBgrResultImage3_8UC3);
	cv::Mat m_preproResultImage2_16UC1;
	m_preproResultImage_16UC1.copyTo(m_preproResultImage2_16UC1);

	int subarow;

	cv::Mat subc = cv::Mat(3, 1, CV_32FC1);

	//learning

	//HOG
	cv::Mat RectRGBImg;
	cv::Mat resizedRGBImage;
	cv::Mat GrayRGBImg = cv::Mat(66, 50, CV_32FC1);
	cv::HOGDescriptor hog(cvSize(48, 64), cvSize(32, 32), cvSize(16, 16), cvSize(16, 16), 8);// �ֱ��� WinSize BlockSize BlockStride cellSize nBins

	// Normal Vectors
	cv::Mat RectDImg;
	cv::Mat resizedDImage;
	cv::Mat resizedDImage_8UC1;
	vector<float>abc;//feature[A/C,B/C]
	float b[2];//���뵽ÿ��abc�����ֵ[A/C,B/C];

	//��Ȳ�;
	cv::Mat DepthDif;
	cv::Mat resizedDepthDif;
	//FILE *fDetectee, *fOverlapper, *fBackground;

	//     cout << "Pos = " << endl << " " << Pos << endl << endl; 

	//if ( g_openNi.m_ImageGenerator.GetFrameID() % 50 == 0  &&  g_openNi.m_ImageGenerator.GetFrameID() <= 2000)
	if ((g_openNi.getCurFrameNum() - 1) % 50 == 0 && (g_openNi.getCurFrameNum() - 1) <= 2000)
	{
		//learning ��������ÿ50֡ѡ��һ������������һ����40����Ign �� Pos����40�У�ÿ�д��������֡�еĵڼ�����
		PosRow++;
		PosCol = 0;
		IgnRow++;
		IgnCol = 0;
		sampleFeatureMatCol = 0;

		for (int j = 50; j < m_preproResultImage2_16UC1.rows; ++j)
		{
			cv::circle(m_oriBgrResultImage_8UC3, cv::Point(540, j), 1, cv::Scalar(255, 255, 255), -1, 8); //��ɫ;

			for (int i = 1; i < m_preproResultImage2_16UC1.cols; ++i)//540����������Բ�Ҫ��
			{
				if (m_preproResultImage2_16UC1.at<unsigned short>(j, i) > 800 && m_preproResultImage2_16UC1.at<unsigned short>(j, i) < 6000)
					// �˵�����ֵ������Χ��; 
				{
					if (	 //������Ȳ�������һ����Χ��;
						abs(m_preproResultImage2_16UC1.at<unsigned short>(j, i - 1) - m_preproResultImage2_16UC1.at<unsigned short>(j, i)) > 200)
					{
						t = 0;
						//cout << i << " " << j << endl;
						r = Data3DHelper::GetSizeInImageBySizeIn3D(150, m_preproResultImage2_16UC1.at<unsigned short>(j, i));
						//r2 = Data3DHelper::GetSizeInImageBySizeIn3D(120, m_preproResultImage2_16UC1.at<unsigned short>(j,i));
						// ��ʵ�еĿ�����������ռ����������;
						//ȡ��һ�еĵ�;
						for (k = r; k > -r && i + k > 0 && i + k < 640; k--)
						{
							if (abs(m_preproResultImage2_16UC1.at<unsigned short>(j, i) - m_preproResultImage2_16UC1.at<unsigned short>(j - 1, i + k)) < 180)//��һ�л�һ���ߣ���Ϊ2r��Ҫ����������ȶ���С����ô����Ϊ��ͷ��������ߣ��ҵ���⣩
								//abs����Ϊ��������Ϊ0
								t = t + 1;//���Ƕ���;
							else
								t = t + 0;	//�Ƕ���;
						}//for															
						if (t == 0)
						{
							s = 0;
							for (lj = -2; lj < 1; lj++)
							{
								for (li = -1; li > -r && i + li > 0 && i + li < 640; li--)
								{
									if (
										m_oriBgrResultImage3_8UC3.at<cv::Vec3b>(j + lj, i + li)[0] == 0 &&
										m_oriBgrResultImage3_8UC3.at<cv::Vec3b>(j + lj, i + li)[1] == 255 &&
										m_oriBgrResultImage3_8UC3.at<cv::Vec3b>(j + lj, i + li)[2] == 255     //��ɫ;
										)
										s = s + 1;//֮ǰ�Ѿ���ǹ�����; ��Ϊ��������ɨ�裬�����ж����i-1��i+li������j-2��j�����Χ��û����ǹ�
									else
										s = s + 0;	//֮ǰû�б�ǹ��ĵ�;	
								}//for li
							}//for lj	
							if (s == 0)
							{
								//pointnum++;
								// printf("frame = %d\n",g_openNi.m_ImageGenerator.GetFrameID());
								// printf("pointnum = %d\n",pointnum);
								// printf("j = %d\t i = %d\n",j,i);
								//  printf("r = %d\n",r);
								cv::circle(m_oriBgrResultImage3_8UC3, cv::Point(i, j), 3, cv::Scalar(0, 255, 255), -1, 8);//��ɫ;

								if (int(i - 1.5*r - 1) < 0 || int(i + 1.5*r + 1) > 640 || int(j + 3.3*r + 1) > 480 || int(j - 0.7*r - 1) < 0)
								{
									//��ȡ����;  
									// printf("ȡ����;\n");
									continue;
								}
								else
								{//��ȡ�ĵ������;
									pointnum++;

									if (TRAIN)//��TRAINΪtrue������ѵ�������� ;
									{
										//printf("pointnum = %d\n",pointnum);
										if (pointnum == Ign.at<float>(IgnRow, IgnCol)){
											//������;
											IgnCol++;
											// printf("����;\n");
											// printf("IgnRow, IgnCol = %d\t%d\n",IgnRow,IgnCol);
											continue;
										}// if ��ȡ����;
										else
										{//��ȥ��ȡ�����ĵ�;
											// printf("r= %d\n",r);

											//                                             //  ========================================================
											//                                             //Copy Mat  HOG
											RectRGBImg.create(4 * r + 2, 3 * r + 2, CV_8UC3);//rows, cols, type 
											MatHelper::GetRectMat(m_oriBgrResultImage2_8UC3, RectRGBImg, int(i - 1.5*r - 1), int(j - 0.7*r - 1), 3 * r + 2, 4 * r + 2);//width height 
											cv::resize(RectRGBImg, resizedRGBImage, cv::Size(48, 64));//col row
											RectRGBImg.release();
											vector<float>feature;//�������;
											hog.compute(resizedRGBImage, feature, cv::Size(16, 16), cv::Size(0, 0)); //���ü��㺯����ʼ����;
											//                                             //Copy Mat  HOG
											//                                             //========================================================


											//======================================================
											//���; HOD
											// RectDImg.create(4*r+2,3*r+2, CV_16UC1);
											// MatHelper::GetRectDepthMat(m_preproResultImage2_16UC1, RectDImg,int(i-1.5*r-1),int(j-0.7*r-1),3*r+2,4*r+2 );
											// cv::resize(RectDImg,resizedDImage,cv::Size(48,64));
											// RectDImg.release();
											// vector<float>feature;//�������;
											// resizedDImage.convertTo(resizedDImage_8UC1, CV_8U , 255.0/7096.0);
											// hog.compute(resizedDImage_8UC1, feature,cv::Size(16,16),cv::Size(0,0));

											//                                             cv::imshow("...",resizedDImage_8UC1);
											//                                             cv::waitKey(1);
											//                                             getchar();
											//                                             int featurecount = 0;
											//                                             for(std::vector<float>::iterator m = feature.begin(); m != feature.end(); m++ )    //�õ������ķ�ʽ������������ֵ;
											//                                             {
											//                                                 cout<<*m<<endl;
											//                                                 featurecount++;
											//                                             }
											//                                             printf("featurecount; = %d\n",featurecount);
											//                                             getchar();

											//======================================================


											//�ж���or�� ���ļ�;     
											if (pointnum == Pos.at<float>(PosRow, PosCol)){
												//������ ;
												for (sampleFeatureMatCol = 0; sampleFeatureMatCol < DescriptorDim; sampleFeatureMatCol++)
													sampleFeatureMat.at<float>(sampleFeatureMatRow, sampleFeatureMatCol) = feature[sampleFeatureMatCol];

												sampleLabelMat.at<float>(sampleFeatureMatRow, 0) = 1;
												sampleFeatureMatRow++;
												PosCol++;



												cv::rectangle(m_oriBgrResultImage_8UC3, cv::Rect(i - 1.5*r, j - 0.7*r, 3 * r, 4 * r), cv::Scalar(0, 0, 255), 1, 8); //��ɫ //С��;
												char buf1[200];
												sprintf(buf1, "%d", pointnum);
												cv::putText(m_oriBgrResultImage_8UC3, buf1, cvPoint(i, j), CV_FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 255, 0)); //��ɫ; 

												cv::circle(m_preproResultImage_16UC1, cv::Point(i, j), 3, cv::Scalar(0, 255, 255), -1, 8);
												cv::rectangle(m_preproResultImage_16UC1, cv::Rect(i - 1.5*r, j - 0.7*r, 3 * r, 4 * r), cv::Scalar(0, 0, 255), 1, 8);
												//cv::rectangle(m_preproResultImage_16UC1,cv::Rect(i-r2,j-1,2*r2,2*r2),cv::Scalar(0,0,255),1,8);
											}
											else{
												//������;                                   
												for (sampleFeatureMatCol = 0; sampleFeatureMatCol < DescriptorDim; sampleFeatureMatCol++)

													sampleFeatureMat.at<float>(sampleFeatureMatRow, sampleFeatureMatCol) = feature[sampleFeatureMatCol];

												sampleLabelMat.at<float>(sampleFeatureMatRow, 0) = -1;
												sampleFeatureMatRow++;

												cv::rectangle(m_oriBgrResultImage_8UC3, cv::Rect(i - 1.5*r, j - 0.7*r, 3 * r, 4 * r), cv::Scalar(255, 0, 0), 1, 8); //��ɫ //С��;
												char buf1[200];
												sprintf(buf1, "%d", pointnum);
												cv::putText(m_oriBgrResultImage_8UC3, buf1, cvPoint(i, j), CV_FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 255, 0)); //��ɫ;   
											}
										}//else ��ȥ��ȡ�����ĵ�;
									}// TRAIN = true


									else
										//��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����;  
									{
										//                                         //========================================================
										//                                         //Copy Mat  HOG
										RectRGBImg.create(4 * r + 2, 3 * r + 2, CV_8UC3);//rows, cols, type   
										MatHelper::GetRectMat(m_oriBgrResultImage2_8UC3, RectRGBImg, int(i - 1.5*r - 1), int(j - 0.7*r - 1), 3 * r + 2, 4 * r + 2);//width height 
										cv::resize(RectRGBImg, resizedRGBImage, cv::Size(48, 64));//col row
										RectRGBImg.release();
										vector<float>feature;//�������;
										hog.compute(resizedRGBImage, feature, cv::Size(16, 16)); //���ü��㺯����ʼ����;
										//                                         //Copy Mat  HOG
										//                                         //========================================================


										//======================================================
										//���; HOD
										//RectDImg.create(4*r+2,3*r+2, CV_16UC1);
										//MatHelper::GetRectDepthMat(m_preproResultImage2_16UC1, RectDImg,int(i-1.5*r-1),int(j-0.7*r-1),3*r+2,4*r+2 );
										//// resizedDImage.create(64, 48, CV_16UC1);
										//cv::resize(RectDImg,resizedDImage,cv::Size(48,64));
										//RectDImg.release();
										//vector<float>feature;//�������;
										//resizedDImage.convertTo(resizedDImage_8UC1, CV_8U , 255.0/7096.0);
										//hog.compute(resizedDImage_8UC1, feature,cv::Size(16,16),cv::Size(0,0));
										//===============================================================


										cv::Mat ResponseMat = cv::Mat(1, DescriptorDim, CV_32FC1);
										for (int responsecol = 0; responsecol < DescriptorDim; responsecol++)
											ResponseMat.at<float>(0, responsecol) = feature[responsecol];
										float response = cvsvm.predict(ResponseMat);//, true);
										// printf("response = %f\n",response);
										int response1 = int(response);
										// printf(" response1 = %d\n ",response1);
										// getchar();

										if (response1 == -1){// > -2.5 ??
											//������;
											//  printf("������;\n");
											// cv::rectangle(m_oriBgrResultImage_8UC3,cv::Rect(i-1.5*r,j-0.7*r,3*r,4*r),cv::Scalar(255,0,0),1,8); //��ɫ //С��;
											continue;
										}
										else {
											//������;
											// printf("������;\n");
											cv::circle(m_oriBgrResultImage_8UC3, cv::Point(i, j), 3, cv::Scalar(0, 255, 255), -1, 8);//��ɫ;
											cv::rectangle(m_oriBgrResultImage_8UC3, cv::Rect(i - 1.5*r, j - 0.7*r, 3 * r, 4 * r), cv::Scalar(0, 0, 255), 1, 8); //��ɫ //С��;
											char buf1[200];
											sprintf(buf1, "%d", pointnum);
											cv::putText(m_oriBgrResultImage_8UC3, buf1, cvPoint(i, j), CV_FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0, 255, 0)); //��ɫ; 

											//���ͼ��;
											cv::circle(m_preproResultImage_16UC1, cv::Point(i, j), 3, cv::Scalar(0, 255, 255), -1, 8);
											cv::rectangle(m_preproResultImage_16UC1, cv::Rect(i - 1.5*r, j - 0.7*r, 3 * r, 4 * r), cv::Scalar(0, 0, 255), 1, 8);
										}
										// }//��ȡ�ĵ������;	                                
									}//TRAIN =false	

									//��ѵ����ʱ��;




									//                                       cv::rectangle(m_oriBgrResultImage_8UC3,cv::Rect(i-1.5*r,j-0.7*r,3*r,4*r),cv::Scalar(0,0,255),1,8); //��ɫ //С��;
									//                                       char buf1[200];
									//                                       sprintf(buf1,"%d",pointnum);
									//                                       cv::putText(m_oriBgrResultImage_8UC3,buf1,cvPoint(i,j),CV_FONT_HERSHEY_DUPLEX,0.5,CV_RGB(0,255,0)); //��ɫ; 

								}//��ȡ�ĵ�;
							}//if s = 0
						}//if t=0
					}//if �ٽ�����Ȳ�;
				} //if �˵�����ֵ������Χ�� ;
			}//for i
		}//for j



	}//frame % 50 && <= 1250


	else
	{
		/*if ( g_openNi.m_ImageGenerator.GetFrameID() % 50 == 0 && g_openNi.m_ImageGenerator.GetFrameID() <= 2050 && g_openNi.m_ImageGenerator.GetFrameID()> 2000 )*/
		if ((g_openNi.getCurFrameNum() - 1) % 50 == 0 && (g_openNi.getCurFrameNum() - 1) <= 2050 && (g_openNi.getCurFrameNum() - 1) > 2000)
		{
			//1250֡����

			//cout << "sampleFeatureMat = " << endl << " " << sampleFeatureMat << endl << endl;
			//getchar();
			printf("sampleFeatureMat rows clos; = %d\n%d\n", sampleFeatureMat.rows, sampleFeatureMat.cols);
			// getchar();
			//cout << "sampleLabelMat = " << endl << " " << sampleLabelMat << endl << endl; 
			printf("sampleLabelMat rows clos; = %d\n%d\n", sampleLabelMat.rows, sampleLabelMat.cols);
			//
			// getchar();
			//ѵ��SVM������  ;
			MatHelper::PrintMat("32F", sampleFeatureMat, "sampleFeatureMat.txt");
			MatHelper::PrintMat("32F", sampleLabelMat, "sampleLabelMat.txt");
			printf("print ok!\n");

			CvSVMParams param;
			param.svm_type = CvSVM::C_SVC;
			//param.C           = 0.1;
			param.kernel_type = CvSVM::LINEAR;
			param.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 3000, 1e-7);// 100

			//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����  
			//                   CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);  
			//                   //SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01  ;
			//                 //  CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);  
			//                   CvSVMParams param(CvSVM::C_SVC, CvSVM::RBF, 0, 1, 0, 1, 0, 0, 0, criteria);
			// Set up SVM's parameters

			cout << "��ʼѵ��SVM������" << endl;
			//CvSVM cvsvm;
			cvsvm.train(sampleFeatureMat, sampleLabelMat, cv::Mat(), cv::Mat(), param);//ѵ��������  ;
			cout << "ѵ�����" << endl;
			int supportVectorNum = cvsvm.get_support_vector_count();//֧�������ĸ��� ; 
			cout << "֧������������" << supportVectorNum << endl;

			for (int tempFeatureMat = 0; tempFeatureMat < PosSamNO + NegSamNO; tempFeatureMat++){
				cv::Mat ResponseMat = cv::Mat(1, DescriptorDim, CV_32FC1);
				for (int responsecol = 0; responsecol < DescriptorDim; responsecol++)
					ResponseMat.at<float>(0, responsecol) = sampleFeatureMat.at<float>(tempFeatureMat, responsecol);
				float response = cvsvm.predict(ResponseMat);
				int response1 = int(response);
				// printf("label = %d\n",int(sampleLabelMat.at<float>(temp,0)));
				// printf("response1 = %d\n",response1);
				if (int(sampleLabelMat.at<float>(tempFeatureMat, 0)) == 1 && response1 == -1){
					//  printf(" 1 -- -1 :%d\n",tempFeatureMat);
					label1++;
				}
				if (int(sampleLabelMat.at<float>(tempFeatureMat, 0)) == -1 && response1 == 1){
					// printf(" -1 -- 1 :%d\n",tempFeatureMat);
					label0++;
				}

			}
			printf("label = 1 response1 = -1 : %d\n", label1);
			printf("label = -1 response1 = 1 : %d\n", label0);
			cvsvm.save("SVM_HOG1.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�; 

			getchar();

		}// frame 1250 - 1300 
	}//else



}