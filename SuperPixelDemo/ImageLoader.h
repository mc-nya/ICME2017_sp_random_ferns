#include"headfile.h"
#include<io.h>

class ImageLoader{
private:
	string path = "";		//���ݼ���·��
	string preDepthFilename, preColorFilename;		//��ȼ���ɫͼƬ��ǰ׺
	string sufDepthFilename, sufColorFilename;		//��ȼ���ɫͼƬ����׺
	int count;				//�ļ���ʾΪpath+ǰ׺+count+��׺
	int maxcount;
	Mat depthImage;
	Mat colorImage;		//��ȼ���ɫͼ��
	int existFlag;
public:
	/*���캯�� ��������Ϊ�� string ���ݼ���·��
							string ����ļ���ǰ׺
							string ����ļ�����׺
							string ��ɫ�ļ���ǰ׺
							string ��ɫ�ļ�����׺
							int ��ʼ���
							int ������)	*/
	ImageLoader(string datapath,string preDFilename,string sufDFilename,string preCfilename,string sufCfilename,int initcount,int endcount){
		count = initcount-1;
		maxcount = endcount;
		path = datapath;
		preDepthFilename = preDFilename;
		preColorFilename = preCfilename;
		sufDepthFilename = sufDFilename;
		sufColorFilename = sufCfilename;
		next();
		
	}
	//ȡ���õ���һ��ͼƬ	����ֵ=1 �ɹ�����	����0 ��������  
	int next(){   
		count++;		
		while (load() == 0){
			if (count > maxcount){
				return 0;
			}
			count++;
		}	
		return 1;
	}
	//����ͼƬ ����1���سɹ�  0����ʧ�ܣ��ļ������ڣ���ų������ޣ�
	bool load(){		//����1���سɹ�
		if (count > maxcount){		//��������
			return 0;
		}
		string dfilename = path + preDepthFilename + to_string(count) + sufDepthFilename;
		string cfilename = path + preColorFilename + to_string(count) + sufColorFilename;	//�ļ�������
		existFlag = testExist(dfilename, cfilename);
		if (!existFlag){		//�ļ�������
			return 0;
		}
		depthImage.release();
		colorImage.release();
		depthImage = imread(dfilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		colorImage = imread(cfilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		return 1;
	}
	//�ж��ļ��Ƿ���ڣ�������ȺͲ�ɫ�ļ���������·��������1�����
	bool testExist(string dfilename, string cfilename){		
		char cfile1[200], cfile2[200];
		strncpy(cfile1, dfilename.c_str(), dfilename.length());
		strncpy(cfile2, cfilename.c_str(), cfilename.length());
		cfile1[dfilename.length()] = '\0';
		cfile2[cfilename.length()] = '\0';
		if (_access(cfile1, 0) == -1 || _access(cfile2, 0) == -1){
			return 0;
		}
		return 1;
	}
	//��ȡ���mat
	Mat getDepthImage(){
		return depthImage;
	}
	//��ȡ��ɫͼmat
	Mat getColorImage(){
		return colorImage;
	}
	//��ȡ��ǰ���
	int getCount(){
		return count;
	}
};