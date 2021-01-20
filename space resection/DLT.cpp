#include <stdio.h>
#include <vector>
#include <iostream>
#include <cv.h>
using namespace std;
using namespace cv;

typedef struct{
	double X;
	double Y;
	double Z;
}PtGround;//ground control point

typedef struct{
	double x;
	double y;
}PtImage;//image point

typedef struct{
	double Xs, Ys, Zs;
	double U, W, K;
}OutParameters;//outerior parameters


typedef struct{
	double x0, y0, f;
}InParameters;//interior parameters

typedef struct{
	double k1;
}DistortCoefficients;// distortion coefficients

typedef struct{
	OutParameters outDelta;
	InParameters inDelta;
	DistortCoefficients distDelta;
}Delta;

typedef struct{
	int gcpIndex;
	PtGround gcp;
}gcpInfo;

typedef struct{
	int imgPtIndex;
	PtImage imgPt;
}imgPtInfo;

const int UNKNOWNPARA = 12;//11+1
const int IMAGENUM = 2;

vector<imgPtInfo> calculateML(Mat matA, Mat x, Mat matL, vector<gcpInfo> gcps, char *outputFileName)
{
	vector<imgPtInfo> gptsRes;
	imgPtInfo gptRes;
	FILE *res = fopen(outputFileName, "w");
	cv::Mat vv(gcps.size() * 2, 1, CV_64F);
	vv = matA*x - matL;
	fprintf(res, "residual error:\n");
	for (int i = 0; i < vv.rows / 2; i++)
	{
		fprintf(res, "%d\t%lf\t%lf\n", gcps[i].gcpIndex, vv.at<double>(2 * i, 0), vv.at<double>(2 * i + 1, 0));
		if (abs(vv.at<double>(i, 0))>3) cout << "error:" << i << endl;
		gptRes.imgPtIndex = gcps[i].gcpIndex;
		gptRes.imgPt.x = vv.at<double>(2 * i, 0);
		gptRes.imgPt.y = vv.at<double>(2 * i + 1, 0);
		gptsRes.push_back(gptRes);
	}
	cv::Mat v = vv.t()*vv;
	double sumV = v.at<double>(0.0);
	double m0 = sqrt(sumV / (2 * gcps.size() - UNKNOWNPARA));
	double* m = new double[gcps.size()];
	cv::Mat Q = (matA.t()*matA).inv();
	cout << "Li中误差为" << endl;
	fprintf(res, "Li中误差为\n");
	for (int i = 0; i < UNKNOWNPARA; i++)
	{
		m[i] = sqrt(Q.at<double>(i, i))*m0;
		fprintf(res, "%g\t", m[i]);
		printf("%lf\n", m[i]);
	}
	printf("单位权中误差为:%lf像素\n", m0);
	fprintf(res, "单位权中误差为:%lf像素\n", m0);
	fclose(res);
	return gptsRes;
}
//calculate image point coordinates residual
void calculateM0(vector<imgPtInfo> resL1, vector<imgPtInfo> resL2)
{
	FILE *fRes = fopen("GCPRes", "w");
	fprintf(fRes, "控制点量测值单位权中误差为\n");
	for (int i = 0; i < resL1.size(); i++)
	{
		double vv = resL1[i].imgPt.x*resL1[i].imgPt.x + resL1[i].imgPt.y*resL1[i].imgPt.y + resL2[i].imgPt.x*resL2[i].imgPt.x + resL1[i].imgPt.y*resL1[i].imgPt.y;
		double m0 = sqrt(vv / (4 - 3));
		fprintf(fRes, "%d\t%lf\n", resL1[i].imgPtIndex, m0);
	}
	fclose(fRes);
	cout << "gcp m0 output OK" << endl;
}

Mat getL0(vector<gcpInfo> gcps,vector<imgPtInfo> imgPts)
{
	//calculate initial value of L
	
	Mat M(11, 11, CV_64F);
	//5 equations, 2.5 ground control points
	for (int i = 0; i < 6; i++)
	{
		
		M.at<double>(2*i, 0) = gcps[i].gcp.X;
		M.at<double>(2 * i, 1) = gcps[i].gcp.Y;
		M.at<double>(2 * i, 2) = gcps[i].gcp.Z;
		M.at<double>(2 * i, 3) = 1;
		M.at<double>(2 * i, 4) = 0;
		M.at<double>(2 * i, 5) = 0;
		M.at<double>(2 * i, 6) = 0;
		M.at<double>(2 * i, 7) = 0;
		M.at<double>(2 * i, 8) = gcps[i].gcp.X*imgPts[i].imgPt.x;
		M.at<double>(2 * i, 9) = gcps[i].gcp.Y*imgPts[i].imgPt.x;
		M.at<double>(2 * i, 10) = gcps[i].gcp.Z*imgPts[i].imgPt.x;
		if (i == 5) break;
		M.at<double>(2 * i + 1, 0) = 0;
		M.at<double>(2 * i + 1, 1) = 0;
		M.at<double>(2 * i + 1, 2) = 0;
		M.at<double>(2 * i + 1, 3) = 0;
		M.at<double>(2 * i + 1, 4) = gcps[i].gcp.X;
		M.at<double>(2 * i + 1, 5) = gcps[i].gcp.Y;
		M.at<double>(2 * i + 1, 6) = gcps[i].gcp.Z;
		M.at<double>(2 * i + 1, 7) = 1;
		M.at<double>(2 * i + 1, 8) = gcps[i].gcp.X*imgPts[i].imgPt.y;
		M.at<double>(2 * i + 1, 9) = gcps[i].gcp.Y*imgPts[i].imgPt.y;
		M.at<double>(2 * i + 1, 10) = gcps[i].gcp.Z*imgPts[i].imgPt.y;
	}
	Mat X(11, 1, CV_64F);
	for (int i = 0; i < 6; i++)
	{
		X.at<double>(2*i, 0) = -imgPts[i].imgPt.x;
		if (i == 5) break;
		X.at<double>(2*i+1, 0) = -imgPts[i].imgPt.y;
	}
	Mat L = M.inv()*X;
	
	return L;
}
InParameters getx0y0(Mat L)
{
	InParameters inPara;
	inPara.f = 21 * 1000 / 5.1966;
	double temp = L.at<double>(8, 0)*L.at<double>(8, 0) + L.at<double>(9, 0)*L.at<double>(9, 0) + L.at<double>(10, 0)*L.at<double>(10, 0);
	inPara.x0 = -(L.at<double>(0, 0)*L.at<double>(8, 0) + L.at<double>(1, 0)*L.at<double>(9, 0) + L.at<double>(2, 0)*L.at<double>(10, 0)) / temp;
	inPara.y0 = -(L.at<double>(4, 0)*L.at<double>(8, 0) + L.at<double>(5, 0)*L.at<double>(9, 0) + L.at<double>(6, 0)*L.at<double>(10, 0)) / temp;
	return inPara;
}
Mat getAccurateL(Mat L0,InParameters inPara, vector<gcpInfo> gcps, vector<imgPtInfo> imgPts,vector<imgPtInfo> &gcpsRes, bool getRes=FALSE,char* fileName="ResL1")
{
	Mat M(gcps.size()*2, UNKNOWNPARA, CV_64F);
	Mat W(gcps.size() * 2, 1, CV_64F);
	for (int i = 0; i < gcps.size(); i++)
	{
		double A = L0.at<double>(8, 0)*gcps[i].gcp.X + L0.at<double>(9, 0)*gcps[i].gcp.Y + L0.at<double>(10, 0)*gcps[i].gcp.Z + 1;
		double r2 = (imgPts[i].imgPt.x - inPara.x0)*(imgPts[i].imgPt.x - inPara.x0) + (imgPts[i].imgPt.y - inPara.y0)*(imgPts[i].imgPt.y - inPara.y0);
		M.at<double>(2 * i , 0) = gcps[i].gcp.X / A;
		M.at<double>(2 * i , 1) = gcps[i].gcp.Y / A;
		M.at<double>(2 * i , 2) = gcps[i].gcp.Z / A;
		M.at<double>(2 * i , 3) = 1 / A;
		M.at<double>(2 * i , 4) = 0;
		M.at<double>(2 * i , 5) = 0;
		M.at<double>(2 * i , 6) = 0;
		M.at<double>(2 * i , 7) = 0;
		M.at<double>(2 * i , 8) = imgPts[i].imgPt.x*gcps[i].gcp.X / A;
		M.at<double>(2 * i , 9) = imgPts[i].imgPt.x*gcps[i].gcp.Y / A;
		M.at<double>(2 * i , 10) = imgPts[i].imgPt.x*gcps[i].gcp.Z / A;
		M.at<double>(2 * i , 11) = (imgPts[i].imgPt.x-inPara.x0)*r2;

		M.at<double>(2 * i+1, 0) = 0;
		M.at<double>(2 * i+1, 1) = 0;
		M.at<double>(2 * i+1, 2) = 0;
		M.at<double>(2 * i+1, 3) = 0;
		M.at<double>(2 * i+1, 4) = gcps[i].gcp.X / A;
		M.at<double>(2 * i+1, 5) = gcps[i].gcp.Y / A;
		M.at<double>(2 * i+1, 6) = gcps[i].gcp.Z / A;
		M.at<double>(2 * i+1, 7) = 1 / A;
		M.at<double>(2 * i+1, 8) = imgPts[i].imgPt.y*gcps[i].gcp.X / A;
		M.at<double>(2 * i+1, 9) = imgPts[i].imgPt.y*gcps[i].gcp.Y / A;
		M.at<double>(2 * i+1, 10) = imgPts[i].imgPt.y*gcps[i].gcp.Z / A;
		M.at<double>(2 * i+1, 11) = (imgPts[i].imgPt.y - inPara.y0)*r2;

		W.at<double>(2 * i, 0) = imgPts[i].imgPt.x / A;
		W.at<double>(2 * i+1, 0) = imgPts[i].imgPt.y / A;
	}
	Mat L = -(M.t()*M).inv()*M.t()*W;
	if (getRes)
	{
		gcpsRes=calculateML(-M, L, W, gcps, fileName);
	}
	return L;
}
void updateImagePoint(Mat L, vector<imgPtInfo> &imgPts,InParameters inPara)
{
	for (int i = 0; i < imgPts.size(); i++)
	{
		double r2 = (imgPts[i].imgPt.x - inPara.x0)*(imgPts[i].imgPt.x - inPara.x0) + (imgPts[i].imgPt.y - inPara.y0)*(imgPts[i].imgPt.y - inPara.y0);
		imgPts[i].imgPt.x += (imgPts[i].imgPt.x - inPara.x0)*r2*L.at<double>(UNKNOWNPARA - 1, 0);
		imgPts[i].imgPt.y += (imgPts[i].imgPt.y - inPara.y0)*r2*L.at<double>(UNKNOWNPARA - 1, 0);
	}
}

vector<gcpInfo> getX0Y0Z0(Mat L1, Mat L2, vector<imgPtInfo> uImgPts1, vector<imgPtInfo> uImgPts2)
{
	//calculate image point in object space 
	vector<gcpInfo> uGpts;
	for (int i = 0; i < uImgPts1.size(); i++)
	{
		Mat N0(3, 3, CV_64F);
		Mat Q0(3, 1, CV_64F);
		N0.at<double>(0, 0) = L1.at<double>(0, 0) + uImgPts1[i].imgPt.x*L1.at<double>(8, 0);
		N0.at<double>(0, 1) = L1.at<double>(1, 0) + uImgPts1[i].imgPt.x*L1.at<double>(9, 0);
		N0.at<double>(0, 2) = L1.at<double>(2, 0) + uImgPts1[i].imgPt.x*L1.at<double>(10, 0);
		N0.at<double>(1, 0) = L1.at<double>(4, 0) + uImgPts1[i].imgPt.y*L1.at<double>(8, 0);
		N0.at<double>(1, 1) = L1.at<double>(5, 0) + uImgPts1[i].imgPt.y*L1.at<double>(9, 0);
		N0.at<double>(1, 2) = L1.at<double>(6, 0) + uImgPts1[i].imgPt.y*L1.at<double>(10, 0);

		N0.at<double>(2, 0) = L2.at<double>(0, 0) + uImgPts2[i].imgPt.x*L2.at<double>(8, 0);
		N0.at<double>(2, 1) = L2.at<double>(1, 0) + uImgPts2[i].imgPt.x*L2.at<double>(9, 0);
		N0.at<double>(2, 2) = L2.at<double>(2, 0) + uImgPts2[i].imgPt.x*L2.at<double>(10, 0);

		Q0.at<double>(0, 0) = -(L1.at<double>(3, 0) + uImgPts1[i].imgPt.x);
		Q0.at<double>(1, 0) = -(L1.at<double>(7, 0) + uImgPts1[i].imgPt.y);
		Q0.at<double>(2, 0) = -(L2.at<double>(3, 0) + uImgPts2[i].imgPt.x);

		Mat S0 = N0.inv()*Q0;
		gcpInfo gpt;
		gpt.gcpIndex = uImgPts1[i].imgPtIndex;
		gpt.gcp.X = S0.at<double>(0, 0);
		gpt.gcp.Y = S0.at<double>(1, 0);
		gpt.gcp.Z = S0.at<double>(2, 0);
		uGpts.push_back(gpt);
	}
	return uGpts;
}
vector<gcpInfo> getAccurateXYZ(Mat L1, vector<imgPtInfo> uimgpts1, Mat L2, vector<imgPtInfo> uimgpts2, vector<gcpInfo> ugpt0)
{
	vector<gcpInfo> uGpt;
	Mat N(IMAGENUM*2, 3, CV_64F);
	Mat Q(IMAGENUM*2, 1, CV_64F);
	for (int i = 0; i < uimgpts1.size(); i++)
	{
		// left image
		double A1 = L1.at<double>(8, 0)*ugpt0[i].gcp.X + L1.at<double>(9, 0)*ugpt0[i].gcp.Y + L1.at<double>(10, 0)*ugpt0[i].gcp.Z + 1;
		N.at<double>(0, 0) = (L1.at<double>(0, 0) + uimgpts1[i].imgPt.x*L1.at<double>(8, 0)) / (-A1);
		N.at<double>(0, 1) = (L1.at<double>(1, 0) + uimgpts1[i].imgPt.x*L1.at<double>(9, 0)) / (-A1);
		N.at<double>(0, 2) = (L1.at<double>(2, 0) + uimgpts1[i].imgPt.x*L1.at<double>(10, 0)) / (-A1);

		N.at<double>(1, 0) = (L1.at<double>(4, 0) + uimgpts1[i].imgPt.y*L1.at<double>(8, 0)) / (-A1);
		N.at<double>(1, 1) = (L1.at<double>(5, 0) + uimgpts1[i].imgPt.y*L1.at<double>(9, 0)) / (-A1);
		N.at<double>(1, 2) = (L1.at<double>(6, 0) + uimgpts1[i].imgPt.y*L1.at<double>(10, 0)) / (-A1);

		Q.at<double>(0, 0) = (L1.at<double>(3, 0) + uimgpts1[i].imgPt.x) / A1;
		Q.at<double>(1, 0) = (L1.at<double>(7, 0) + uimgpts1[i].imgPt.y) / A1;

		// image 2
		double A2 = L2.at<double>(8, 0)*ugpt0[i].gcp.X + L2.at<double>(9, 0)*ugpt0[i].gcp.Y + L2.at<double>(10, 0)*ugpt0[i].gcp.Z + 1;
		N.at<double>(2, 0) = (L2.at<double>(0, 0) + uimgpts2[i].imgPt.x*L2.at<double>(8, 0)) / (-A2);
		N.at<double>(2, 1) = (L2.at<double>(1, 0) + uimgpts2[i].imgPt.x*L2.at<double>(9, 0)) / (-A2);
		N.at<double>(2, 2) = (L2.at<double>(2, 0) + uimgpts2[i].imgPt.x*L2.at<double>(10, 0)) / (-A2);

		N.at<double>(3, 0) = (L2.at<double>(4, 0) + uimgpts2[i].imgPt.y*L2.at<double>(8, 0)) / (-A2);
		N.at<double>(3, 1) = (L2.at<double>(5, 0) + uimgpts2[i].imgPt.y*L2.at<double>(9, 0)) / (-A2);
		N.at<double>(3, 2) = (L2.at<double>(6, 0) + uimgpts2[i].imgPt.y*L2.at<double>(10, 0)) / (-A2);

		Q.at<double>(2, 0) = (L2.at<double>(3, 0) + uimgpts2[i].imgPt.x) / A2;
		Q.at<double>(3, 0) = (L2.at<double>(7, 0) + uimgpts2[i].imgPt.y) / A2;
		
		Mat S = (N.t()*N).inv()*N.t()*Q;
		gcpInfo gpt;
		gpt.gcpIndex = ugpt0[i].gcpIndex;
		gpt.gcp.X = S.at<double>(0, 0);
		gpt.gcp.Y = S.at<double>(1, 0);
		gpt.gcp.Z = S.at<double>(2, 0);
		uGpt.push_back(gpt);
	}
	return uGpt;
}

int getCorrImgPtIndex(int imagePtIndex, vector<imgPtInfo> imgPtArray)
{
	for (int i = 0; i < imgPtArray.size(); i++)
	{
		if (imagePtIndex == imgPtArray[i].imgPtIndex)
			return i;
	}
	//cout << "not found the corresponding point " << imagePtIndex << endl;
	return -1;
}

int getCorrespondingGCPIndex(int imagePtIndex, gcpInfo* gcpArray, int gcpNum)
{
	for (int i = 0; i < gcpNum; i++)
	{
		if (imagePtIndex == gcpArray[i].gcpIndex)
			return i;
	}
	//cout << "not found the corresponding point " << imagePtIndex << endl;
	return -1;
}

double getMaxDiffL(Mat L0, Mat L)
{
	double maxDiff = 0;
	for (int i = 0; i < L0.rows; i++)
	{
		double diff = abs(L0.at<double>(i, 0) - L.at<double>(i, 0));
		if (diff>maxDiff)
		{
			maxDiff = diff;
		}
	}
	return maxDiff;
}
double getMaxDiffDist(vector<gcpInfo> XYZ0, vector<gcpInfo> XYZ1)
{
	double maxDist = 0;
	for (int i = 0; i < XYZ0.size(); i++)
	{
		double dist = sqrt((XYZ0[i].gcp.X - XYZ1[i].gcp.X)*(XYZ0[i].gcp.X - XYZ1[i].gcp.X) + (XYZ0[i].gcp.Y - XYZ1[i].gcp.Y)*(XYZ0[i].gcp.Y - XYZ1[i].gcp.Y) + (XYZ0[i].gcp.Z - XYZ1[i].gcp.Z)*(XYZ0[i].gcp.Z - XYZ1[i].gcp.Z));
		if (dist>maxDist)
		{
			maxDist = dist;
		}
	}
	return maxDist;
}
Mat getOteriorPara(Mat L)
{
	double temp = sqrt(L.at<double>(8, 0)*L.at<double>(8, 0) + L.at<double>(9, 0)*L.at<double>(9, 0) + L.at<double>(10, 0)*L.at<double>(10, 0));
	double a3 = L.at<double>(8, 0) / temp;
	double b3 = L.at<double>(9, 0) / temp;
	double c3 = L.at<double>(10, 0) / temp;
	Mat angle(6, 1, CV_64F);
	angle.at<double>(0, 0) = atan(a3 / c3);//fai
	angle.at<double>(1, 0) = asin(-b3);//omega
	angle.at<double>(2, 0) = atan(b3);//kapa
	Mat L3(3, 1, CV_64F);
	L3.at<double>(0, 0) = -L.at<double>(3, 0);
	L3.at<double>(1, 0) = -L.at<double>(7, 0);
	L3.at<double>(2, 0) = -1;
	Mat A(3, 3, CV_64F);
	A.at<double>(0, 0) = L.at<double>(0, 0);
	A.at<double>(0, 1) = L.at<double>(1, 0);
	A.at<double>(0, 2) = L.at<double>(2, 0);
	A.at<double>(1, 0) = L.at<double>(4, 0);
	A.at<double>(1, 1) = L.at<double>(5, 0);
	A.at<double>(1, 2) = L.at<double>(6, 0);
	A.at<double>(2, 0) = L.at<double>(8, 0);
	A.at<double>(2, 1) = L.at<double>(9, 0);
	A.at<double>(2, 2) = L.at<double>(10, 0);
	Mat X = A.inv()*L3;
	angle.at<double>(3,0) = X.at<double>(0,0);
	angle.at<double>(4, 0) = X.at<double>(1, 0);
	angle.at<double>(5, 0) = X.at<double>(2, 0);
	return angle;
}
void getDsDbeta(Mat L, InParameters inPara)
{
	double temp = L.at<double>(8, 0)*L.at<double>(8, 0) + L.at<double>(9, 0)*L.at<double>(9, 0) + L.at<double>(10, 0)*L.at<double>(10, 0);
	double r3 = sqrt(1 / temp);
	double A = r3*r3*(L.at<double>(0, 0)*L.at<double>(0, 0) + L.at<double>(1, 0)*L.at<double>(1, 0) + L.at<double>(2, 0)*L.at<double>(2, 0)) - inPara.x0*inPara.x0;
	double B = r3*r3*(L.at<double>(4, 0)*L.at<double>(4, 0) + L.at<double>(5, 0)*L.at<double>(5, 0) + L.at<double>(6, 0)*L.at<double>(6, 0)) - inPara.y0*inPara.y0;
	double C = r3*r3*(L.at<double>(0, 0)*L.at<double>(4, 0) + L.at<double>(1, 0)*L.at<double>(5, 0) + L.at<double>(2, 0)*L.at<double>(6, 0)) - inPara.x0*inPara.y0;
	double beta = asin(sqrt(C*C / (A*B)));
	double ds = sqrt(A / B) - 1;
	double fx = sqrt(A*cos(beta)*cos(beta));
	double b1;
}
int main(void)
{
	FILE *imgPtFile1 = fopen("5506.txt", "r");
	FILE *imgPtFile2 = fopen("point.txt", "r");
	FILE *gcpFile = fopen("gcp", "r");
	if (!imgPtFile1 || !imgPtFile2 || !gcpFile)
	{
		cout << "file not found" << endl;
		return -1;
	}
	//read file info
	char buff[255];
	for (int i = 0; i < 4; i++)
	{
		fgets(buff, 255, imgPtFile1);
		fgets(buff, 255, imgPtFile2);
	}
	//read ground control points coordinates and establish search list
	int gcpNum = 0;
	cout << "read ground point ";
	fscanf(gcpFile, "%d", &gcpNum);
	gcpInfo* gcpArray = new gcpInfo[gcpNum];
	for (int j = 0; j < gcpNum; j++)
	{
		fscanf(gcpFile, "%d %lf %lf %lf", &gcpArray[j].gcpIndex, &gcpArray[j].gcp.X, &gcpArray[j].gcp.Y, &gcpArray[j].gcp.Z);
		gcpArray[j].gcp.Z = -gcpArray[j].gcp.Z;
	}
	cout << "ok " << endl;
	cout << "read image point ";
	//read right image coordinates and establish search list
	vector<imgPtInfo> imgPtArray2;
	imgPtInfo imgPt2;
	while ((fscanf(imgPtFile2, " %d %lf %lf", &imgPt2.imgPtIndex, &imgPt2.imgPt.x, &imgPt2.imgPt.y)) == 3)
	{
		imgPtArray2.push_back(imgPt2);
	}
	//read left image coordinates
	
	//control point image coordinates
	vector<imgPtInfo> imgPtVec1;
	vector<imgPtInfo> imgPtVec2;

	vector<imgPtInfo> imgPtUnknown1;
	vector<imgPtInfo> imgPtUnknown2;

	vector<gcpInfo> gcpVec;
	imgPtInfo imgPt1;
	while ((fscanf(imgPtFile1, " %d %lf %lf", &imgPt1.imgPtIndex, &imgPt1.imgPt.x, &imgPt1.imgPt.y)) == 3)
	{
		int imgPtIndex = getCorrImgPtIndex(imgPt1.imgPtIndex, imgPtArray2);
		int gcpIndex = getCorrespondingGCPIndex(imgPt1.imgPtIndex, gcpArray, gcpNum);
		//the point is valid only when it appears in both images
		if (imgPtIndex > 0)
		{
			if (imgPt1.imgPtIndex<100)
			{
				imgPtUnknown1.push_back(imgPt1);
				imgPtUnknown2.push_back(imgPt2);
			}
			if (gcpIndex > 0)
			{
				imgPtVec1.push_back(imgPt1);
				imgPtVec2.push_back(imgPtArray2[imgPtIndex]);
				gcpVec.push_back(gcpArray[gcpIndex]);
			}
		}
	}
	cout << "ok " << endl;
	Mat L10 = getL0(gcpVec, imgPtVec1);
	Mat L20 = getL0(gcpVec, imgPtVec2);

	InParameters inPara1 = getx0y0(L10);
	InParameters inPara2 = getx0y0(L20);

	vector<imgPtInfo> gcpsRes1;
	vector<imgPtInfo> gcpsRes2;

	Mat L1 = getAccurateL(L10, inPara1, gcpVec, imgPtVec1, gcpsRes1);
	Mat L2 = getAccurateL(L20, inPara2, gcpVec, imgPtVec2, gcpsRes2);
	//calculate L of left image
	int iter = 0;
	double diff1 = getMaxDiffL(L10,L1);
	while (diff1 > 0.001)
	{
		Mat oriL1;
		L1.copyTo(oriL1);
		inPara1 = getx0y0(oriL1);
		L1 = getAccurateL(oriL1, inPara1, gcpVec, imgPtVec1, gcpsRes1);
		diff1 = getMaxDiffL(L1, oriL1);
		iter++;
		if (iter > 20)
		{
			cout << "error:iterate too many times" << endl;
			break;
		}
	}
	//output left image L residual
	getAccurateL(L1, inPara1, gcpVec, imgPtVec1, gcpsRes1, true, "ResL1");

	//calculate L of right image
	iter = 0;
	double diff2 = getMaxDiffL(L20, L2);
	while (diff2 > 0.001)
	{
		Mat oriL2;
		L2.copyTo(oriL2);
		inPara2 = getx0y0(oriL2);
		L2 = getAccurateL(oriL2, inPara2, gcpVec, imgPtVec2, gcpsRes2);
		diff2 = getMaxDiffL(L2, oriL2);
		iter++;
		if (iter > 20)
		{
			cout << "error:iterate too many times" << endl;
			break;
		}
	}
	//output right image L residual
	getAccurateL(L2, inPara2, gcpVec, imgPtVec2, gcpsRes2, true, "ResL2");
	//calculate gcps m0
	calculateM0(gcpsRes1, gcpsRes2);
	cout << L1 << endl;
	cout << L2 << endl;
	//update image point
	updateImagePoint(L1, imgPtVec1, inPara1);
	updateImagePoint(L2, imgPtVec2, inPara2);

	//calculate XYZ
	vector<double> m0Vec;
	vector<gcpInfo> XYZ0Vec = getX0Y0Z0(L1, L2, imgPtUnknown1, imgPtUnknown2);
	vector<gcpInfo> XYZVec = getAccurateXYZ(L1, imgPtUnknown1, L2, imgPtUnknown2, XYZ0Vec);
	double maxDist = getMaxDiffDist(XYZ0Vec, XYZVec);
	iter = 0;
	while (maxDist > 0.1)
	{
		vector<gcpInfo> tempXYZ = XYZVec;
		XYZVec = getAccurateXYZ(L1, imgPtUnknown1, L2, imgPtUnknown2, tempXYZ);
		maxDist = getMaxDiffDist(XYZVec, tempXYZ);
		iter++;
		if (iter > 20)
		{
			cout << "error:iterate too many times" << endl;
			break;
		}
	}
	FILE *fugp = fopen("objectCoor", "w");
	fprintf(fugp, "待定点的物方空间坐标为\n");
	for (int i = 0; i < XYZVec.size(); i++)
	{
		fprintf(fugp, "%d\t%lf\t%lf\t%lf\n", XYZVec[i].gcpIndex, XYZVec[i].gcp.X, XYZVec[i].gcp.Y, XYZVec[i].gcp.Z);
	}
	FILE *fRes = fopen("Lresult", "w");
	fprintf(fRes, "L1：\n");
	for (int i = 0; i < L1.rows; i++)
	{
		fprintf(fRes, "%g\t", L1.at<double>(i, 0));
	}
	fprintf(fRes, "\nL2：\n");
	for (int i = 0; i < L2.rows; i++)
	{
		fprintf(fRes, "%g\t", L2.at<double>(i, 0));
	}
	fprintf(fRes, "内方位元素为：\n");
	fprintf(fRes, "L1: %lf\t%lf\t%lf\n", inPara1.x0, inPara1.y0, inPara1.f);
	fprintf(fRes, "L2: %lf\t%lf\t%lf\n", inPara2.x0, inPara2.y0, inPara2.f);
	Mat outeriorPara1 = getOteriorPara(L1);
	Mat outeriorPara2 = getOteriorPara(L2);
	fprintf(fRes, "外方位元素为：\n");
	fprintf(fRes, "L1:\n");
	for (int i = 0; i < 6; i++)
	{
		fprintf(fRes, "%lf\t", outeriorPara1.at<double>(i, 0));
	}
	fprintf(fRes, "\nL2:\n");
	for (int i = 0; i < 6; i++)
	{
		fprintf(fRes, "%lf\t", outeriorPara2.at<double>(i, 0));
	}
	
	fclose(fRes);
	system("pause");
	return 0;
}