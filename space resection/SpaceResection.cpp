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
}ptGround;//control point on ground
typedef struct{
	double x;
	double y;
}ptImage;//point in image
typedef struct{
	double Xs, Ys, Zs;
	double U, W, K;
}outPosParameters;
typedef struct{
	double a11, a12, a13, a14, a15, a16;
	double a21, a22, a23, a24, a25, a26;
}lfA;
outPosParameters getdetPara(double f, double H, vector<ptGround>ptGs, vector<ptImage>ptIs,outPosParameters outpos,cv::Mat &matA,cv::Mat &matL)
{
	//calculate R matrix
	double a1, a2, a3, b1, b2, b3, c1, c2, c3;
	double x0 = 0;
	double y0 = 0;
	a1 = cos(outpos.U)*cos(outpos.K) - sin(outpos.U)*sin(outpos.W)*sin(outpos.K);//a1
	a2 = -cos(outpos.U)*sin(outpos.K) - sin(outpos.U)*sin(outpos.W)*cos(outpos.K);//a2
	a3 = -sin(outpos.U)*cos(outpos.W);//a3
	b1 = cos(outpos.W)*sin(outpos.K);//b1
	b2 = cos(outpos.W)*cos(outpos.K);//b2
	b3 = -sin(outpos.W);//b3
	c1 = sin(outpos.U)*cos(outpos.K) + cos(outpos.U)*sin(outpos.W)*sin(outpos.K);//c1
	c2 = -sin(outpos.U)*sin(outpos.K) + cos(outpos.U)*sin(outpos.W)*cos(outpos.K);//c2
	c3 = cos(outpos.U)*cos(outpos.W);//c3
	printf("[%lf,%lf,%lf\n%lf,%lf,%lf\n%lf,%lf,%lf]\n", a1, a2, a3, b1, b2, b3, c1, c2, c3);
	vector<ptImage>ptEstimates;
	ptImage ptEstimate;
	vector<lfA> As;
	lfA A;
	for (int i = 0; i < ptIs.size(); i++)
	{
		double _X = a1 * (ptGs[i].X - outpos.Xs) + b1 * (ptGs[i].Y - outpos.Ys) + c1 * (ptGs[i].Z - outpos.Zs);
		double _Y = a2 * (ptGs[i].X - outpos.Xs) + b2 * (ptGs[i].Y - outpos.Ys) + c2 * (ptGs[i].Z - outpos.Zs);
		double _Z = a3 * (ptGs[i].X - outpos.Xs) + b3 * (ptGs[i].Y - outpos.Ys) + c3 * (ptGs[i].Z - outpos.Zs);
		//x0=0  y0=0
		ptEstimate.x = -1 * f*_X / _Z;
		ptEstimate.y = -1 * f*_Y / _Z;
		ptEstimates.push_back(ptEstimate);
		A.a11 = (a1*f + a3*ptIs[i].x) / _Z;
		A.a12 = (b1*f + b3*ptIs[i].x) / _Z;
		A.a13 = (c1*f + c3*ptIs[i].x) / _Z;
		A.a21 = (a2*f + a3*ptIs[i].y) / _Z;
		A.a22 = (b2*f + b3*ptIs[i].y) / _Z;
		A.a23 = (c2*f + c3*ptIs[i].y) / _Z;
		A.a14 = ptIs[i].y*sin(outpos.W) - (ptIs[i].x / f*(ptIs[i].x*cos(outpos.K) - ptIs[i].y*sin(outpos.K)) + f*cos(outpos.K))*cos(outpos.W);
		A.a15 = -1 * f*sin(outpos.K) - ptIs[i].x / f*(ptIs[i].x*sin(outpos.K) + ptIs[i].y*cos(outpos.K));
		A.a16 = ptIs[i].y;
		A.a24 = -1.0*ptIs[i].x*sin(outpos.W) - (ptIs[i].y / f*(ptIs[i].x*cos(outpos.K) - ptIs[i].y*sin(outpos.K)) - f*sin(outpos.K))*cos(outpos.W);
		A.a25 = -1.0 * f*cos(outpos.K) - ptIs[i].y / f*(ptIs[i].x*sin(outpos.K) + ptIs[i].y*cos(outpos.K));
		A.a26 = -1.0 * ptIs[i].x;
		/*A.a11 = -1 * f / H*cos(outpos.K);
		A.a12 = -1 * f / H*sin(outpos.K);
		A.a13 = -1*ptIs[i].x / H;
		A.a14 = -1 * (f + ptIs[i].x*ptIs[i].x / f)*cos(outpos.K) + ptIs[i].x*ptIs[i].y*sin(outpos.K) / f;
		A.a15 = -1.0 * ptIs[i].x*ptIs[i].y*cos(outpos.K) / f - (f + ptIs[i].x*ptIs[i].x / f)*sin(outpos.K);
		A.a16 = ptIs[i].y;
		A.a21 = -1*A.a12;
		A.a22 = A.a11;
		A.a23 = -1 * ptIs[i].y / H;
		A.a24 = -1 * ptIs[i].x*ptIs[i].y*cos(outpos.K) / f + (f + ptIs[i].y*ptIs[i].y / f)*sin(outpos.K);
		A.a25 = -1 * (f + ptIs[i].y*ptIs[i].y / f)*cos(outpos.K) - ptIs[i].x*ptIs[i].y*sin(outpos.K) / f*sin(outpos.K);
		A.a26 = -1 * ptIs[i].x;*/

		As.push_back(A);
	}
	
	for (int i = 0; i < ptIs.size(); i++)
	{
		matA.at<double>(2 * i, 0) = As[i].a11;
		matA.at<double>(2 * i, 1) = As[i].a12;
		matA.at<double>(2 * i, 2) = As[i].a13;
		matA.at<double>(2 * i, 3) = As[i].a14;
		matA.at<double>(2 * i, 4) = As[i].a15;
		matA.at<double>(2 * i, 5) = As[i].a16;

		matA.at<double>(2 * i + 1, 0) = As[i].a21;
		matA.at<double>(2 * i + 1, 1) = As[i].a22;
		matA.at<double>(2 * i + 1, 2) = As[i].a23;
		matA.at<double>(2 * i + 1, 3) = As[i].a24;
		matA.at<double>(2 * i + 1, 4) = As[i].a25;
		matA.at<double>(2 * i + 1, 5) = As[i].a26;
		matL.at<double>(2 * i, 0) = ptIs[i].x - ptEstimates[i].x;
		matL.at<double>(2 * i + 1, 0) = ptIs[i].y - ptEstimates[i].y;
	}
	cv::Mat matX = (matA.t()*matA).inv()*matA.t()*matL;
	
	outPosParameters delta;
	delta.Xs = matX.at<double>(0, 0);
	delta.Ys = matX.at<double>(1, 0);
	delta.Zs = matX.at<double>(2, 0);
	delta.U = matX.at<double>(3, 0);
	delta.W = matX.at<double>(4, 0);
	delta.K = matX.at<double>(5, 0);
	return delta;
}
void updatePara(outPosParameters &outPara, outPosParameters delta)
{
	outPara.K += delta.K;
	outPara.U += delta.U;
	outPara.W += delta.W;
	outPara.Xs += delta.Xs;
	outPara.Ys += delta.Ys;
	outPara.Zs += delta.Zs;
	printf("outside para: %lf,%lf,%lf,%lf,%lf,%lf\n", outPara.Xs, outPara.Ys, outPara.Zs, outPara.U, outPara.W, outPara.K);
	printf("delta: %lf,%lf,%lf,%lf,%lf,%lf\n", delta.Xs, delta.Ys, delta.Zs, delta.U, delta.W, delta.K);
}
void outPutM(outPosParameters delta,int n,cv::Mat matA,cv::Mat matL)
{
	cv::Mat x(6, 1, CV_64F);
	x.at<double>(0, 0) = delta.Xs;
	x.at<double>(1, 0) = delta.Ys;
	x.at<double>(2, 0) = delta.Zs;
	x.at<double>(3, 0) = delta.U;
	x.at<double>(4, 0) = delta.W;
	x.at<double>(5, 0) = delta.K;
	cv::Mat vv(8, 1, CV_64F);
	vv= matA*x - matL;
	cv::Mat v = vv.t()*vv;
	double sumV = v.at<double>(0.0);
	double m0 = sqrt(sumV / (2 * n - 6))*1000;
	double m[6];
	cv::Mat Q = (matA.t()*matA).inv();
	cout << "外方位元素中误差为" << endl;
	for (int i = 0; i < 6; i++)
	{
		m[i]=sqrt(Q.at<double>(i, i))*m0;
		printf("%lf\n", m[i]);
	}
	printf("单位权中误差为:%lfmm\n", m0);
}
int main(void)
{
	double f = 153.24*0.001;//mm
	double m = 15000;
	double H = m*f;
	vector<ptGround> ptGroundControls;
	vector<ptImage> ptImageControls;
	FILE *fp = fopen("points.txt", "rb");
	if (!fp)
	{
		cout << "File" << fp << "not found" << endl;
		return -1;
	}
	ptGround ptGroundControl;
	ptImage ptImageControl;
	while ((fscanf(fp, "%lf %lf %lf %lf %lf", &ptImageControl.x, &ptImageControl.y, &ptGroundControl.X, &ptGroundControl.Y, &ptGroundControl.Z)) == 5)
	{
		ptImageControl.x *= 0.001;
		ptImageControl.y *= 0.001;
		ptGroundControls.push_back(ptGroundControl);
		ptImageControls.push_back(ptImageControl);//像主点坐标系中的坐标x0=0,y0=0
	}
	fclose(fp);
	outPosParameters outPara;
	outPara.K = 0;
	outPara.U = 0;
	outPara.W = 0;
	outPara.Zs = 0;
	outPara.Xs = 0;
	outPara.Ys = 0;
	for (int i = 0; i < ptGroundControls.size(); i++)
	{
		outPara.Xs += ptGroundControls[i].X;
		outPara.Ys += ptGroundControls[i].Y;
		outPara.Zs += ptGroundControls[i].Z;
	}
	outPara.Xs = outPara.Xs / ptGroundControls.size();
	outPara.Ys = outPara.Ys / ptGroundControls.size();
	outPara.Zs = outPara.Zs / ptGroundControls.size();
	outPara.Zs += H;
	cv::Mat matA((ptGroundControls.size() * 2), 6, CV_64F);
	cv::Mat matL(matA.rows, 1, CV_64F);
	outPosParameters delta = getdetPara(f, H, ptGroundControls, ptImageControls, outPara,matA,matL);
	int iteration = 0;
	while (abs(delta.K)>0.00003 || abs(delta.U) > 0.00003 || abs(delta.W) > 0.00003)
	{
		if (iteration > 10)
		{
			cout << "some faults have happened" << endl;
			break;
		}
		updatePara(outPara, delta);
		delta = getdetPara(f, H, ptGroundControls, ptImageControls, outPara,matA,matL);
		iteration += 1;
	}
	cout << "iteration="<<iteration << endl;
	updatePara(outPara, delta);
	int n = ptGroundControls.size();
	outPutM(delta,n,matA,matL);
	system("pause");
	return 0;
}