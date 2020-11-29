#include "ICP.h"
#include <vector>
#include "svd.h"
#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace gs;


void selfCheck()
{
	//rotate and translate a defined vector then registrate these two point set.
	

	clock_t start, finish;
	double totaltime = 0;

	FILE* f = fopen("hand-low-tri.ply", "r");//original data file
	FILE* newF = fopen("new-low-hand.ply", "w");//data file after rotation and translation
	FILE* resultFile = fopen("registed-result.ply", "w");//result data after icp registration
	//写文件头信息
	char fileInfo[255];
	for (int i = 0; i < 10; i++)
	{
		fgets(fileInfo, 255, f);
		fputs(fileInfo, newF);
		fputs(fileInfo, resultFile);
	}
	vector<Point*> oriPtData;
	float x, y, z;
	for (int j = 0; j < 55225; j++)
	{
		if ((fscanf(f, "%f %f %f \n", &x, &y, &z)) == 3)
		{
			Point* pt = new Point();
			pt->pos[0] = x;
			pt->pos[1] = y;
			pt->pos[2] = z;
			oriPtData.push_back(pt);
		}
	}
	//对原始数据进行旋转平移，并写入文件中存储
	float XRotateAngle = 30.0 / 180 * 3.1415926;
	float YRotateAngle = 30.0 / 180 * 3.1415926;
	float ZRotateAngle = 30.0 / 180 * 3.1415926;
	float rotationMatrix[9] = { 1, 0, 0, 0, cos(XRotateAngle), -sin(XRotateAngle), 0, sin(XRotateAngle), cos(XRotateAngle) };
	float translation[3] = { 0.5, 0, 0 };
	vector<Point*> newPtData;
	for (int i = 0; i < oriPtData.size(); i++)
	{
		Point* tempPt1 = new Point();
		Point* tempPt2 = new Point();
		rotate(oriPtData[i], rotationMatrix, tempPt1);
		translate(tempPt1, translation, tempPt2);
		newPtData.push_back(tempPt2);
		fprintf(newF, "%f %f %f\r", tempPt2->pos[0], tempPt2->pos[1], tempPt2->pos[2]);
	}
	printf("rotated point file writes successfully\n");
	//计算icp配准后的结果并且写入文件中
	start = clock();
	icp(newPtData, oriPtData);
	finish = clock();
	for (int i = 0; i < newPtData.size(); i++)
	{
		fprintf(resultFile, "%f %f %f\r", newPtData[i]->pos[0], newPtData[i]->pos[1], newPtData[i]->pos[2]);
	}
	printf("\n icp registrate successfully\n");
	//写面文件数据
	char buff[255];
	int t1, t2, t3, t4;
	while ((fscanf(f, "%d %d %d %d", &t1, &t2, &t3, &t4)) == 4)
	{
		fprintf(newF, "%d %d %d %d\r", t1, t2, t3, t4);
		fprintf(resultFile, "%d %d %d %d\r", t1, t2, t3, t4);
	}
	totaltime = (finish - start) / CLOCKS_PER_SEC;
	printf("\nThe total cost time is %lf\n", totaltime);
	fclose(f);
	fclose(newF);
	fclose(resultFile);
}

void registrate()
{
	//registrate different point set directly 

	clock_t start, finish;
	double totaltime = 0;

	FILE* flow = fopen("hand-low-tri.ply", "r");//original low resolution data file
	FILE* fhigh = fopen("new-high.ply", "r");//original high resolution data file
	FILE* resultFile = fopen("icp-result.ply", "w");//result data after icp registration
	//写文件头信息
	char fileInfo[255];
	for (int i = 0; i < 10; i++)
	{
		fgets(fileInfo, 255, flow);
		fputs(fileInfo, resultFile);
		fgets(fileInfo, 255, fhigh);
	}
	printf("file header write OK\n");
	vector<Point*> lowResPtData;
	vector<Point*> highResPtData;
	float x, y, z;
	for (int j = 0; j < 55225; j++)
	{
		if ((fscanf(flow, "%f %f %f \n", &x, &y, &z)) == 3)
		{
			Point* pt = new Point();
			pt->pos[0] = x;
			pt->pos[1] = y;
			pt->pos[2] = z;
			lowResPtData.push_back(pt);
		}
	}
	for (int j = 0; j < 96942; j++)
	{
		if ((fscanf(fhigh, "%f %f %f \n", &x, &y, &z)) == 3)
		{
			Point* pt = new Point();
			pt->pos[0] = x;
			pt->pos[1] = y;
			pt->pos[2] = z;
			highResPtData.push_back(pt);
		}
	}
	printf("point data read OK\n");
	//计算icp配准后的结果并且写入文件中
	start = clock();
	icp(highResPtData, lowResPtData);
	finish = clock();
	for (int i = 0; i < highResPtData.size(); i++)
	{
		fprintf(resultFile, "%f %f %f\r", highResPtData[i]->pos[0], highResPtData[i]->pos[1], highResPtData[i]->pos[2]);
	}
	printf("\n icp registrate successfully\n");
	//写面文件数据
	char buff[255];
	int t1, t2, t3, t4;
	while ((fscanf(fhigh, "%d %d %d %d", &t1, &t2, &t3, &t4)) == 4)
	{
		fprintf(resultFile, "%d %d %d %d\r", t1, t2, t3, t4);
	}
	totaltime = (finish - start) / CLOCKS_PER_SEC;
	printf("\nThe total cost time is %lf\n", totaltime);
	fclose(flow);
	fclose(fhigh);
	fclose(resultFile);
}
int main(void){
	selfCheck();
	system("pause");
	return 0;
}
