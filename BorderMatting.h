#ifndef BORDERMATTING_H_
#define BORDERMATTING_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <math.h>
using namespace std;
using namespace cv;
//存储点的信息
struct point {
	int x, y;
};
struct para_point {
	point p;
	int index, section;
	double delta, sigma;
};
struct inf_point {
	point p;
	int dis, area;
};
//存储参数
struct dands {
	int delta, sigma;
};
//一些类型的定义
typedef vector<double[30][10]> Energyfunction;
typedef vector<dands[30][10]> Record;
typedef vector<para_point> Contour;
typedef unordered_map<int, inf_point> Strip;
//轮廓上视为相邻的8个
#define nstep 8
const int nx[nstep] = { 0, 1, 0, -1, -1, -1, 1, 1 };
const int ny[nstep] = { 1, 0, -1, 0, -1, 1, -1, 1 };

#define COE 10000
//TU的width（根据论文中的实现，为6）
#define stripwidth 6
//L是41指的是边长，这里的L是边长的一般
#define L 20
//欧式距离为1的相邻点
#define rstep 4
const int rx[rstep] = { 0,1,0,-1 };
const int ry[rstep] = { 1,0,-1,0 };
#define MAXNUM 9999999;
//两个参数的分割层数
#define sigmaLevels  10
#define deltaLevels  40
class BorderMatting
{
public:
	BorderMatting();
	~BorderMatting();
	//borderMatting的构造函数，也是其对外提供的接口。
	void borderMatting(const Mat& oriImg, const Mat& mask, Mat& borderMask);
private:
	//利用深度优先搜索对轮廓进行参数计算
	void ParameterizationContour(const Mat& edge);
	//深度优先搜索遍历整个轮廓，并对 contour 进行构造。
	void dfs(int x, int y, const Mat& mask, Mat& amask);
	//初始化TU，用无序图来存储，hash 值就是其坐标值。
	void StripInit(const Mat& mask);
	//利用DP算法进行能量最小化，求 sigma 和 delta 的值
	void EnergyMinimization(const Mat& oriImg, const Mat& mask);
	//计算平滑项的值
	inline double varyTerm(double _ddelta, double _dsigma){ return (lamda1*pow(_ddelta, 2.0) + lamda2*pow(_dsigma, 2.0)) / 200; }
	//变量初始化
	void init(const Mat& img);
	//计算从一个起始点开始，整个轮廓的数据项之和
	double dataTerm(int index, point p, double uf, double ub, double cf, double cb, double delta, double sigma, const Mat& gray);
	//利用bfs计算所有alpha的值
	void CalculateMask(Mat& bordermask, const Mat& mask);
	//显示处理后的图片
	void display(const Mat& oriImg, const Mat& mask);
	//公式13中两个lamda的值
	const int lamda1 = 50;
	const int lamda2 = 1000;
	//独立轮廓个数
	int sections; 
	//以轮廓上不同点为中心的区域个数（即轮廓上点的个数）
	int areaCount;
	//轮廓
	Contour contour; 
	//TU
	Strip strip; 
	int rows, cols;
	//使用DP算法时存储中间值
	double ef[5000][deltaLevels][sigmaLevels];
	dands rec[5000][deltaLevels][sigmaLevels];
	vector<dands> vecds;
};

#endif