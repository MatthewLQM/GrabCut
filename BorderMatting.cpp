#include "BorderMatting.h"
BorderMatting::BorderMatting(){}
BorderMatting::~BorderMatting(){}
//判断x是否在l和r直接。
inline bool outrange(int _x, int _l, int _r){
	if (_x<_l || _x>_r)	return true;
	else return false;
}
//变量初始化。
void BorderMatting::init(const Mat& _img){
	rows = _img.rows;
	cols = _img.cols;
	sections = 0;
	areaCount = 0;
	contour.clear();
	strip.clear();
	vecds.clear();
}
//利用 Canny 算法进行边缘检测，结果存放在 _rs 中。
void BorderDetection(const Mat& _img, Mat& _rs){
	Mat edges;
	Canny(_img, edges, 3, 9);
	edges.convertTo(_rs, CV_8UC1);
}
//深度优先搜索遍历整个轮廓，并对 contour 进行构造。
void BorderMatting::dfs(int _x, int _y, const Mat& _edge, Mat& _color){
	//标记遍历到的点
	_color.at<uchar>(_x, _y) = 255;
	para_point pt;
	pt.p.x = _x; pt.p.y = _y; //坐标
	pt.index = areaCount++;//给轮廓上每一个点分配独立index
	pt.section = sections;//所属轮廓
	contour.push_back(pt); //放入轮廓vector
	//枚举(x,y)相邻点
	for (int i = 0; i < nstep; i++) {
		int zx = nx[i], zy = ny[i];
		int newx = _x + zx, newy = _y + zy;
		//超出图像范围
		if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1)) continue;
		//不是轮廓上的点
		if (_edge.at<uchar>(newx, newy) == 0)continue;
		//已经被遍历过
		if (_color.at<uchar>(newx, newy) != 0)continue;
		//从(newx,newy)出发，继续深搜遍历轮廓
		dfs(newx, newy, _edge, _color);
	}
}
//利用深度优先搜索对轮廓进行参数计算。
void BorderMatting::ParameterizationContour(const Mat& _edge)
{
	int rows = _edge.rows, cols = _edge.cols;
	sections = 0; 
	areaCount = 0; 
	//遍历标记
	Mat color(_edge.size(), CV_8UC1, Scalar(0));
	bool flag = false;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			//(i,j)是轮廓上的点且未被遍历过
			if (_edge.at<uchar>(i, j) != 0 && color.at<uchar>(i, j) == 0){
				//对其进行遍历并使轮廓数加一
				dfs(i, j, _edge, color);
				sections++;
			}
}
//初始化TU，用无序图来存储，hash 值就是其坐标值。
void BorderMatting::StripInit(const Mat& _mask){
	Mat color(_mask.size(), CV_32SC1, Scalar(0));//遍历标记
	//从轮廓出发，宽搜标记TU，标记TU所属区域————对应的中心轮廓点
	//初始化队列：加入轮廓上所有点
	vector<point> queue;
	for (int i = 0; i < contour.size(); i++){
		inf_point ip;
		ip.p = contour[i].p; //坐标
		ip.dis = 0; //距离中心点的欧氏距离
		ip.area = contour[i].index; //所属区域
		strip[ip.p.x*COE + ip.p.y] = ip; //将点加入条带，key（hash）值为其坐标
		queue.push_back(ip.p); //将点加入队列
		color.at<int>(ip.p.x, ip.p.y) = ip.area + 1; //遍历标记：区域号+1
	}
	//宽搜遍历TU，将
	int l = 0;
	while (l < queue.size()){
		point p = queue[l++]; //取出点
		inf_point ip = strip[p.x*COE + p.y]; //从strip中得到相关信息
		//只遍历TU内的点
		if (abs(ip.dis) >= stripwidth) break;
		int x = ip.p.x, y = ip.p.y;
		//枚举相邻点
		for (int i = 0; i < rstep; i++)	{
			int newx = x + rx[i], newy = y + ry[i];
			//超出图像范围
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))	continue;
			inf_point nip;
			//如果已经被遍历过
			if (color.at<int>(newx, newy) != 0)	continue;
			else nip.p.x = newx; nip.p.y = newy;
			nip.dis = abs(ip.dis) + 1;//欧式距离+1
			//如果该点属于背景，欧氏距离取负
			if ((_mask.at<uchar>(newx, newy) & 1) != 1) nip.dis = -nip.dis;
			nip.area = ip.area;
			//加入TU中。
			strip[nip.p.x*COE + nip.p.y] = nip;
			queue.push_back(nip.p);
			//遍历标记：区域号+1
			color.at<int>(newx, newy) = nip.area + 1;
		}
	}
}
//一维高斯密度函数
inline double Gaussian(double _x, double _delta, double _sigma){
	const double PI = 3.14159;
	double e = exp(-(pow(_x - _delta, 2.0) / (2.0*_sigma)));
	double rs = 1.0 / (pow(_sigma, 0.5)*pow(2.0*PI, 0.5))*e;
	return rs;
}
//论文中公式15（1）
inline double ufunc(double _a, double _uf, double _ub){
	return (1.0 - _a)*_ub + _a*_uf;
}
//论文中公式15（2）
inline double cfunc(double _a, double _cf, double _cb){
	return pow(1.0 - _a, 2.0)*_cb + pow(_a, 2.0)*_cf;
}
//sigmoid函数,当做soft step-function（论文图6.c)
inline double Sigmoid(double _r, double _delta, double _sigma){
	double rs = -(_r - _delta) / _sigma;
	rs = exp(rs);
	rs = 1.0 / (1.0 + rs);
	return rs;
}
//计算某一个点的数据项值
inline double dataTermPoint(inf_point _ip, float _I, double _delta, double _sigma, double _uf, double _ub, double _cf, double _cb){
	double alpha = Sigmoid((double)_ip.dis / (double)stripwidth, _delta, _sigma);
	double D = Gaussian(_I, ufunc(alpha, _uf, _ub), cfunc(alpha, _cf, _cb));
	D = -log(D) / log(2.0);
	return D;
}
//计算从一个起始点开始，整个轮廓的数据项之和
double BorderMatting::dataTerm(int _index, point _p, double _uf, double _ub, double _cf, double _cb, double _delta, double _sigma, const Mat& _gray){
	vector<inf_point> queue;
	map<int, bool> color;
	double sum = 0;
	inf_point ip = strip[_p.x*COE + _p.y]; //从strip中获取中心点信息
	sum += dataTermPoint(ip, _gray.at<float>(ip.p.x, ip.p.y), _delta, _sigma, _uf, _ub, _cf, _cb);
	queue.push_back(ip);//加入队列
	color[ip.p.x*COE + ip.p.y] = true;//标记遍历
	//宽搜遍历以p为中心点的区域
	int l = 0;
	while (l < queue.size())
	{
		inf_point ip = queue[l++];
		if (abs(ip.dis) >= stripwidth)break;
		int x = ip.p.x;
		int y = ip.p.y;
		//遍历相邻点
		for (int i = 0; i < rstep; i++)	{
			int newx = x + rx[i], newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))	continue;
			if (color[newx*COE + newy])	continue;
			inf_point newip = strip[newx*COE + newy];
			//属于以p为中心点的区域
			if (newip.area == _index) 
				sum += dataTermPoint(newip, _gray.at<float>(newx, newy), _delta, _sigma, _uf, _ub, _cf, _cb);
			queue.push_back(newip);//加入队列
			color[newx*COE + newy] = true;//标记遍历
		}
	}
	return sum;
}
/*计算L*L区域的前背景均值和方差*/
void calMeanAndCov(point _p, const Mat& _gray, const Mat& _mask, double& _uf, double& _ub, double& _cf, double& _cb){
	int len = L;
	double sumf = 0, sumb = 0;
	int cntf = 0, cntb = 0;
	int rows = _gray.rows, cols = _gray.cols;
	//计算均值
	for (int x = _p.x - len; x <= _p.x + len; x++)
		for (int y = _p.y - len; y <= _p.y + len; y++)
			if (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1))){
				float g = _gray.at<float>(x, y);
				//背景
				if ((_mask.at<uchar>(x, y) & 1) == 0){
					sumb += g;
					cntb++;
				}
				//前景
				else {
					sumf += g;
					cntf++;
				}
			}

	_uf = (double)sumf / (double)cntf; //前景均值
	_ub = (double)sumb / (double)cntb; //背景均值
	//计算方差
	_cf = 0;
	_cb = 0;
	for (int x = _p.x - len; x <= _p.x + len; x++)
		for (int y = _p.y - len; y <= _p.y + len; y++)
			if (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1))){
				float g = _gray.at<float>(x, y);
				//背景
				if ((_mask.at<uchar>(x, y) & 1) == 0)
					_cb += pow(g - _ub, 2.0);
				//前景
				else _cf += pow(g - _uf, 2.0);
			}
	_cf /= (double)cntf; //前景方差
	_cb /= (double)cntb; //背景方差
}
//计算sigma的离散值
inline double sigma(int _level){ return 0.1*(_level); }
//计算delta的离散值
inline double delta(int level) { return 0.025*level; }
//利用DP算法进行能量最小化，求 sigma 和 delta 的值
void BorderMatting::EnergyMinimization(const Mat& _oriImg, const Mat& _mask){
	//转换为灰度图
	Mat gray;
	cvtColor(_oriImg, gray, COLOR_BGR2GRAY);
	gray.convertTo(gray, CV_32FC1, 1.0 / 255.0);
	//能量最小化求每个区域的delta和sigma
	//枚举轮廓上每一个点，即条带各区域中心点
	for (int i = 0; i < contour.size(); i++) {
		para_point pp = contour[i];
		int index = pp.index;
		double uf, ub, cf, cb;
		//求L*L区域的前背景均值和方差
		calMeanAndCov(pp.p, gray, _mask, uf, ub, cf, cb);
		for (int d0 = 0; d0 < deltaLevels; d0++) //枚举delta
			for (int s0 = 0; s0 < sigmaLevels; s0++){ //枚举sigma
				double sigma0 = sigma(s0), delta0 = delta(d0);
				ef[index][d0][s0] = MAXNUM;
				//计算term D
				double D = dataTerm(index, pp.p, uf, ub, cf, cb, delta0, sigma0, gray);
				//计算能量方程:termD + termV
				if (index == 0) {
					ef[index][d0][s0] = D;
					continue;
				}
				for (int d1 = 0; d1 < deltaLevels; d1++)//枚举index-1时的delta
					for (int s1 = 0; s1 < sigmaLevels; s1++){//枚举index-1时的sigma
						double delta1 = delta(d1), sigma1 = sigma(s1);
						double Vterm = 0;
						if (contour[i - 1].section == pp.section){//与上一点属于同一轮廓
							Vterm = varyTerm(delta0 - delta1, sigma0 - sigma1);
						}
						double rs = ef[index - 1][d1][s1] + Vterm + D;
						if (rs < ef[index][d0][s0]) {
							dands ds;
							ds.sigma = s1; ds.delta = d1;
							ef[index][d0][s0] = rs;
							rec[index][d0][s0] = ds;
						}
					}
			}
	}
	//找总能量最小值
	double minE = MAXNUM;
	dands ds;
	//记录每个区域的delta和sigma
	vecds = vector<dands>(areaCount);
	for (int d0 = 0; d0< deltaLevels; d0++)
		for (int s0 = 0; s0 < sigmaLevels; s0++)
		{
			if (ef[areaCount - 1][d0][s0] < minE) {
				minE = ef[areaCount - 1][d0][s0];
				ds.delta = d0;
				ds.sigma = s0;
			}
		}
	//记录总能量最小时，每个区域的delta和sigma
	vecds[areaCount - 1] = ds;
	for (int i = areaCount - 2; i >= 0; i--){
		dands ds0 = vecds[i + 1];
		dands ds = rec[i + 1][ds0.delta][ds0.sigma];
		vecds[i] = ds;
	}
}
//调整alpha的值，如果太小，用0代替，如果太大，用1代替
inline double adjustA(double _a){
	if (_a < 0.01) return 0;
	if (_a > 0.99) return 1;
	return _a;
}
//利用bfs计算所有alpha的值
void BorderMatting::CalculateMask(Mat& _alphaMask, const Mat& _mask){
	_alphaMask = Mat(_mask.size(), CV_32FC1, Scalar(0));
	Mat visit(_mask.size(), CV_32SC1, Scalar(0));//遍历标记
	//从轮廓出发，宽搜遍历图像，计算alpha
	//初始化队列：加入轮廓上所有点											
	vector<inf_point> queue;
	for (int i = 0; i < contour.size(); i++){
		inf_point ip;
		ip.p = contour[i].p; //坐标
		ip.dis = 0; //距离中心点的欧氏距离
		ip.area = contour[i].index; //所属区域
		queue.push_back(ip); //将点加入队列
		visit.at<int>(ip.p.x, ip.p.y) = 1; //遍历标记
		//计算alpha
		dands ds = vecds[ip.area];
		double alpha = Sigmoid((double)ip.dis / (double)stripwidth, delta(ds.delta), sigma(ds.sigma));
		alpha = adjustA(alpha);//调整alpha
		_alphaMask.at<float>(ip.p.x, ip.p.y) = (float)alpha;
	}
	//宽搜遍历条带
	int l = 0;
	while (l < queue.size()) {
		inf_point ip = queue[l++]; //取出点
		int x = ip.p.x, y = ip.p.y;
		for (int i = 0; i < rstep; i++){//枚举相邻点
			int newx = x + rx[i], newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))	continue;
			if (visit.at<int>(newx, newy) != 0)	continue;
			inf_point nip;
			nip.p.x = newx; nip.p.y = newy;
			nip.dis = abs(ip.dis) + 1;//欧式距离+1
			if ((_mask.at<uchar>(newx, newy) & 1) != 1)	nip.dis = -nip.dis;
			nip.area = ip.area;
			queue.push_back(nip); //加入队列
			visit.at<int>(newx, newy) = 1; //遍历标记
			//计算alpha
			dands ds = vecds[nip.area];
			double alpha = Sigmoid((double)nip.dis / (double)stripwidth, delta(ds.delta), sigma(ds.sigma));
			alpha = adjustA(alpha);//调整alpha
			_alphaMask.at<float>(nip.p.x, nip.p.y) = (float)alpha;
		}
	}
}
//显示处理后的图片
void BorderMatting::display(const Mat& _originImage, const Mat& _alphaMask){
	vector<Mat> ch_img(3),ch_bg(3);
	//分离前景和背景通道，设背景色为黑色
	Mat img;
	_originImage.convertTo(img, CV_32FC3, 1.0 / 255.0);
	cv::split(img, ch_img);
	Mat bg = Mat(img.size(), CV_32FC3, Scalar(0, 0, 0));
	cv::split(bg, ch_bg);
	//根据alpha的值计算加上透明度以后的图像
	ch_img[0] = ch_img[0].mul(_alphaMask) + ch_bg[0].mul(1.0 - _alphaMask);
	ch_img[1] = ch_img[1].mul(_alphaMask) + ch_bg[1].mul(1.0 - _alphaMask);
	ch_img[2] = ch_img[2].mul(_alphaMask) + ch_bg[2].mul(1.0 - _alphaMask);
	//合并三通道
	Mat res;
	cv::merge(ch_img, res);
	//显示结果
	imshow("img", res);
}
//borderMatting的构造函数，也是其对外提供的接口。
void BorderMatting::borderMatting(const Mat& _originImage, const Mat& _mask, Mat& _alphaMask) {
	//初始化参数
	init(_originImage);
	//进行轮廓检测
	Mat edge = _mask & 1;
	edge.convertTo(edge, CV_8UC1, 255);
	BorderDetection(edge, edge);
	//轮廓参数化
	ParameterizationContour(edge);
	//计算TU矩阵
	Mat tmask;
	_mask.convertTo(tmask, CV_8UC1);
	StripInit(tmask);
	//利用DP算法进行能量最小化得到参数值
	EnergyMinimization(_originImage, _mask);
	//根据得到的参数值计算每个像素点的alpha
	CalculateMask(_alphaMask, _mask);
	//结果不明显，所以进行轻微的高斯滤波，使得结果显示范围增加
	GaussianBlur(_alphaMask, _alphaMask, Size(7, 7), 9);
	//显示 borderMatting 的结果
	display(_originImage, _alphaMask);
}