#include "GMM.h"
#include "GrabCut.h"
#include "CutGraph.h"
#include <iostream>
#include <limits>
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
using namespace cv;
using namespace std;

//计算 Beta 的值。根据论文中的公式（5）。 
static double calcuBeta(const Mat& _img) {
	double beta;
	double totalDiff = 0;
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			if (x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				totalDiff += diff.dot(diff);
			}
			if (y > 0 && x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				totalDiff += diff.dot(diff);
			}
			if (y > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				totalDiff += diff.dot(diff);
			}
			if (y > 0 && x < _img.cols - 1) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				totalDiff += diff.dot(diff);
			}
		}
	}
	totalDiff *= 2;
	if (totalDiff <= std::numeric_limits<double>::epsilon()) beta = 0;
	else beta = 1.0 / (2 * totalDiff / (8 * _img.cols*_img.rows - 6 * _img.cols - 6 * _img.rows + 4));
	return beta;
}
//计算相邻像素的权重差，由于对称性，八个点我们只需要计算四个方向。 
static void calcuNWeight(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma) {
	const double gammaDiv = _gamma / std::sqrt(2.0f);
	_l.create(_img.size(), CV_64FC1);
	_ul.create(_img.size(), CV_64FC1);
	_u.create(_img.size(), CV_64FC1);
	_ur.create(_img.size(), CV_64FC1);
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			if (x - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				_l.at<double>(y, x) = _gamma*exp(-_beta*diff.dot(diff));
			}
			else _l.at<double>(y, x) = 0;
			if (x - 1 >= 0 && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				_ul.at<double>(y, x) = gammaDiv*exp(-_beta*diff.dot(diff));
			}
			else _ul.at<double>(y, x) = 0;
			if (y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				_u.at<double>(y, x) = _gamma*exp(-_beta*diff.dot(diff));
			}
			else _u.at<double>(y, x) = 0;
			if (x + 1 < _img.cols && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				_ur.at<double>(y, x) = gammaDiv*exp(-_beta*diff.dot(diff));
			}
			else _ur.at<double>(y, x) = 0;
		}
	}
}
//根据输入矩阵设置 mask，矩阵外肯定是背景，矩阵内可能是前景。 
static void initMaskWithRect(Mat& _mask, Size _imgSize, Rect _rect) {
	_mask.create(_imgSize, CV_8UC1);
	_mask.setTo(MUST_BGD);
	_rect.x = _rect.x > 0 ? _rect.x : 0;
	_rect.y = _rect.y > 0 ? _rect.y : 0;
	_rect.width = _rect.x + _rect.width > _imgSize.width ? _imgSize.width - _rect.x : _rect.width;
	_rect.height = _rect.y + _rect.height > _imgSize.height ? _imgSize.height - _rect.y : _rect.height;
	(_mask(_rect)).setTo(Scalar(MAYBE_FGD));
}
//利用 kmeans 方法初始化 GMM 模型
static void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
	const int kmeansItCount = 10;
	Mat bgdLabels, fgdLabels;
	vector<Vec3f> bgdSamples, fgdSamples;
	Point p;
	for (p.y = 0; p.y < img.rows; p.y++){
		for (p.x = 0; p.x < img.cols; p.x++){
			if (mask.at<uchar>(p) == MUST_BGD || mask.at<uchar>(p) == MAYBE_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
			else
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
		}
	}
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	kmeans(_bgdSamples, GMM::K, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, GMM::K, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);
	//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
	bgdGMM.learningBegin();
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.learningEnd();

	fgdGMM.learningBegin();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.learningEnd();
}
//迭代循环第一步，为每个像素分配GMM中所属的高斯模型，保存在partIndex中。
static void assignGMMS(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, Mat& _partIndex) {
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(p);
			uchar t = _mask.at<uchar>(p);
			if (t == MUST_BGD || t == MAYBE_BGD)_partIndex.at<int>(p) = _bgdGMM.choice(color);
			else _partIndex.at<int>(p) = _fgdGMM.choice(color);
		}
	}
}
//迭代循环第二步，根据得到的结果计算GMM参数值。
static void learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _partIndex) {
	_bgdGMM.learningBegin();
	_fgdGMM.learningBegin();
	Point p;
	for (int i = 0; i < GMM::K; i++) {
		for (p.y = 0; p.y < _img.rows; p.y++) {
			for (p.x = 0; p.x < _img.cols; p.x++) {
				int tmp = _partIndex.at<int>(p);
				if (tmp == i) {
					if (_mask.at<uchar>(p) == MUST_BGD || _mask.at<uchar>(p) == MAYBE_BGD)
						_bgdGMM.addSample(tmp, _img.at<Vec3b>(p));
					else
						_fgdGMM.addSample(tmp, _img.at<Vec3b>(p));
				}
			}
		}
	}
	_bgdGMM.learningEnd();
	_fgdGMM.learningEnd();
}
//根据得到的结果构造图，使用助教给的现成的库 Done
static void getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur, CutGraph& _graph) {
	int vCount = _img.cols*_img.rows;
	int eCount = 2 * (4 * vCount - 3 * _img.cols - 3 * _img.rows + 2);
	_graph = CutGraph(vCount, eCount);
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			int vNum = _graph.addVertex();
			Vec3b color = _img.at<Vec3b>(p);
			double wSource = 0, wSink = 0;
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				wSource = -log(_bgdGMM.tWeight(color));
				wSink = -log(_fgdGMM.tWeight(color));
			}
			else if (_mask.at<uchar>(p) == MUST_BGD) wSink = _lambda;
			else wSource = _lambda;
			_graph.addVertexWeights(vNum, wSource, wSink);
			if (p.x > 0) {
				double w = _l.at<double>(p);
				_graph.addEdges(vNum, vNum - 1, w);
			}
			if (p.x > 0 && p.y > 0) {
				double w = _ul.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols - 1, w);
			}
			if (p.y > 0) {
				double w = _u.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols, w);
			}
			if (p.x < _img.cols - 1 && p.y > 0) {
				double w = _ur.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols + 1, w);
			}
		}
	}
}
//进行分割 Done
static void estimateSegmentation(CutGraph& _graph, Mat& _mask) {
	_graph.maxFlow();
	Point p;
	for (p.y = 0; p.y < _mask.rows; p.y++) {
		for (p.x = 0; p.x < _mask.cols; p.x++) {
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				if (_graph.isSourceSegment(p.y*_mask.cols + p.x))
					_mask.at<uchar>(p) = MAYBE_FGD;
				else _mask.at<uchar>(p) = MAYBE_BGD;
			}
		}
	}
}
GrabCut2D::~GrabCut2D(void) {}
//GrabCut 主函数
void GrabCut2D::GrabCut(InputArray _img, InputOutputArray _mask, Rect rect,	InputOutputArray _bgdModel, InputOutputArray _fgdModel,
	int iterCount, int mode) {
	std::cout << "Execute GrabCut Function: Please finish the code here!" << std::endl;
	//一.参数解释：
	//输入：
	//cv::InputArray _img,     :输入的color图像(类型-cv:Mat)
	//cv::Rect rect            :在图像上画的矩形框（类型-cv:Rect) 
	//int iterCount :           :每次分割的迭代次数（类型-int)
	//中间变量
	//cv::InputOutputArray _bgdModel ：   背景模型（推荐GMM)（类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//cv::InputOutputArray _fgdModel :    前景模型（推荐GMM) （类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//输出:
	//cv::InputOutputArray _mask  : 输出的分割结果 (类型： cv::Mat)
	//二. 伪代码流程：
	//1.Load Input Image: 加载输入颜色图像;
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
	if (mode == GC_WITH_RECT)initMaskWithRect(mask, img.size(), rect);
	//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
	GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
	//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
	if (mode == GC_WITH_RECT || mode == GC_WITH_MASK)initGMMs(img, mask, bgdGMM, fgdGMM);
	if (iterCount <= 0)return;
	//6.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
	//计算平滑项，数据项的计算在GMM模型中实现
	const double gamma = 50;
	const double beta = calcuBeta(img);
	Mat leftW, upleftW, upW, uprightW;
	calcuNWeight(img, leftW, upleftW, upW, uprightW, beta, gamma);
	//7.Estimate Segmentation(调用maxFlow库进行分割)
	Mat compIdxs(img.size(), CV_32SC1);
	const double lambda = 9 * gamma;
	//进行迭代
	for (int i = 0; i < iterCount; i++) {
		CutGraph graph;
		assignGMMS(img, mask, bgdGMM, fgdGMM, compIdxs);
		learnGMMs(img, mask, bgdGMM, fgdGMM, compIdxs);
		getGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
		estimateSegmentation(graph, mask);
	}
	//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
}