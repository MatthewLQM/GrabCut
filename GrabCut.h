#pragma once
#include <cv.h>
#include <iostream>
enum
{
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2  
};
enum {
	MUST_BGD = 0,
	MUST_FGD = 1,
	MAYBE_BGD = 2,
	MAYBE_FGD = 3
};

class GrabCut2D
{
public:
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
		int iterCount, int mode );  

	~GrabCut2D(void);
};




