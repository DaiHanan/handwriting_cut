#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
using namespace cv;
using namespace std;

string path = "D:/01_files/05_upward/02_school/02_handwriting/02_pics/page01/";
int cols;//页面列数
int rows;//页面行数
int wordCount = 0;//当前字数

vector<vector<int> >* _src;
vector<vector<int> >* _dst;

/*
* @brief 对输入图像进行细化,骨骼化
* @param src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
* @param dst为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
*/
void thinImage8(Mat& src, Mat& dst)
{
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	vector<uchar*> mFlag; //用于标记需要删除的点    
	while (true)
	{
		//步骤一   
		for (int i = 0; i < height; ++i)
		{
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//获得九个点对象，注意边界问题
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)//条件1判断 =0为孤立点 =1为端点 =7为内部点
				{
					//条件2计算  ap=0内部点或者孤立点 ；ap>=2为断开点或者内部点 不能删除
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;
					//条件2、3、4判断
					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)//删除右方或者下方的点
					{
						//标记    
						mFlag.push_back(p + j);
					}
				}
			}
		}
		//将标记的点删除    
		for (vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}
		//直到没有点满足，算法结束    
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空    
		}

		//步骤二，根据情况该步骤可以和步骤一封装在一起成为一个函数
		for (int i = 0; i < height; ++i)
		{
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记    
				//  p9 p2 p3    
				//  p8 p1 p4    
				//  p7 p6 p5    
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;
					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0) //删除的点是左方或者上方的点
					{
						//标记    
						mFlag.push_back(p + j);
					}
				}
			}
		}
		//将标记的点删除    
		for (vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}
		//直到没有点满足，算法结束    
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空    
		}
	}
}
void thinImage4(Mat& src, Mat& dst)
{
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	vector<uchar*> mFlag; //用于标记需要删除的点    
	while (true)
	{
		//  p9 p2 p3
		//  p8 p1 p4    
		//  p7 p6 p5    
	//步骤一   
		for (int i = 0; i < height; ++i)
		{
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//获得九个点对象，注意边界问题
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p81 = (j == 0) ? 0 : *(p + j - 2); //左射线
				uchar p82 = (j == 0) ? 0 : *(p + j - 3);
				uchar p41 = (j == width - 1) ? 0 : *(p + j + 2);//右射线
				uchar p21 = (i == 0) ? 0 : *(p - dst.step + j - 1);//上射线
				uchar p22 = (i == 0) ? 0 : *(p - dst.step + j - 2);
				uchar p61 = (i == height - 1) ? 0 : *(p + dst.step + j + 1);//下射线


				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)//条件1判断 =0为孤立点 =1为端点 =7为内部点
				{
					//条件2计算 邻域四点是否都在
					int ap = 0;
					if (p2 == 0) ++ap;
					if (p4 == 1) ++ap;
					if (p6 == 1) ++ap;
					if (p8 == 1) ++ap;


					if (ap != 4)//判断和是否为4
					{
						if (((p8 == 1 && p81 == 1 && p82 == 1) || (p4 == 1 && p41 == 1)) && ((p2 == 1 && p21 == 1 && p22 == 1) || (p6 == 1 && p61 == 1)))  //水平左点射线>2，水平右点射线>=2，同理上和下，可得此点不是关键点，可以删除
						{
							if (ap < 2)
							{
								//标记    
								mFlag.push_back(p + j);
							}

						}

					}
				}
			}
		}
		//将标记的点删除    
		for (vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}
		//直到没有点满足，算法结束    
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空    
		}

		//步骤二
		for (int i = 0; i < height; ++i)
		{
			uchar* p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记    
				//  p9 p2 p3    
				//  p8 p1 p4    
				//  p7 p6 p5    
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p81 = (j == 0) ? 0 : *(p + j - 2); //左射线
				uchar p82 = (j == 0) ? 0 : *(p + j - 3);
				uchar p41 = (j == width - 1) ? 0 : *(p + j + 2);//右射线
				uchar p21 = (i == 0) ? 0 : *(p - dst.step + j - 1);//上射线
				uchar p22 = (i == 0) ? 0 : *(p - dst.step + j - 2);
				uchar p61 = (i == height - 1) ? 0 : *(p + dst.step + j + 1);//下射线


				if ((p1 = 1 && p4 == 1 && p6 == 1) || (p2 == 1 && p1 == 1 && p8 == 1) || (p1 = 1 && p4 == 1 && p2 == 1) || (p1 = 1 && p8 == 1 && p6 == 1))
				{
					int ap = 0;
					if (p2 == 0) ++ap;
					if (p4 == 1) ++ap;
					if (p6 == 1) ++ap;
					if (p8 == 1) ++ap;

					if (((p8 == 1 && p81 == 1 && p82 == 1) || (p4 == 1 && p41 == 1)) && ((p2 == 1 && p21 == 1 && p22 == 1) || (p6 == 1 && p61 == 1)))  //水平左点射线>2，水平右点射线>=2，同理上和下，可得此点不是关键点，可以删除
					{
						if (ap < 2)
						{
							//标记    
							mFlag.push_back(p + j);
						}

					}

				}
			}
		}
		//将标记的点删除    
		for (vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}
		//直到没有点满足，算法结束    
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空    
		}
	}
}


//切割字体提取
void getWordByCut(int fromRow, int toRow, int fromCol, int toCol) {
	const vector<vector<int> >& src = *_src, & dst = *_dst;
	wordCount++;//更新字数
	//创建三通道图
	int wordRows = toRow - fromRow + 1, wordCols = toCol - fromCol + 1;
	cv::Mat srcImage(wordRows, wordCols, CV_8UC3);
	cv::Mat dstImage(wordRows, wordCols, CV_8UC3);
	//设置像素值
	for (int i = 0; i < wordRows; i++)
	{
		for (int j = 0; j < wordCols; j++)
		{
			int valI = i + fromRow, valJ = j + fromCol;
			//二值化
			if (src[valI][valJ] == 1) {//黑色
				srcImage.at<cv::Vec3b>(i, j) =
					cv::Vec3b(0, 0, 0);
			}
			else {//背景（白色）
				srcImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
			//细化
			if (dst[valI][valJ] == 1) {//黑色
				dstImage.at<cv::Vec3b>(i, j) =
					cv::Vec3b(0, 0, 0);
			}
			else {//背景（白色）
				dstImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
		}
	}
	//保存
	//cv::imshow("原图", srcImage);
	char ss[10];
	sprintf_s(ss, "%03d", wordCount);
	cv::imwrite(path + "words/" + ss + "(" + to_string(fromRow) + "," + to_string(fromCol) + ")_s.bmp", srcImage);
	cv::imwrite(path + "words/" + ss + "(" + to_string(fromRow) + "," + to_string(fromCol) + ").bmp", dstImage);
}

void shrinkRange(int fromRow, int toRow, int fromCol, int toCol) {
	const vector<vector<int> >& val = *_src;
	bool finish = false;
	//左右
	while (fromCol <= toCol && finish == false) {
		int i = fromRow;
		for (; i <= toRow; i++) {
			if (val[i][fromCol] == 1) break;//左边没有遇到黑色
		}
		if (i > toRow) fromCol++;//删除该列
		else finish = true;

		i = fromRow;
		for (; i <= toRow; i++) {
			if (val[i][toCol] == 1) break;//右边没有遇到黑色
		}
		if (i > toRow) {//删除该列
			toCol--;
			finish = false;
		}
	}
	if (fromCol >= toCol) return;//没有字体
	//上下
	finish = false;
	while (fromRow <= toRow && finish == false) {
		int i = fromCol;
		for (; i <= toCol; i++) {
			if (val[fromRow][i] == 1) break;//上边没有遇到黑色
		}
		if (i > toCol) fromRow++;//删除该行
		else finish = true;

		i = fromCol;
		for (; i <= toCol; i++) {
			if (val[toRow][i] == 1) break;//下边没有遇到黑色
		}
		if (i > toCol) {//删除该行
			toRow--;
			finish = false;
		}
	}
	if (fromRow >= toRow) return;//没有字体
	//存在字体则切割
	getWordByCut(fromRow, toRow, fromCol, toCol);
}

//切割列数
void cutByCol(int fromRow, int toRow) {
	const vector<vector<int> >& val = *_src;
	int fromCol = 0;
	bool hasBlack = false;//是否已经经过非纯白列
	int blackCol;//首次经过非白行的列下标
	for (int j = 1; j < cols; j++) {
		int i = fromRow;
		for (; i <= toRow; i++) {
			if (val[i][j] == 1) {//遇到黑色，标记并换列
				if (!hasBlack)
					blackCol = j;
				hasBlack = true;
				break;
			}
		}
		//如果没遇到黑色且存在黑色标记，说明可以进行行切割
		if (i > toRow && hasBlack && j - blackCol > 80) {
			shrinkRange(fromRow, toRow, fromCol, j);
			//标志重置
			fromCol = j + 1;
			hasBlack = false;
			j++;
		}
	}
}

//切割行数
void cutByRow() {
	const vector<vector<int> >& val = *_src;
	int fromRow = 0;
	bool hasBlack = false;//是否已经经过非纯白行
	int blackRow;//首次经过非白行的行下标
	for (int i = 1; i < rows; i++) {
		int j = 0;
		for (; j < cols; j++) {
			if (val[i][j] == 1) {//遇到黑色，标记并换行
				if(!hasBlack) 
					blackRow = i;
				hasBlack = true;
				break;
			}
		}
		//如果没遇到黑色且存在黑色标记，说明可以进行行切割
		if (j == cols && hasBlack && i - blackRow > 80) {
			//按列切割
			cutByCol(fromRow, i);
			//标志重置
			fromRow = i + 1;
			hasBlack = false;
			i++;
		}
	}
}

void cutWords(const Mat& src, const Mat& dst) {
	//操作矩阵 
	_src = new vector<vector<int> >(rows, vector<int>(cols, 0));
	_dst = new vector<vector<int> >(rows, vector<int>(cols, 0));
	vector<vector<int> > & srcVal = *_src, & dstVal = *_dst;
	//初始化像素值 0-白色 1-黑色
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			srcVal[i][j] = src.at<uchar>(i, j) == 255 ? 0 : 1;
			dstVal[i][j] = dst.at<uchar>(i, j) == 255 ? 0 : 1;
		}
	}
	//首先按行切割
	cutByRow();
}

void main()
{
	//初始化
	Mat src;
	src = imread(path + "page1.jpg", CV_8UC1);//00000071
	rows = src.rows;
	cols = src.cols;

	//骨架化
	//namedWindow("原图", 1);
	//imshow("原图", src);
	GaussianBlur(src, src, Size(7, 7), 0, CV_8UC1);//高斯滤波
	//imshow("二值化图像", src);

	threshold(src, src, 140, 1, cv::THRESH_BINARY_INV);//二值化，前景为1，背景为0
	Mat dst;
	thinImage8(src, dst);//图像细化
	thinImage4(dst, dst);//图像细化

	src = src * 255;
	Mat src1 = 255 - src;
	//imshow("二值化图像", src1);
	imwrite(path + "001_s(" + to_string(rows) + "," + to_string(cols) + ").bmp", src1);
	dst = dst * 255;
	Mat dst1 = 255 - dst;
	//imshow("细化图像", dst1);
	imwrite(path + "001(" + to_string(rows) + "," + to_string(cols) + ").bmp", dst1);
	
	//切割字体(二值化和细化)
	cutWords(src1, dst1);

	waitKey(0);
}
