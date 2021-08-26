// Contactless_Fingerprint.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <iostream>
// Function Declarations
void Control_Window_Func();
void SkinTone_ROI_Func();
void Binary_Skintone_Func();

using namespace std;
using namespace cv;

//Global Variables
VideoCapture Capture(0);
int Hue_Low = 0;
int Hue_High = 12;

int Sat_Low = 35;
int Sat_High = 255;

int Value_Low = 60;
int Value_High = 255;
Mat Image_hsv, Image_Skintone;
Mat Video_Frame;
vector<vector<Point> > Hand_contour;


int main()
{

	if (!Capture.isOpened())
	{
		cout << "Error! Opening Inbuilt Camera" << endl;
		return -1;
	}

	Control_Window_Func();

	SkinTone_ROI_Func();

	Capture.release();
	return 0;


}

void Control_Window_Func()
{

	//create a window called "Control"
	namedWindow("Control", WINDOW_KEEPRATIO);

  //Create trackbars in "Control" window
	//Hue (0 - 179)
	cvCreateTrackbar("Hue_Low", "Control", &Hue_Low, 179); 
	cvCreateTrackbar("Hue_High", "Control", &Hue_High, 179);
	//Saturation (0 - 255)
	cvCreateTrackbar("Sat_Low", "Control", &Sat_Low, 255); 
	cvCreateTrackbar("Sat_High", "Control", &Sat_High, 255);
	//Value (0 - 255)
	cvCreateTrackbar("Value_Low", "Control", &Value_Low, 255); 
	cvCreateTrackbar("Value_High", "Control", &Value_High, 255);

}

void SkinTone_ROI_Func()
{
	while (Capture.isOpened())
	{
		Binary_Skintone_Func();

		Point2f rect_points[4];
		
		for (size_t i = 0; i < Hand_contour.size(); i++)
		{
			double area = contourArea(Hand_contour[i]);
			if (area > 500) 
			{
				RotatedRect Finger_Box = minAreaRect(Hand_contour[i]);
				Finger_Box.points(rect_points);

				Point2f center = Finger_Box.center;
				double width = Finger_Box.size.width;
				double height = Finger_Box.size.height;


				//reducing the size of rectangle
				if (height > width)
				{ 
					Finger_Box.size.height = (float)(0.33) * Finger_Box.size.height;
					Finger_Box.center = (rect_points[1] + rect_points[2]) / 2 + (rect_points[0] - rect_points[1]) / 6;                  
				}
				else
				{
					Finger_Box.size.width = (float)(0.33) * Finger_Box.size.width;
					Finger_Box.center = (rect_points[2] + rect_points[3]) / 2 + (rect_points[0] - rect_points[3]) / 6;
				}
				Finger_Box.points(rect_points);
				Point2f center1 = Finger_Box.center;


				for (int j = 0; j < 4; j++)
				{
					line(Video_Frame, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 2, 8); // Drawing lines around the rectangle
				}

				drawContours( Video_Frame, Hand_contour, i, Scalar( 0, 255, 0 ), 1 ); // Draw the largest Hand_contour
			}
		}

		namedWindow("Region Of Interest", 1);
		imshow("Region Of Interest", Video_Frame);
		if (waitKey(30) >= 0) break;
	}
}

void Binary_Skintone_Func()
{
	Capture >> Video_Frame;


	cvtColor(Video_Frame, Image_hsv, CV_BGR2HSV);  //Color conversion

	inRange(Image_hsv, Scalar(Hue_Low, Sat_Low, Value_Low), Scalar(Hue_High, Sat_High, Value_High), Image_Skintone);//Skin tone segmentation

	imshow("Binary Image", Image_Skintone);


	//Image smoothing depends on blurSize
	int an = 5, blurSize = 7;
	//GaussianBlur(Image_Skintone, Image_Skintone, Size(5,5), 0,0);
	medianBlur(Image_Skintone, Image_Skintone, blurSize);
	//imshow("Image Smoothing", Image_Skintone);
	// Creating a structuring element Image_Kernel
	Mat Image_Kernel = getStructuringElement(MORPH_ELLIPSE, Size(an * 2 + 1, an * 2 + 1), Point(an, an));
	//Erosion API
	erode(Image_Skintone, Image_Skintone, Image_Kernel);
	//imshow("Erode Image", Image_Skintone);
	//Dilation API
	dilate(Image_Skintone, Image_Skintone, Image_Kernel);
	//imshow("Dilate Image", Image_Skintone);

	int a = Video_Frame.rows;

	Mat labelImage(Video_Frame.size(), CV_32S);
	for (int i = 0; i < 25; i++) {
		line(Image_Skintone, { 0, a }, { Video_Frame.cols, a }, Scalar(0, 0, 0), 30, 8);
		a = a - 30;
		//returns number of connected components
		int label = connectedComponents(Image_Skintone, labelImage, 8);
		//i.e if there are more than one connected components with white pixels it will break
		if (label > 2)
			break;
	}
	namedWindow("SkinTone_Sensing", 1);
	imshow("SkinTone_Sensing", Image_Skintone);


	//Finding the Hand_contour

	findContours(Image_Skintone, Hand_contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
}

