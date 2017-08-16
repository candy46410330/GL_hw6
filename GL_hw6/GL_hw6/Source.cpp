/*  
 * Face Detection in Digital Photographs  
 *  
 * We leverage the AdaBoost learning-based face detection method with Haar-like features to detect faces in photos.  
 * However, OpenCV supplies Haar Cascade classifier for object (face) detection.
 * 
 * (C) Matt Swanson & TrekLee 
 * http://seeyababy.blogspot.com/  
 *  
*/ 

// Path setting for OpenCV Library
//#pragma comment(lib,"cv.lib")
//#pragma comment(lib,"cxcore.lib")
//#pragma comment(lib,"highgui.lib")

// Path setting for OpenCV Header files
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <stdio.h>
#include <iostream>

using namespace std;

// the minimum object size
int min_face_height    = 20;
int min_face_width    = 10;

//Face Detection
int runFaceDetection(IplImage* image_detect);

//void main()
//{    
//    IplImage* pImg = cvLoadImage( "lena.jpg", 1);    
//    runFaceDetection(pImg);
//    cvReleaseImage( &pImg ); 
//}

int runFaceDetection(IplImage* image_detect)
{ 
    // Load the pre-trained Haar classifier data.
    CvHaarClassifierCascade* classifier = (CvHaarClassifierCascade*)cvLoad("haarcascade_frontalface_alt.xml", 0, 0, 0);
 
    // Quit the application if the input source or the classifier data failed to load properly.
    if( !classifier)
    {
        cerr << "ERROR: Could not load classifier cascade." << endl;
        return -1;
    }

    // Create a CvMemStorage object for use by the face detection function.
    CvMemStorage* facesMemStorage = cvCreateMemStorage(0);

    IplImage* tempFrame = cvCreateImage(cvSize(image_detect->width, 
        image_detect->height), IPL_DEPTH_8U, image_detect->nChannels);

    // Copy the current frame into the temporary image.  Also, make 
    // sure the images have the same orientation.
    if(image_detect->origin == IPL_ORIGIN_TL)
    {
        cvCopy(image_detect, tempFrame, 0);
    }
    else
    {
        cvFlip(image_detect, tempFrame, 0);
    }

    /* Perform face detection on the temporary image, adding a rectangle around the detected face. */

    // "Resets" the memory but does not deallocate it.
    cvClearMemStorage(facesMemStorage);
 
    // Run the main object recognition function.  The arguments are: 
    // 1. the image to use
    // 2. the pre-trained Haar classifier cascade data
    // 3. memory storage for rectangles around recognized objects
    // 4. a scale factor "by which the search window is scaled between the 
    //    subsequent scans, for example, 1.1 means increasing window by 10%"
    // 5. the "minimum number (minus 1) of neighbor rectangles that makes up 
    //    an object. All the groups of a smaller number of rectangles than 
    //    min_neighbors-1 are rejected. If min_neighbors is 0, the function 
    //    does not any grouping at all and returns all the detected candidate 
    //    rectangles, which may be useful if the user wants to apply a 
    //    customized grouping procedure."
    // 6. flags which determine the mode of operation
    // 7. the minimum object size (if possible, increasing this will 
    //    really speed up the process)
    CvSeq* faces = cvHaarDetectObjects(tempFrame, classifier, facesMemStorage, 1.1, 
        2, CV_HAAR_DO_CANNY_PRUNING, cvSize(min_face_width, min_face_height));

    // If any faces were detected, draw rectangles around them.
    if (faces)
    {    
        for(int i = 0; i < faces->total; ++i)
        {
            // Setup two points that define the extremes of the rectangle, 
            // then draw it to the image..
            CvPoint point1, point2;
            CvRect* rectangle = (CvRect*)cvGetSeqElem(faces, i);
            
            point1.x = rectangle->x;
            point2.x = rectangle->x + rectangle->width;
            point1.y = rectangle->y;
            point2.y = rectangle->y + rectangle->height;

            cvRectangle(tempFrame, point1, point2, CV_RGB(255,0,0), 3, 8, 0);
        }
    }
    
    // Show the result in the window.
    cvNamedWindow("Face Detection Result", 1);
    cvShowImage("Face Detection Result", tempFrame);
    cvWaitKey(0);
    cvDestroyWindow("Face Detection Result");

    // Clean up allocated OpenCV objects.
    cvReleaseMemStorage(&facesMemStorage);
    cvReleaseImage(&tempFrame);
    cvReleaseHaarClassifierCascade( &classifier );

    return 0;
}

///////////////////////////////////////////////////////////
//#include<stdio.h>
//#include<math.h>
//#include<opencv\cv.h>
//#include<opencv\highgui.h>
//#include<opencv2\objdetect\objdetect.hpp>
//#include<opencv2\highgui\highgui.hpp>
//#include<opencv2\imgproc\imgproc.hpp>
//#include<vector>
//
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	CascadeClassifier face_cascade, eye_cascade;
//	if (!face_cascade.load("C:\opencv\build\share\OpenCV\haarcascades\haarcascade_frontalface_alt2.xml")) {
//		printf("Error loading cascade file for face");
//		return 1;
//	}
//	if (!eye_cascade.load("C:\opencv\build\share\OpenCV\haarcascades\haar\haarcascade_eye.xml")) {
//		printf("Error loading cascade file for eye");
//		return 1;
//	}
//	VideoCapture capture(0); //-1, 0, 1 device id
//	if (!capture.isOpened())
//	{
//		printf("error to initialize camera");
//		return 1;
//	}
//	Mat cap_img, gray_img;
//	vector<Rect> faces, eyes;
//	while (1)
//	{
//		capture >> cap_img;
//		waitKey(10);
//		cvtColor(cap_img, gray_img, CV_BGR2GRAY);
//		cv::equalizeHist(gray_img, gray_img);
//		face_cascade.detectMultiScale(gray_img, faces, 1.1, 10, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, cvSize(0, 0), cvSize(300, 300));
//		for (int i = 0; i < faces.size(); i++)
//		{
//			Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
//			Point pt2(faces[i].x, faces[i].y);
//			Mat faceROI = gray_img(faces[i]);
//			eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
//			for (size_t j = 0; j< eyes.size(); j++)
//			{
//				//Point center(faces[i].x+eyes[j].x+eyes[j].width*0.5, faces[i].y+eyes[j].y+eyes[j].height*0.5);
//				Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
//				int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
//				circle(cap_img, center, radius, Scalar(255, 0, 0), 2, 8, 0);
//			}
//			rectangle(cap_img, pt1, pt2, cvScalar(0, 255, 0), 2, 8, 0);
//		}
//		imshow("Result", cap_img);
//		waitKey(3);
//		char c = waitKey(3);
//		if (c == 27)
//			break;
//	}
//	return 0;
//}

////////////////////////////////////////////////////////////
//#include <cv.h>
//#include <cxcore.h> 
//#include <highgui.h> 
//#include <iostream>
//using namespace std;
//// the minimum object size
//int min_face_height = 50;
//int min_face_width = 50;
//int main(int argc, char ** argv){
//	string image_name = "lena.bmp";
//	// Load image
//	IplImage* image_detect = cvLoadImage(image_name.c_str(), 1);
//	string cascade_name = "C:\opencv\build\share\OpenCV\haarcascades/haarcascade_frontalface_alt.xml";
//
//	// Load cascade
//	CvHaarClassifierCascade* classifier = (CvHaarClassifierCascade*)cvLoad(cascade_name.c_str(), 0, 0, 0);
//	if (!classifier){
//		cerr << "ERROR: Could not load classifier cascade." << endl;
//		system("pause");
//		return -1;
//	}
//	CvMemStorage* facesMemStorage = cvCreateMemStorage(0);
//	IplImage* tempFrame = cvCreateImage(cvSize(image_detect->width, image_detect->height), IPL_DEPTH_8U, image_detect->nChannels);
//	if (image_detect->origin == IPL_ORIGIN_TL){
//		cvCopy(image_detect, tempFrame, 0);
//	}
//	else{
//		cvFlip(image_detect, tempFrame, 0);
//	}
//	cvClearMemStorage(facesMemStorage);
//	CvSeq* faces = cvHaarDetectObjects(tempFrame, classifier, facesMemStorage, 1.1, 3
//		, CV_HAAR_DO_CANNY_PRUNING, cvSize(min_face_width, min_face_height));
//	if (faces){
//		for (int i = 0; i<faces->total; ++i){
//			// Setup two points that define the extremes of the rectangle,
//			// then draw it to the image
//			CvPoint point1, point2;
//			CvRect* rectangle = (CvRect*)cvGetSeqElem(faces, i);
//			point1.x = rectangle->x;
//			point2.x = rectangle->x + rectangle->width;
//			point1.y = rectangle->y;
//			point2.y = rectangle->y + rectangle->height;
//			cvRectangle(tempFrame, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
//		}
//	}
//	// Save the image to a file
//	cvSaveImage("02.bmp", tempFrame);
//	// Show the result in the window
//	cvNamedWindow("Face Detection Result", 1);
//	cvShowImage("Face Detection Result", tempFrame);
//	cvWaitKey(0);
//	cvDestroyWindow("Face Detection Result");
//	// Clean up allocated OpenCV objects
//	cvReleaseMemStorage(&facesMemStorage);
//	cvReleaseImage(&tempFrame);
//	cvReleaseHaarClassifierCascade(&classifier);
//	cvReleaseImage(&image_detect);
//	system("pause");
//	return EXIT_SUCCESS;
//}
