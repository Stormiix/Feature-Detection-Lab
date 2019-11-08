#include <iostream>
#include <cstdio>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define DEBUG 0  // Debug mode shows the frames after each processing step. Set to 1 to enable it

int main(){
    
  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap("../video1.mp4");
    
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
   

  Mat  firstFrame;

  // Capture first frame
  cap >> firstFrame;

  // Crop rectangle of interest
  Rect roi;
  roi.x = 24;
  roi.y = 44;
  roi.width = 196-24;
  roi.height = 210-44;

  Mat croppedFirstFrame;

  /* Crop the original image to the defined ROI and convert to grayscale */
  cvtColor(firstFrame(roi), croppedFirstFrame, CV_BGR2GRAY);

  while(1){ 
    Mat frame, greyFrame;
    // Capture frame-by-frame
    cap >> frame;

    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    // Display the resulting frame
    if (DEBUG)
        imshow( "Frame", frame );
    cvtColor(frame, greyFrame, CV_BGR2GRAY);
    if (DEBUG)
        imshow( "Grey", greyFrame );
	
    int minHessian = 400;
	// Initiate the SURF detector
    Ptr<SURF> detector = SURF::create(minHessian);
    vector<KeyPoint> keypointsObj, keypoints;

    // Detect the keypoints using SURF Detector
    detector->detect(greyFrame, keypoints);
    detector->detect(croppedFirstFrame, keypointsObj);

    // Draw keypoints
    Mat frameKeypoints;
    drawKeypoints(greyFrame, keypoints, frameKeypoints);
    if (DEBUG)
        imshow( "Keypoints", frameKeypoints );

    // Calculate descriptors (feature vectors)
    Mat descriptorsObj, descriptors;
    detector->compute( greyFrame, keypoints, descriptors );
    detector->compute( croppedFirstFrame, keypointsObj, descriptorsObj );

    //  Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptorsObj, descriptors, matches);

    Mat frameMatches;

    drawMatches(croppedFirstFrame, keypointsObj, greyFrame, keypoints,
               matches, frameMatches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    if (DEBUG)
        //Show detected matches
        imshow( "Detected Matches", frameMatches );

    vector<Point2f> obj;
    vector<Point2f> scene;

    for (int i = 0;i < keypointsObj.size(); i++){
        obj.push_back(keypointsObj[matches[i].queryIdx].pt);
        scene.push_back(keypoints[matches[i].trainIdx].pt);
    }

    Mat H = findHomography(obj, scene, CV_RANSAC);

    vector<Point2f> objCorners(4);
    objCorners[0] = Point2f(0, 0);
    objCorners[1] = Point2f((float) croppedFirstFrame.cols, 0);
    objCorners[2] = Point2f((float) croppedFirstFrame.cols, (float) croppedFirstFrame.rows);
    objCorners[3] = Point2f(0, (float) croppedFirstFrame.rows);

    vector<Point2f> sceneCorners(4);
    perspectiveTransform(objCorners, sceneCorners, H);

    line(frameMatches, sceneCorners[0] + Point2f((float) croppedFirstFrame.cols, 0), sceneCorners[1] + Point2f((float) croppedFirstFrame.cols, 0), Scalar(0,255,0),4);
    line(frameMatches, sceneCorners[1] + Point2f((float) croppedFirstFrame.cols, 0), sceneCorners[2] + Point2f((float) croppedFirstFrame.cols, 0), Scalar(0,255,0),4);
    line(frameMatches, sceneCorners[2] + Point2f((float) croppedFirstFrame.cols, 0), sceneCorners[3] + Point2f((float) croppedFirstFrame.cols, 0), Scalar(0,255,0),4);
    line(frameMatches, sceneCorners[3] + Point2f((float) croppedFirstFrame.cols, 0), sceneCorners[0] + Point2f((float) croppedFirstFrame.cols, 0), Scalar(0,255,0),4);

    imshow( "Matches", frameMatches);

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;

  }
  
  // When everything done, release the video capture object
  cap.release();
 
  // Closes all the frames
  destroyAllWindows();
     
  return 0;
}
