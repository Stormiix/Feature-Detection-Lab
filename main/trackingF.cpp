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

#define DEBUG 0

int main(){
    
  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap("../video1.mp4");
    
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
   

  Mat  previousFrame;

  // Capture first frame and set it as the first "previousFrame"
  cap >> previousFrame;

  // Crop rectangle of interest
  Rect roi(24,44,196-24,210-44);

  /* Crop the previousFrame to the defined ROI and convert to grayscale*/
  Mat croppedPreviousFrame;
  cvtColor(previousFrame(roi), croppedPreviousFrame, CV_BGR2GRAY);

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
	
	// Convert the current frame to grayscale
	cvtColor(frame, greyFrame, CV_BGR2GRAY);
    
	if (DEBUG)
        imshow( "Grey", greyFrame );

    // Detect the keypoints using SURF Detector
    int minHessian = 400;

    Ptr<SURF> detector = SURF::create(minHessian);
    vector<KeyPoint> keypointsObj, keypoints;

    detector->detect(greyFrame, keypoints);
    detector->detect(croppedPreviousFrame, keypointsObj);

    // Draw keypoints
    Mat frameKeypoints;
    drawKeypoints(greyFrame, keypoints, frameKeypoints);
    if (DEBUG)
        imshow( "Keypoints", frameKeypoints );

    // Calculate descriptors (feature vectors)
    Mat descriptorsObj, descriptors;
    detector->compute( greyFrame, keypoints, descriptors );
    detector->compute( croppedPreviousFrame, keypointsObj, descriptorsObj );

    //  Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match( descriptorsObj, descriptors, matches);

    Mat frameMatches;

    drawMatches(croppedPreviousFrame, keypointsObj, greyFrame, keypoints,
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
    objCorners[1] = Point2f((float) croppedPreviousFrame.cols, 0);
    objCorners[2] = Point2f((float) croppedPreviousFrame.cols, (float) croppedPreviousFrame.rows);
    objCorners[3] = Point2f(0, (float) croppedPreviousFrame.rows);

    vector<Point2f> sceneCorners(4);
    perspectiveTransform(objCorners, sceneCorners, H);

    line(frameMatches, sceneCorners[0] + Point2f((float) croppedPreviousFrame.cols, 0), sceneCorners[1] + Point2f((float) croppedPreviousFrame.cols, 0), Scalar(0,255,0),4);
    line(frameMatches, sceneCorners[1] + Point2f((float) croppedPreviousFrame.cols, 0), sceneCorners[2] + Point2f((float) croppedPreviousFrame.cols, 0), Scalar(0,255,0),4);
    line(frameMatches, sceneCorners[2] + Point2f((float) croppedPreviousFrame.cols, 0), sceneCorners[3] + Point2f((float) croppedPreviousFrame.cols, 0), Scalar(0,255,0),4);
    line(frameMatches, sceneCorners[3] + Point2f((float) croppedPreviousFrame.cols, 0), sceneCorners[0] + Point2f((float) croppedPreviousFrame.cols, 0), Scalar(0,255,0),4);

    imshow( "Matches", frameMatches);

	// Reverse the perspectiveTransform applied to the our matched frame
    Mat rotated;
    warpPerspective(greyFrame, rotated, findHomography(sceneCorners, objCorners, CV_RANSAC), rotated.size(), INTER_LINEAR, BORDER_CONSTANT);

	// Crop and use the unwarped image as the previousFrame for the next iteration
    Rect roi(0,0,196-24,210-44);
    croppedPreviousFrame = rotated(roi);

    if (DEBUG)
        imshow( "Rotated previous frame", croppedPreviousFrame);

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
