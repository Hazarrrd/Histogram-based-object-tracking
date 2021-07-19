/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template (sample program)
 *	provided for the assignment LAB 4 "GradientBasedTracker"
 *
 *	This code has been tested using:
 *	- Operative System: Ubuntu 18.04
 *	- OpenCV version: 3.4.4
 *	- Eclipse version: 2019-12
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
//includes
#include <stdio.h> 								//Standard I/O library
#include <numeric>								//For std::accumulate function
#include <string> 								//For std::to_string function
#include <opencv2/opencv.hpp>					//opencv libraries

#include "GradientBasedTracker.hpp"
#include "utils.hpp" 							//for functions readGroundTruthFile & estimateTrackingPerformance
#include "ShowManyImages.hpp"

//namespaces
using namespace cv;
using namespace std;
using namespace tracker;

//Macros declaration

//BINS_NUMBER amount of bins that will be used in histograms
#define BINS_NUMBER 9
//[CANDIDATE_GRID_SIDE x CANDIDATE_GRID_SIDE] is amount of candidates generated in every frame
//(CANDIDATE_GRID_SIDE is side of the grid used to candidates generation,
//thus the total candidate amount is CANDIDATE_GRID_SIDE x CANDIDATE_GRID_SIDE)
#define CANDIDATE_GRID_SIDE 10
//GRID_PIXEL_STRIDE the pixel distance between candidates rectangles generated in grid
#define GRID_PIXEL_STRIDE 2
//CHANNEL_TYPE is the id of channel of interest for tracker
//				 0 - gray
//				 1 - H from HSV
//				 2 - S from HSV
//				 3 - B from BGR
//				 4 - G from BGR
//				 5 - R from BGR
#define CHANNEL_TYPE 0

//NORMALIZATION tells, if histograms should be normalized
//ACCORDING TO ALGORITHM IT SHOULD BE ALWAYS FALSE - DONT CHANGE IT!
#define NORMALIZATION_GRAD false

//main function
int main(int argc, char ** argv)
{
	cout<< "Lab4_3_gradient_tracker" << endl <<
			" Params: chan" << CHANNEL_TYPE << " cands " << CANDIDATE_GRID_SIDE << " stride " << GRID_PIXEL_STRIDE << " bins " << BINS_NUMBER  <<
			" normal " << NORMALIZATION_GRAD << endl;;
	//PLEASE CHANGE 'dataset_path' & 'output_path' ACCORDING TO YOUR PROJECT
	std::string dataset_path = "/home/janek/avsa/AVSA2020datasets/AVSA_lab4_datasets/datasets/";
	std::string output_path = "/home/janek/avsa/AVSA2020results/outvideos/";	//location to save output videos

	// dataset paths
	//std::string sequences[] = {"bolt1",										//test data for lab4.1, 4.3 & 4.5
	//						   "sphere","car1",								//test data for lab4.2
	//						   "ball2","basketball",						//test data for lab4.4
	//						   "bag","ball","road",};						//test data for lab4.6
	std::string sequences[] = {"bolt1"};
	std::string image_path = "%08d.jpg"; 									//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 						//file for ground truth data. DO NOT CHANGE
	int NumSeq = sizeof(sequences)/sizeof(sequences[0]);					//number of sequences

	//Loop for all sequence of each category
	for (int s=0; s<NumSeq; s++ )
	{
		Mat frame;										//current Frame
		int frame_idx=0;								//index of current Frame
		std::vector<Rect> list_bbox_est, list_bbox_gt;	//estimated & groundtruth bounding boxes
		std::vector<double> procTimes;					//vector to accumulate processing times

		std::string inputvideo = dataset_path + "/" + sequences[s] + "/img/" + image_path; //path of videofile. DO NOT CHANGE
		std::string inputGroundtruth = dataset_path + "/" + sequences[s] + "/" + groundtruth_file;//path of groundtruth file. DO NOT CHANGE

		cout << inputvideo << endl;
		cout << inputGroundtruth << endl;

		if (argc<2){
			cout << "No arguments passed, default 'code' mode" << endl;
			cout << "If you want, you can pass one argument:" << endl <<
					"path to the video sequence" << endl <<
					"example: /home/janek/avsa/AVSA2020datasets/AVSA_lab4_datasets/datasets//bolt1" << endl;

		} else if(argc==2){
			cout << "OK, we are going to use video " << argv[1] << endl;
			inputvideo = string(argv[1]) + "/img/" + image_path;
			inputGroundtruth = string(argv[1]) + "/" + groundtruth_file;
			s = NumSeq;
		} else if(argc>2){
			cout << "To many arguments, pass only path to one video" << endl <<
					"example: /home/janek/avsa/AVSA2020datasets/AVSA_lab4_datasets/datasets//bolt1" << endl;
			cout << "default 'code' mode" << endl;
		}


		VideoCapture cap(inputvideo);	// reader to grab frames from videofile

		//check if videofile exists
		if (!cap.isOpened())
			throw std::runtime_error("Could not open video file " + inputvideo); //error if not possible to read videofile

		// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
		cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));//cv::Size frame_size(700,460);

		VideoWriter outputvideo(output_path+"outvid_" + sequences[s]+".avi",CV_FOURCC('X','V','I','D'),10, frame_size);	//xvid compression (cannot be changed in OpenCV)

		//Read ground truth file and store bounding boxes
		list_bbox_gt = readGroundTruthFile(inputGroundtruth); //read groundtruth bounding boxes

		//main loop for the sequence
		std::cout << "Displaying sequence at " << inputvideo << std::endl;
		std::cout << "  with groundtruth at " << inputGroundtruth << std::endl;

		cap >> frame;
		Mat frame_for_crop;
		Mat frame_for_candidates;
		vector<Rect>  list_candidates;

		//params for drawing histograms
		int hist_w = 250, hist_h = 250;
		int bin_w = cvRound( (double) hist_w/BINS_NUMBER );
		// initialization of tracking class,
		GradientBasedTracker tracker(frame,list_bbox_gt[0],BINS_NUMBER,CANDIDATE_GRID_SIDE,GRID_PIXEL_STRIDE,CHANNEL_TYPE, NORMALIZATION_GRAD);
		for (;;) {
			//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)

			if (!frame.data)
				break;
			frame.copyTo(frame);
			frame.copyTo(frame_for_crop);
			frame.copyTo(frame_for_candidates);

			//Time measurement
			double t = (double)getTickCount();
			frame_idx=cap.get(cv::CAP_PROP_POS_FRAMES);			//get the current frame

			////////////////////////////////////////////////////////////////////////////////////////////
			//DO TRACKING
			//Change the following line with your own code

			//Conducting the tracking step for video's frame - not in usage because we want to get list_candidates for experiment visualisation
			//list_bbox_est.push_back(tracker.execute_tracking_step(frame));

			//extracts channel of interest from the frame
			tracker.convert_RGB_to_channel(frame);
			//generates candidates
			list_candidates = tracker.generate_candidates();
			//scores candidates and return the best one
			list_bbox_est.push_back(tracker.find_best_candidate(list_candidates));

			////////////////////////////////////////////////////////////////////////////////////////////

			//Time measurement
			procTimes.push_back(((double)getTickCount() - t)*1000. / cv::getTickFrequency());
			//std::cout << " processing time=" << procTimes[procTimes.size()-1] << " ms" << std::endl;

			// plot frame number & groundtruth bounding box for each frame
			putText(frame, std::to_string(frame_idx), cv::Point(10,15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
			rectangle(frame, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));		//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));	//draw bounding box (estimation)

			// Plot candidates grid on the frame
			//vector_candidates = tracker.generate_candidates();
			for (auto it = begin (list_candidates); it != end (list_candidates); ++it){
				rectangle(frame_for_candidates, *it, Scalar(255, 0, 0));
			}
			rectangle(frame_for_candidates, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));
			rectangle(frame_for_candidates, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));


			//Prepering histogram visualisation
			Mat histImage( hist_w, hist_h, CV_8UC3, Scalar( 0,0,0) );
			Mat hist_est = tracker.calculate_HOG(list_bbox_est[frame_idx-1]);
			Mat hist_gt = tracker.calculate_HOG(list_bbox_gt[frame_idx-1]);
			Mat hist_template;
			normalize(hist_est, hist_est, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
			normalize(hist_gt, hist_gt, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
			normalize(tracker.gt_hist, hist_template, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

			putText(histImage, "[HOG] Frame 0", Point(10,20),FONT_HERSHEY_COMPLEX_SMALL, 0.75,Scalar(255,0,0));
			putText(histImage, "actual frame " + std::to_string(frame_idx) + " GT", Point(10,40),FONT_HERSHEY_COMPLEX_SMALL, 0.75,Scalar(0,255,0));
			putText(histImage, "prediction " + std::to_string(frame_idx), Point(10,60),FONT_HERSHEY_COMPLEX_SMALL, 0.75,Scalar(0,0,255));
			for( int i = 1; i < hist_gt.rows; i++ )
			{
				line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist_gt.at<float>(i-1)) ),
					  Point( bin_w*(i), hist_h - cvRound(hist_gt.at<float>(i)) ),
					  Scalar( 0, 255, 0), 2, 8, 0  );
				line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist_est.at<float>(i-1)) ),
					  Point( bin_w*(i), hist_h - cvRound(hist_est.at<float>(i)) ),
					  Scalar( 0, 0, 255), 2, 8, 0  );
				line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist_template.at<float>(i-1)) ),
									  Point( bin_w*(i), hist_h - cvRound(hist_template.at<float>(i)) ),
									  Scalar( 255, 0, 0), 2, 8, 0  );
			}

			ShowManyImages("Lab4_3_gradient_tracker-TRACKING|PREDICTION|CANDIDATES GRID|HISTOGRAMS", 4, frame,frame_for_crop(list_bbox_est[frame_idx-1]), frame_for_candidates,
					histImage);
			outputvideo.write(frame);//save frame to output video

			cap >> frame;
			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
		}

		//comparison groundtruth & estimation
		vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);

		//print stats about processing time and tracking performance
		std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;

		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
	printf("Finished program.");
	return 0;
}
