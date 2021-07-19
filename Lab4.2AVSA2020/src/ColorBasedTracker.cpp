/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB4.1: ColorBasedTracker
 *	yracker.cpp
 *
 * 	Authors: Sergio Romero & Jan Sieradzki
 *	IPCV & I2ICSI - 2021
 */

#include "ColorBasedTracker.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace tracker;


#define MEASUREMENT_SIZE 2
#define CONTROL_PARAMS 0

/**
 *	Initialize the color-histogram tracker
 *
 * \frame initial frame of video, ground truth histogram gt_hist is take from this one
 *
 * \ground_truth ground truth rectangle of the first frame
 *
 * \bins amount of bins that will be used in histograms
 *
 * \cand  [cand_param x cand_param] is amount of candidates generated in every frame
 *		  (cand_param is side of the grid used to candidates generation,
 *		   thus the total candidate amount is cand_param x cand_param)
 *
 * \pix_stride the pixel distance between candidates rectangles generated in grid
 *
 * \channel_id the id of channel of interest for tracker
 *				 0 - gray
 *				 1 - H from HSV
 *				 2 - S from HSV
 *				 3 - B from BGR
 *				 4 - G from BGR
 *				 5 - R from BGR
 *
 * \normal tells, if histograms should be normalized
 *
 * \return void (it's a starter function).
 *
 */
ColorBasedTracker::ColorBasedTracker(Mat frame, Rect ground_truth, int bins, int cand, int pix_stride, int channel_id, bool normal)
{
	normalization = normal;
	bins_param = bins;
	cand_param = cand;
	p_stride = pix_stride;
	channel = channel_id;
	//extracts channel of interest from the frame
	convert_RGB_to_channel(frame);

	// set up values range for the histogram
	// if we are looking at h in hsv, set 180, else 256

	ranges[0] = 0;
	if (channel_id == 1){
		ranges[1] = 180;
	} else {
		ranges[1] = 256;
	}
	const float * range[] = {ranges};


	//calculating histogram
	gt_hist = calculate_histogram(ground_truth,range);

	//saving last prediction for future frame tracking
	last_prediction = ground_truth;
}

// destructor
ColorBasedTracker::~ColorBasedTracker(void)
{
}

/**
 *  Generate_candidates function is responsible for generation candidates.
 *  Candidates are generated in grid centered on previous frame rectangle prediction.
 *  If the grid size is even - e. g. 6x6 - the grid is slightly shifted to the left
 *  and down with respect to previous frame rectangle prediction (since such square does
 *  not have 'middle' rectangle.
 *
 *  The grid size depends from parameter cand_param:
 *  [cand_param x cand_param] is amount of candidates generated in every frame
 *	(cand_param is side of the grid used to candidates generation,
 *	thus the total candidate amount is cand_param x cand_param)
 *
 *	The function returns the vector of candidates (sized cand_param x cand_param),
 *	where there is always included prediction from previous frame.
 */
vector<Rect>  ColorBasedTracker::generate_candidates(){
	vector<Rect> candidates;
	// dimensions of previous frame result-rectangle
	float height = last_prediction.height;
	float width = last_prediction.width;
	int prev_x = last_prediction.x;
	int prev_y = last_prediction.y;

	int counter = cand_param/2;

	// creating the grid of candidates centered on the result rectangle obtained in previous frame
	for(int i=0;i< counter;i++){
		for(int j=0;j< counter;j++){
			candidates.push_back(Rect(prev_x+p_stride*(i+1),prev_y+p_stride*j,width,height));
			candidates.push_back(Rect(prev_x+p_stride*(i+1),prev_y-p_stride*(j+1),width,height));
			candidates.push_back(Rect(prev_x-p_stride*i,prev_y+p_stride*j,width,height));
			candidates.push_back(Rect(prev_x-p_stride*i,prev_y-p_stride*(j+1),width,height));
		}
	}

	// if the grid has even side, e. g. 6x6, add the column to the left,
	// and row to the up of already created grid
	if (cand_param%2!=0){
		for(int j=0;j< counter;j++){
			candidates.push_back(Rect(prev_x+p_stride*(j+1),prev_y+p_stride*counter,width,height));
			candidates.push_back(Rect(prev_x-p_stride*j,prev_y+p_stride*counter,width,height));
			candidates.push_back(Rect(prev_x-p_stride*counter,prev_y-p_stride*(j+1),width,height));
			candidates.push_back(Rect(prev_x-p_stride*counter,prev_y+p_stride*j,width,height));
		}
		candidates.push_back(Rect(prev_x-p_stride*counter,prev_y+p_stride*counter,width,height));
	}

	//Deleting candidates out of frame bounds
	int right_x_limit = actual_frame.cols - width;
	int down_y_limit = actual_frame.rows - height;
	for (auto iter = candidates.begin() ; iter != candidates.end(); ) {
	  if (iter->x<width || iter->x>right_x_limit ||
				(iter->y)<height || (iter->y)>down_y_limit) {
		  iter = candidates.erase(iter);
	  } else {
	    ++iter;
	  }
	}
	//return vector of candidates
	return candidates;
}

/**
 * Function find_best_candidate calculates histograms, scores them all with Bhattacharyya distance
 * and selects best candidate, taking rectangle that has minimal Bhattacharyya distance (to ground truth
 * histogram gt_hist obtained from first frame)
 *
 *  \candidates vector of candidates
 *  \return the candidate rectangle that will be object's final prediction for tracked frame
 */
Rect ColorBasedTracker::find_best_candidate(vector<Rect> candidates){
	vector<double> hist_comp_scores;
	Mat candidate_hist;

	// value's range parameter for histogram
	const float * range[] = {ranges};

	//iterating through all candidates
	for (auto it = begin (candidates); it != end (candidates); ++it) {
		// calculating histogram of candidate
		candidate_hist = calculate_histogram(*it,range);
		// computing Bhattacharyya distance
		hist_comp_scores.push_back(compareHist( gt_hist, candidate_hist, CV_COMP_BHATTACHARYYA ));
	}
	// finding index of candidate, which has the smallest distance from the ground through histogram gt_hist
	int minElementIndex = min_element(hist_comp_scores.begin(),hist_comp_scores.end()) - hist_comp_scores.begin();
	// returning the best candidate for actual frame tracking
	last_prediction = candidates[minElementIndex];
	return last_prediction;
}



/**
 * Function execute_tracking_step conducts tracker step for every frame. Firstly it extracts channel of interest from the frame,
 * later it generates candidates and finally scores them and chooses the best rectangle-candidate, which is returned as the
 * result of tracking
 */
Rect ColorBasedTracker::execute_tracking_step(Mat frame)
{
	//extracts channel of interest from the frame
	convert_RGB_to_channel(frame);
	//generates candidates
	vector<Rect> list_candidates = generate_candidates();
	//scores candidates and return the best one
	return find_best_candidate(list_candidates);
}

/**
 * Function calculates histograms for given candidate rectangle
 *
 * \rectangle the candidate for which histogram will be computed
 * \range the range of histogram values
 *
 * \return hist the functions returns the candidate's histogram
 */
Mat ColorBasedTracker::calculate_histogram(Rect rectangle, const float * range[])
{
	// matrix which will keep histogram
	Mat hist;
	Mat img_to_compute = actual_frame(rectangle);
	// calculating histogram of candidate
	calcHist( &img_to_compute, 1, 0, Mat(), hist, 1, &bins_param, range,  true, false);

	// normalizing histogram
	if (normalization){
		normalize( hist, hist, 0.01, 1, NORM_MINMAX, -1, Mat() );
	}
	return hist;
}

/**
 * Function convert_RGB_to_channel extracts appropriate channel from the frame, type of channel depends from parameter channel_id
 * Channel_id - channel mapping
 * 0 - gray
 * 1 - H from HSV
 * 2 - S from HSV
 * 3 - B from BGR
 * 4 - G from BGR
 * 5 - R from BGR
 */
void ColorBasedTracker::convert_RGB_to_channel(Mat frame)
{
	Mat split_frame[3];
	switch(channel) {
	   case 0  :
		   cvtColor(frame, actual_frame, cv::COLOR_BGR2GRAY);
		   break;
	   case 1  :
		   cvtColor(frame, actual_frame, cv::COLOR_BGR2HSV);
		   split(actual_frame, split_frame);
		   actual_frame = split_frame[0];
		   break;
	   case 2 :
		   cvtColor(frame, actual_frame, cv::COLOR_BGR2HSV);
		   split(actual_frame, split_frame);
		   actual_frame = split_frame[1];
	   	   break;
	   case 3:
		   split(frame, split_frame);
		   actual_frame = split_frame[0];
	   	   break;
	   case 4:
		   split(frame, split_frame);
		   actual_frame = split_frame[1];
	   	   break;
	   case 5:
		   split(frame, split_frame);
		   actual_frame = split_frame[2];
	   	   break;
	}
}

