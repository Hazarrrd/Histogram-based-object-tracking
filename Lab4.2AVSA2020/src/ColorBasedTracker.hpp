/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB4.1: ColorBasedTracker
 *	tracker.hpp
 *
 * 	Authors: Sergio Romero & Jan Sieradzki
 *	VPULab-UAM 2020
 */


#include <opencv2/opencv.hpp>
#ifndef ColorBasedTracker_HPP_INCLUDE
#define ColorBasedTracker_HPP_INCLUDE

using namespace cv;
using namespace std;

namespace tracker {

	//class
	class ColorBasedTracker{
	//Public functions
	public:
		//constructor function
		ColorBasedTracker(Mat frame, Rect ground_truth, int bins, int cand, int pix_stride, int channel_id, bool normal);

		//destructor function
		~ColorBasedTracker(void);

		//generating candidates
		vector<Rect>  generate_candidates(void);

		//scoring and choosing best candidate
		Rect find_best_candidate(vector<Rect> candidates);

		//executes step for every frame
		Rect execute_tracking_step(Mat frame);

		// extracrs channel of interest from frame
		void convert_RGB_to_channel(Mat frame);

		//calculate histogram for the candidate
		Mat calculate_histogram(Rect rectangle, const float * range[]);

		// ground truth histogram of the tracked object (taken from first frame)
		Mat gt_hist;
		// actual frame (with already extracted channel of interest)
		Mat actual_frame;
		// algortihm's prediction of object's position established in the previous frame
		Rect last_prediction;
		// [cand_param x cand_param] is amount of candidates generated in every frame
		// (cand_param is side of the grid used to candidates generation,
		// thus the total candidate amount is cand_param x cand_param)
		int cand_param;
		// the pixel distance between candidates rectangles generated in grid
		int p_stride;
		//the id of channel of interest for tracker
		// 0 - gray
		// 1 - H from HSV
		// 2 - S from HSV
		// 3 - B from BGR
		// 4 - G from BGR
		// 5 - R from BGR
		int channel;
		// amount of bins created in histogram
		int bins_param;

		//normalization tells, if histograms should be normalized
		bool normalization;

		// range of values - parameter for histogram
		float ranges[2];

	};
}

#endif
