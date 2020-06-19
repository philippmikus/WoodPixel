#ifndef VORONOI_TESSELLATION_HPP_
#define VORONOI_TESSELLATION_HPP_

#include "future_base.hpp"
#include "data_grid.hpp"
#include "generate_patches.hpp"

using namespace cv;
using namespace std;

class VoronoiTessellation {
public:
	VoronoiTessellation(cv::Mat image, int seed_count) :
	m_image(image),
	m_width(image.cols),
	m_height(image.rows),
	m_seed_count(seed_count)
	{}
	void clip_facets();
	vector<Point2f> poisson_disk_sampling(int iterations);
	void voronoi_tiles(vector<Point2f> seeds);
	vector<Point2f> lloyds_relaxation();
	vector<vector<Point2f>> get_facets();

	DataGrid<unsigned char> compute_grid();
	vector<PatchRegion> compute_patches();
	
private:
	cv::Mat m_image;
	const int m_width;
	const int m_height;
	const int m_seed_count;

	vector<vector<Point2f>> facets;
	vector<Point2f> centers;

	DataGrid<unsigned char> m_grid;
};

#endif