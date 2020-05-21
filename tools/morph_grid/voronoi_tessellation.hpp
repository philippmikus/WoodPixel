#ifndef VORONOI_TESSELLATION_HPP_
#define VORONOI_TESSELLATION_HPP_

#include "future_base.hpp"

using namespace cv;

class VoronoiTessellation {
public:
	VoronoiTessellation(cv::Mat image, int seed_count) :
	m_image(image),
	m_width(image.cols),
	m_height(image.rows),
	m_seed_count(seed_count)
	{}

	void poisson_disk_sampling(int iterations);
	void voronoi_tiles(std::vector<Point2f> seeds);
	std::vector<Point2f> lloyds_relaxation();
private:
	cv::Mat m_image;
	const int m_width;
	const int m_height;
	const int m_seed_count;

	std::vector<std::vector<Point2f>> facets;
	std::vector<Point2f> centers;
};

#endif