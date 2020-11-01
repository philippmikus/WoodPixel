/*
"Intrinsic Manifold SLIC: A Simple and Efficient Method for Computing
Content-Sensitive Superpixels"
Yong-Jin Liu, Cheng-Chi Yu, Min-Jing Yu, Ying He,
IEEE Transactions on Pattern Analysis and Machine Intelligence,
March 2017, Issue 99.
*/

#ifndef VORONOI_TESSELLATION_HPP_
#define VORONOI_TESSELLATION_HPP_

#include "future_base.hpp"
#include "data_grid.hpp"
#include "generate_patches.hpp"

using namespace cv;
using namespace std;

struct Pixel
{
	float u, v, l, a, b;

	Pixel(float u = 0, float v = 0, float l = 0, float a = 0, float b = 0)
		: u(u), v(v), l(l), a(a), b(b)
	{}

	Pixel operator-(const Pixel m) const
	{
		return Pixel(u - m.u, v - m.v, l - m.l, a - m.a, b - m.b);
	}
};

struct img_pixel_data {
	int label;
	float distance;
	float area;
};

typedef pair<float, int> label;

class Graph
{
	int A;
	int V;
	vector<list<pair<int, float>>> adj;

	int m_width;
	int m_height;

public:
	Graph(int width, int height);
	void add_edge(int u, int v, float w);

	/**
	* Find shortest path using the Threshold algorithm
	*
	* @param s Index of starting node
	* @returns vector containing the shortest distances to all nodes in the graph
	*/
	vector<float> shortest_path(int s, int size, bool check_labels, vector<vector<img_pixel_data>> pixels);

	float calc_threshold(queue<label>& q2, float last_threshold, float last_avg, int i);
	float repartition(queue<label>& q1, queue<label>& q2, float threshold);
};

class VoronoiTessellation {
public:
	VoronoiTessellation(cv::Mat image, int seed_count) :
		m_image(image),
		m_width(image.cols),
		m_height(image.rows),
		m_seed_count(seed_count)
	{}

	/**
	 * Start the segmentation algorithm
	 *
	 * @param iterations Number of iterations
	 * @param superpixles Uses IMSLIC if true, and CVT if false
	 * @returns positions of seeds
	 */
	vector<Point2f> start(int iterations, bool superpixels);

	/************************************************
	*  CVT
	***********************************************/

	/**
	 * Clip facets to image boundary using Sutherland-Hodgman algorithm
	 */
	void clip_facets();

	/**
	 * Compute voronoi-facets
	 *
	 * @param seeds Vector containing the seed points
	 */
	void voronoi_tiles(vector<Point2f> seeds);

	/**
	 * Compute the centroid for each voronoi facet
	 *
	 * @returns facet centroids to use as new seeds
	 */
	vector<Point2f> compute_centroids();

	/**
	 * Get facet list
	 *
	 * @returns list of facets
	 */
	vector<vector<Point2f>> get_facets();

	/**
	* Computes the patch grid
	*
	* @returns patch grid
	*/
	DataGrid<unsigned char> compute_grid();

	/**
	 * Compute the patch regions for each facet
	 */
	vector<PatchRegion> compute_patches();


	/************************************************
	*  IMSLIC
	***********************************************/

	/**
	 * Apply stretch mapping to a Pixel
	 *
	 * @param Pointer to pixel
	 */
	void stretch(Pixel& m);

	/**
	 * Compute SCLIC distance: sqrt( (ds/Ns)^2 + (dc/Nc)^2 )
	 *
	 * @param m1 First Pixel
	 * @param m2 Second Pixel
	 */
	float slic_distance(const Pixel& m1, const Pixel& m2);

	/**
	 * Compute euclidian distance between 2 Pixels
	 *
	 * @param m1 First Pixel
	 * @param m2 Second Pixel
	 */
	float euclidian_distance(const Pixel& m1, const Pixel& m2);

	/**
	 * Compute angle between 2 vectors
	 *
	 * @param m1 First vector
	 * @param m2 Second vector
	 */
	float angle(const Pixel& m1, const Pixel& m2);

	/**
	* Compute area of stretch mapped unit square in given Position
	*
	* @param row Row of target Pixel
	* @param col Col of target Pixel
	*/
	float phi_area(int row, int col);

	/**
	* Content-sensitive initialization of seeds
	*
	* @param seeds Pointer to seed array
	*/
	vector<Point2f> init_seeds(vector<Point2f> seeds);

	/**
	 * Perform IMSLIC algorithm on the image
	 *
	 * @param seeds Vector with seeds, only the size is important
	 */
	std::vector<Point2f> imslic(vector<Point2f> seeds);

	/**
	 * Remove patches with size below the size threshold, and fill them with adjacent labels
	 */
	void remove_small_patches();

	/**
	 * Get most frequent label of a pixel in an 8-connected neighborhood
	 *
	 * @param row Row of target pixel
	 * @param col Col of target pixel
	 * @param i Label of target pixel
	 */
	int most_frequent_neighbor(int row, int col, int i);

	/**
	 * Build a subgraph of given size around the given position.
	 *
	 * @param s_row Row of target position
	 * @param s_col Col of target position
	 * @param size Size of subgraph
	 * @param check_label If true only pixels with the same label as the target position will be added to the graph
	 */
	void build_graph(Graph& g);

	/**
	 * Get all labels in an 8-connected neighborhood around given position
	 *
	 * @param row Row of target pixel
	 * @param col Col of target pixel
	 */
	vector<int> get_neighbors(int row, int col);

	/**
	 * Compute the patch regions for every label
	 */
	vector<PatchRegion> compute_imslic_patches();

	/**
	 * Check if pixel is a corner of a patch
	 *
	 * @param row Row of target pixel
	 * @param col Col of target pixel
	 */
	bool actual_corner(int row, int col);

	/**
	 * Sort data points along the direction of the patch border, starting from one corner point
	 *
	 * @param vertex First corner of the patch border
	 * @param points Data points of the patch border
	 */
	vector<Point2d> sort_edge(Point2d vertex, vector<Point2d> points);

	/**
	 * Find corner points from an array of point candidates. Computes connected components to
	 * find point clusters and computes the mean values of those clusters.
	 *
	 * @param points Array of corner point candidates
	 * @returns pair of the two corners of the patch border
	 */
	pair<Point2d, Point2d> find_corners(vector<Point2d> points);

	/**
	 * Helper function to find connected components
	 */
	void ccrecur(int v, vector<Point2d> points, bool visited[], vector<Point2d>& comp);

	/**
	 * Helper function to find neighboring pixels for computing connected components
	 */
	vector<int> getAdj(int v, vector<Point2d> points);

	/**
	 * Find two corner points in case no two distinct clusters were found
	 *
	 * @param points Array of corner point candidates
	 * @returns pair of the two corners of the patch border
	 */
	pair<Point2d, Point2d> furthest_distance(vector<Point2d> points);

	/**
	 * Replaces each label with the mode of adjacent labels. Smooths out patch boundaries and
	 * fixes outliers.
	 */
	void mode_filter();

	/**
	 * Get the label matrix
	 */
	vector<vector<int>> get_pixels();

	/**
	 * Sort curves so that they form one continuous path around the patch
	 */
	vector<BezierCurve> sort_curves(vector<BezierCurve> curves);

private:
	cv::Mat m_image;
	const int m_width;
	const int m_height;
	const int m_seed_count;

	vector<vector<Point2f>> facets;
	vector<Point2f> centers;
	DataGrid<unsigned char> m_grid;

	std::vector<std::vector<img_pixel_data>> pixels;
	vector<Point2f> curr_seeds;

	// IMSLIC Params
	const int ITER_MAX = 10;
	const int SIZE_TRESHOLD = 50;
	const float Ns = 0.5f;
	const float Nc = 1.0f;
};

#endif