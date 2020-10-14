#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "voronoi_tessellation.hpp"
#include <omp.h>
#include <algorithm>

# define INF 0x3f3f3f3f 

using namespace cv;

typedef pair<int, float> iPair;

/*
Graph implementation from https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-using-priority_queue-stl/
*/
Graph::Graph(int V)
{
	this->V = V;
	adj.resize(V);
}

void Graph::add_edge(int u, int v, float w)
{
	adj[u].push_back(make_pair(v, w));
	adj[v].push_back(make_pair(u, w));
}

vector<float> Graph::shortest_path(int src)
{
	priority_queue< iPair, vector <iPair>, greater<iPair> > pq;

	vector<float> dist(V, INF);

	pq.push(make_pair(0, src));
	dist[src] = 0;

	while (!pq.empty())
	{
		int u = pq.top().second;
		pq.pop();

		list<pair<int, float>>::iterator i;
		for (i = adj[u].begin(); i != adj[u].end(); ++i)
		{
			int v = (*i).first;
			float weight = (*i).second;

			if (dist[v] > dist[u] + weight)
			{
				dist[v] = dist[u] + weight;
				pq.push(make_pair(dist[v], v));
			}
		}
	}

	return dist;
}

bool is_left(Point2f p, Point2f a, Point2f b)
{
	return ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) > 0;
}

Point2f intersection(Point2f clip1, Point2f clip2, Point2f a, Point2f b)
{
	Point2f clip(clip1.x - clip2.x, clip1.y - clip2.y);
	Point2f line(a.x - b.x, a.y - b.y);
	float x = clip1.x * clip2.y - clip1.y * clip2.x;
	float y = a.x * b.y - a.y * b.x;
	float z = 1.0f / (clip.x * line.y - clip.y * line.x);

	return Point2f((x * line.x - y * clip.x) * z, (x * line.y - y * clip.y) * z);
}

void VoronoiTessellation::clip_facets()
{
	Point2f bbox[] = { {0,0}, {(float)m_width - 5,0}, {(float)m_width - 5,(float)m_height - 5}, {0,(float)m_height - 5} };
	int new_facet_size = 0;

	for (int i = 0; i < facets.size(); i++)
	{
		// Sutherland - Hodgman
		vector<Point2f> input_facet(32);
		vector<Point2f> new_facet(32);

		for (int k = 0; k < facets[i].size(); k++) {
			new_facet[k] = facets[i][k];
		}

		new_facet_size = facets[i].size();
		Point2f clip1, clip2, a, b;

		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < new_facet_size; k++) {
				input_facet[k] = new_facet[k];
			}

			int counter = 0;
			clip1 = bbox[j];
			clip2 = bbox[(j + 1) % 4];

			for (int k = 0; k < new_facet_size; k++)
			{
				a = input_facet[k];
				b = input_facet[(k + 1) % new_facet_size];

				if (is_left(a, clip1, clip2) && is_left(b, clip1, clip2))
				{
					new_facet[counter] = b;
					counter++;
				}

				else if (!is_left(a, clip1, clip2) && !is_left(b, clip1, clip2)) {}

				else if (is_left(a, clip1, clip2) && !is_left(b, clip1, clip2))
				{
					new_facet[counter] = intersection(clip1, clip2, a, b);
					counter++;
				}

				else if (!is_left(a, clip1, clip2) && is_left(b, clip1, clip2))
				{
					new_facet[counter] = intersection(clip1, clip2, a, b);
					counter++;
					new_facet[counter] = b;
					counter++;
				}
			}
			new_facet_size = counter;
		}

		for (int j = 0; j < new_facet_size; j++)
		{
			facets[i].resize(new_facet_size);
			facets[i][j] = new_facet[j];
		}
	}

}

void VoronoiTessellation::voronoi_tiles(vector<Point2f> seeds)
{
	Rect rect(0, 0, m_width, m_height);
	Subdiv2D subdiv(rect);
	vector<int> indices;

	for (int i = 0; i < seeds.size(); i++)
	{
		subdiv.insert(seeds[i]);
	}

	subdiv.getVoronoiFacetList(vector<int>(), facets, centers);
}

vector<Point2f> VoronoiTessellation::compute_centroids()
{
	vector<Point2f> new_seeds;
	vector<Point2f> vertices;
	vector<Point2f> vectors;

	for (size_t i = 0; i < facets.size(); i++)
	{
		vertices.resize(facets[i].size());
		vectors.resize(facets[i].size());
		for (size_t j = 0; j < facets[i].size(); j++)
		{
			vertices[j] = facets[i][j];
			vectors[j] = facets[i][j] - vertices[0];
		}

		Point2f centroid(0.0f, 0.0f);
		double total_area = 0.0f;
		for (size_t j = 1; j < facets[i].size() - 1; j++)
		{
			double area = (vectors[j + 1].x * vectors[j].y - vectors[j + 1].y * vectors[j].x) / 2;
			centroid.x += area * (vertices[0].x + vertices[j].x + vertices[j + 1].x) / 3;
			centroid.y += area * (vertices[0].y + vertices[j].y + vertices[j + 1].y) / 3;
			total_area += area;
		}

		centroid.x /= total_area;
		centroid.y /= total_area;

		new_seeds.push_back(centroid);
	}

	return new_seeds;
}

vector<Point2f> VoronoiTessellation::start(int iterations, bool superpixels)
{
	vector<Point2f> seeds;

	for (size_t i = 0; i < m_seed_count; i++)
	{
		int x = rand() % m_height;
		int y = rand() % m_width;
		seeds.push_back(Point2f(x, y));
	}

	if (superpixels) {
		seeds = imslic(seeds);
		return curr_seeds;
	}
	else {
		for (size_t i = 0; i < iterations; i++)
		{
			voronoi_tiles(seeds);
			clip_facets();
			seeds = compute_centroids();
		}
		return seeds;
	}
}

vector<vector<Point2f>> VoronoiTessellation::get_facets()
{
	return facets;
}

vector<vector<int>> VoronoiTessellation::get_pixels()
{
	mode_filter();
	vector<vector<int>> ret;
	ret.resize(m_height, vector<int>(m_width, 0));

	for (int i = 0; i < m_height; i++) {
		for (int j = 0; j < m_width; j++) {
			ret[i][j] = pixels[i][j].label;
		}
	}

	return ret;
}

DataGrid<unsigned char> VoronoiTessellation::compute_grid()
{
	DataGrid<unsigned char> grid;
	vector<Point2f> points;

	for (int i = 0; i < facets.size(); i++) {
		for (int j = 0; j < facets[i].size(); j++) {
			points.push_back(facets[i][j]);
		}
	}

	for (int i = 0; i < points.size(); i++) {
		for (int j = 0; j < points.size(); j++) {
			if (i != j && points[i] == points[j]) {
				points.erase(points.begin() + j);
				j--;
			}
		}
	}

	int size = (int)sqrt(points.size());
	grid.resize(size, size);

	cv::Mat X = grid.to_mat<float>();

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			grid(i, j) = points[i * size + j];
		}
	}
	m_grid = grid;
	return grid;
}

vector<PatchRegion> VoronoiTessellation::compute_patches()
{
	vector<PatchRegion> patches;

	for (int i = 0; i < facets.size(); i++) {
		vector<BezierCurve> lines;
		for (int j = 0; j < facets[i].size(); j++) {
			lines.emplace_back(BezierCurve(facets[i][j], facets[i][(j + 1) % facets[i].size()], 1));
		}
		patches.emplace_back(PatchRegion(0, { 0,0 }, lines, vector<BezierCurve>(), vector<BezierCurve>(), std::vector<BezierCurve>()));
	}

	return patches;
}

void VoronoiTessellation::stretch(Pixel* m)
{
	float lambda1 = 1.f / Ns;
	float lambda2 = 1.f / Nc;
	
	m->u *= lambda1;
	m->v *= lambda1;
	m->l *= lambda2;
	m->a *= lambda2;
	m->b *= lambda2;
}

float VoronoiTessellation::slic_distance(Pixel* m1, Pixel* m2)
{
	float ds = sqrt(pow(m1->u - m2->u, 2) + pow(m1->v - m2->v, 2));
	float dc = sqrt(pow(m1->l - m2->l, 2) + pow(m1->a - m2->a, 2) + pow(m1->b - m2->b, 2));

	float D = sqrt(pow(ds / Ns, 2) + pow(dc / Nc, 2));

	return D;
}

float VoronoiTessellation::euclidian_distance(Pixel* m1, Pixel* m2)
{
	float ret = sqrt(pow(m2->u - m1->u, 2) + pow(m2->v - m1->v, 2) + pow(m2->l - m1->l, 2) + pow(m2->a - m1->a, 2) + pow(m2->b - m1->b, 2));
	return ret;
}

float VoronoiTessellation::angle(Pixel* m1, Pixel* m2)
{
	float dot = m1->u * m2->u + m1->v * m2->v + m1->l * m2->l + m1->a * m2->a + m1->b * m2->b;
	float length1 = sqrt(pow(m1->u, 2) + pow(m1->v, 2) + pow(m1->l, 2) + pow(m1->a, 2) + pow(m1->b, 2));
	float length2 = sqrt(pow(m2->u, 2) + pow(m2->v, 2) + pow(m2->l, 2) + pow(m2->a, 2) + pow(m2->b, 2));

	float cosangle = dot / (length1 * length2);

	float ret = acos(cosangle);
	return ret;
}

float VoronoiTessellation::phi_area(int row, int col) {
	Vec3b a1 = m_image.at<Vec3b>(row - 1, col - 1);
	Vec3b a2 = m_image.at<Vec3b>(row - 1, col);
	Vec3b a3 = m_image.at<Vec3b>(row - 1, col + 1);

	Vec3b a4 = m_image.at<Vec3b>(row, col - 1);
	Vec3b a5 = m_image.at<Vec3b>(row, col);
	Vec3b a6 = m_image.at<Vec3b>(row, col + 1);

	Vec3b a7 = m_image.at<Vec3b>(row + 1, col - 1);
	Vec3b a8 = m_image.at<Vec3b>(row + 1, col);
	Vec3b a9 = m_image.at<Vec3b>(row + 1, col + 1);

	Pixel m1 = { row - 0.5f, col - 0.5f, (a1[0] + a2[0] + a4[0] + a5[0]) / 4.f, (a1[1] + a2[1] + a4[1] + a5[1]) / 4.f, (a1[2] + a2[2] + a4[2] + a5[2]) / 4.f };
	Pixel m2 = { row + 0.5f, col - 0.5f, (a2[0] + a3[0] + a5[0] + a6[0]) / 4.f, (a2[1] + a3[1] + a5[1] + a6[1]) / 4.f, (a2[2] + a3[2] + a5[2] + a6[2]) / 4.f };
	Pixel m3 = { row + 0.5f, col + 0.5f, (a4[0] + a5[0] + a7[0] + a8[0]) / 4.f, (a4[1] + a5[1] + a7[1] + a8[1]) / 4.f, (a4[2] + a5[2] + a7[2] + a8[2]) / 4.f };
	Pixel m4 = { row - 0.5f, col + 0.5f, (a5[0] + a6[0] + a8[0] + a9[0]) / 4.f, (a5[1] + a6[1] + a8[1] + a9[1]) / 4.f, (a5[2] + a6[2] + a8[2] + a9[2]) / 4.f };
	stretch(&m1);
	stretch(&m2);
	stretch(&m3);
	stretch(&m4);

	float area1 = (1.f / 2.f) * slic_distance(&m2, &m1) * slic_distance(&m2, &m3) * sin(angle(&(m1 - m2), &(m3 - m2)));
	float area2 = (1.f / 2.f) * slic_distance(&m4, &m3) * slic_distance(&m4, &m1) * sin(angle(&(m3 - m4), &(m1 - m3)));
	float total_area = area1 + area2;
	
	return total_area;
}

void VoronoiTessellation::init_seeds(vector<Point2f>* seeds)
{
	float cum_area = 0;
	vector<float> A(m_height * m_width);
	for (int row = 1; row < m_height-1; row++) {
		for (int col = 1; col < m_width-1; col++) {
			cum_area += pixels[row][col].area;
			A[row * m_width + col] = cum_area;
		}
	}

	for (int i = 0; i < seeds->size(); i++) {
		float sample = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / A[m_height * m_width - 1]));

		for (int j = 1; j < A.size(); j++) {
			if (A[j - 1] < sample && sample <= A[j])
				seeds->at(i) = Point2f((float)(j % m_width), (float)(j / m_width));
		}
	}
}

std::vector<Point2f> VoronoiTessellation::imslic(std::vector<Point2f> seeds)
{
	cvtColor(m_image, m_image, COLOR_BGR2Lab);
	pixels.resize(m_height, std::vector<img_pixel_data>(m_width, { -1, FLT_MAX, 0 }));

	// compute stretch mapped unit square area for each pixel
	for (int row = 1; row < m_height - 1; row++) {
		for (int col = 1; col < m_width - 1; col++) {
			pixels[row][col].area = phi_area(row, col);
		}
	}

	init_seeds(&seeds);

	float local_search_range = 0;
	for (int i = 0; i < pixels.size(); i++) {
		for (int j = 0; j < pixels[0].size(); j++) {
			local_search_range += pixels[i][j].area;
		}
	}
	local_search_range = 4 * local_search_range / seeds.size();

	int S = (int)sqrt(m_width * m_height / seeds.size());

	for (int iter = 0; iter < ITER_MAX; iter++)
	{
		printf("Iter: %d\n", iter);
		curr_seeds = seeds;

		// init distance and label
		for (int i = 0; i < pixels.size(); i++) {
			for (int j = 0; j < pixels[0].size(); j++) {
				pixels[i][j].distance = FLT_MAX;
				pixels[i][j].label = -1;
			}
		}

		std::vector<float> scaling_factors(seeds.size(), 1.f);

		#pragma omp parallel for
		for (int i = 0; i < seeds.size(); i++) {
			float region_area = 0;
			for (int u = -S; u < S; u++) {
				for (int v = -S; v < S; v++) {
					if (seeds[i].x + u > 0 && seeds[i].x + u < m_height-1
						&& seeds[i].y + v > 0 && seeds[i].y + v < m_width-1)
						region_area += pixels[seeds[i].x + u][seeds[i].y + v].area;
				}
			}
			if (region_area != 0)
				scaling_factors[i] = sqrt(local_search_range / region_area);
		}

		omp_lock_t writelock;
		omp_init_lock(&writelock);

		#pragma omp parallel for
		for (int i = 0; i < seeds.size(); i++) {
			int size = (int)2.0f * S * scaling_factors[i];
			int x = (int)seeds[i].x;
			int y = (int)seeds[i].y;

			Graph subgraph = build_graph(x, y, size, false);

			vector<float> geo_distances = subgraph.shortest_path(x * m_width + y);

			for (int u = -size; u < size; u++) {
				for (int v = -size; v < size; v++) {
					if (x + u > 0 && x + u < m_height-1
						&& y + v > 0 && y + v < m_width-1)
						if (geo_distances[(x + u) * m_width + (y + v)] < pixels[x + u][y + v].distance) {
							omp_set_lock(&writelock);
							pixels[x + u][y + v].distance = geo_distances[(x + u) * m_width + (y + v)];
							pixels[x + u][y + v].label = i;
							omp_unset_lock(&writelock);
						}
				}
			}
		}

		omp_destroy_lock(&writelock);

		#pragma omp parallel for
		for (int i = 0; i < seeds.size(); i++) {
			int size = (int)2.0f * S * scaling_factors[i];
			int x = (int)seeds[i].x;
			int y = (int)seeds[i].y;

			float sum_i_area = 0;
			vector<int> indices;
			for (int row = 0; row < m_height; row++) {
				for (int col = 0; col < m_width; col++) {
					if (pixels[row][col].label == i) {
						sum_i_area += pixels[row][col].area;
						indices.push_back(row * m_width + col);
					}
				}
			}

			if (indices.size() <= 3) continue;

			Graph subgraph = build_graph(x, y, size, true);

			int pi = seeds[i].x * m_width + seeds[i].y;
			int pj = indices[rand() % indices.size()];
			int pk = indices[rand() % indices.size()];

			vector<float> geo_pi = subgraph.shortest_path(pi);
			vector<float> geo_pj = subgraph.shortest_path(pj);
			vector<float> geo_pk = subgraph.shortest_path(pk);

			while (!(geo_pi[pk] + geo_pj[pk]) > 1.2f * geo_pi[pj]) {
				pj = indices[rand() % indices.size()];
				pk = indices[rand() % indices.size()];

				geo_pj = subgraph.shortest_path(pj);
				geo_pk = subgraph.shortest_path(pk);
			}

			float a[3][3] = {
				{0, pow((float)geo_pi[pj], 2), pow((float)geo_pi[pk], 2)},
				{pow((float)geo_pj[pi], 2), 0, pow((float)geo_pj[pk], 2)},
				{pow((float)geo_pi[pk], 2), pow((float)geo_pj[pk], 2), 0}
			};

			Mat delta3(3, 3, CV_32FC1, a);

			float b[3][3] = {
				{ 2.f / 3.f, -1.f / 3.f, -1.f / 3.f},
				{-1.f / 3.f,  2.f / 3.f, -1.f / 3.f},
				{-1.f / 3.f, -1.f / 3.f,  2.f / 3.f}
			};

			Mat H3(3, 3, CV_32FC1, b);
			Mat B3 = (-H3 * (delta3 * H3)) / 2;

			Mat eigen_values, eigen_vectors;
			eigen(B3, eigen_values, eigen_vectors);

			int dim = 0;
			for (int j = 0; j < 3; j++)
				if (eigen_values.at<float>(j, 0) > 0) dim++;

			if (dim >= 2) {
				float l1[2][1] = { { sqrt(eigen_values.at<float>(0,0)) * eigen_vectors.at<float>(0,0)}, {sqrt(eigen_values.at<float>(1,0)) * eigen_vectors.at<float>(1,0)} };
				float l2[2][1] = { { sqrt(eigen_values.at<float>(0,0)) * eigen_vectors.at<float>(0,1)}, {sqrt(eigen_values.at<float>(1,0)) * eigen_vectors.at<float>(1,1)} };
				float l3[2][1] = { { sqrt(eigen_values.at<float>(0,0)) * eigen_vectors.at<float>(0,2)}, {sqrt(eigen_values.at<float>(1,0)) * eigen_vectors.at<float>(1,2)} };

				// embed remaining pixels
				vector<pair<Mat, int>> remaining;

				remaining.push_back(make_pair(Mat(2, 1, CV_32FC1, l1), pi));
				remaining.push_back(make_pair(Mat(2, 1, CV_32FC1, l2), pj));
				remaining.push_back(make_pair(Mat(2, 1, CV_32FC1, l3), pk));

				for (const auto& ind : indices) {
					if (ind != pi != pj != pk) {
						Vec3f deltak = { pow((float)geo_pi[ind], 2), pow((float)geo_pj[ind], 2), pow((float)geo_pk[ind], 2) };

						float c[2][3] = {
							{eigen_vectors.at<float>(0, 0) / sqrt(eigen_values.at<float>(0, 0)), eigen_vectors.at<float>(0, 1) / sqrt(eigen_values.at<float>(0, 0)), eigen_vectors.at<float>(0, 2) / sqrt(eigen_values.at<float>(0, 0))},
							{eigen_vectors.at<float>(1, 0) / sqrt(eigen_values.at<float>(1, 0)), eigen_vectors.at<float>(1, 1) / sqrt(eigen_values.at<float>(1, 0)), eigen_vectors.at<float>(1, 2) / sqrt(eigen_values.at<float>(1, 0))}
						};

						Mat eigen_calc(2, 3, CV_32FC1, c);

						Vec3f delta3_mean = { (delta3.at<float>(0,0) + delta3.at<float>(1,0) + delta3.at<float>(2,0)) / 3,
							(delta3.at<float>(0,1) + delta3.at<float>(1,1) + delta3.at<float>(2,1)) / 3,
							(delta3.at<float>(0,2) + delta3.at<float>(1,2) + delta3.at<float>(2,2)) / 3 };

						Mat t = eigen_calc * Mat(deltak - delta3_mean) / 2;
						float ta = t.at<float>(0, 0);
						float tc = t.at<float>(1, 0);
						remaining.push_back(make_pair(t, ind));
					}
				}

				float sum_x = 0;
				float sum_y = 0;
				for (int j = 0; j < remaining.size(); j++) {
					float test = remaining[j].first.at<float>(0, 0);
					sum_x += remaining[j].first.at<float>(0, 0);
					sum_y += remaining[j].first.at<float>(1, 0);
				}

				Vec2f center_embed = { sum_x / sum_i_area, sum_y / sum_i_area };

				float min_norm = FLT_MAX;
				int min_index = -1;
				for (int j = 0; j < remaining.size(); j++) {
					float curr_norm = norm(remaining[j].first - center_embed);
					if (curr_norm < min_norm) {
						min_norm = curr_norm;
						min_index = remaining[j].second;
					}
				}
				if (min_norm == FLT_MAX) continue;

				Vec2f center = { (float)(min_index / m_width), (float)(min_index % m_width) };
				seeds[i] = center;
			}
			else 
				continue;
		}
	}
	remove_small_patches();
	return seeds;
}

void VoronoiTessellation::remove_small_patches()
{
	vector<int> patch_sizes(m_seed_count);
	for (int row = 0; row < pixels.size(); row++) {
		for (int col = 0; col < pixels[0].size(); col++) {
			if (pixels[row][col].label >= 0)
				patch_sizes[pixels[row][col].label]++;
		}
	}

	for (int i = 0; i < m_seed_count; i++) {
		if (patch_sizes[i] < SIZE_TRESHOLD) {
			for (int row = 1; row < pixels.size()-1; row++) {
				for (int col = 1; col < pixels[0].size()-1; col++) {
					if (pixels[row][col].label == i) {
						pixels[row][col].label = most_frequent_neighbor(row, col, i);
					}
				}
			}

		}
	}
}

int VoronoiTessellation::most_frequent_neighbor(int row, int col, int l)
{
	vector<int> neighbors;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			if (pixels[row + i][col + j].label != l) {
				neighbors.push_back(pixels[row + i][col + j].label);
			}
		}
	}

	sort(neighbors.begin(), neighbors.end());
	int max_count = 1, res = neighbors[0], curr_count = 1;
	
	for (int i = 1; i < neighbors.size(); i++) {
		if (neighbors[i] == neighbors[i - 1])
			curr_count++;
		else {
			if (curr_count > max_count) {
				max_count = curr_count;
				res = neighbors[i - 1];
			}
			curr_count = 1;
		}
	}

	if (curr_count > max_count)
	{
		max_count = curr_count;
		res = neighbors[neighbors.size() - 1];
	}
	
	return res;
}

Graph VoronoiTessellation::build_graph(int s_row, int s_col, int size, bool check_label)
{
	int V = m_width * m_height;
	Graph g(V);

	for (int row = s_row - size; row <= s_row + size; row++) {
		for (int col = s_col - size; col <= s_col + size; col++) {
			if (row > 0 && row < m_height - 2 && col > 0 && col < m_width - 2) {

				if (check_label)
					if (pixels[row][col].label != pixels[s_row][s_col].label) continue;

				Vec3b a = m_image.at<Vec3b>(row, col);
				Pixel m1 = { (float)row, (float)col, (float)a[0], (float)a[1], (float)a[2] };
				stretch(&m1);
				for (int u = -1; u <= 1; u++) {
					for (int v = -1; v <= 1; v++) {
						if (!(u == 0 && v == 0)) {

							if (check_label)
								if (pixels[row+u][col+v].label != pixels[s_row][s_col].label) continue;

							Vec3b b = m_image.at<Vec3b>(row + u, col + v);
							Pixel m2 = { (float)row + u, (float)col + v, (float)b[0], (float)b[1], (float)b[2] };
							stretch(&m2);
							g.add_edge(row * m_width + col, (row + u) * m_width + col + v, euclidian_distance(&m1, &m2));
						}
					}
				}
			}
		}
	}

	return g;
}

vector<int> VoronoiTessellation::get_neighbors(int row, int col)
{
	vector<int> neighbors;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			neighbors.push_back(pixels[row + i][col + j].label);

			if (row + i == 4)			neighbors.push_back(-1);
			if (row + i == m_height - 6)			neighbors.push_back(-2);
			if (col + j == 4)			neighbors.push_back(-3);
			if (col + j == m_width - 6)			neighbors.push_back(-4);
		}
	}

	sort(neighbors.begin(), neighbors.end());
	neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());

	return neighbors;
}

vector<PatchRegion> VoronoiTessellation::compute_imslic_patches()
{
	printf("patches start\n");
	vector<PatchRegion> patches;

	vector<vector<vector<pair<int, Point2d>>>> edge_points(m_seed_count);		// patches -> boundarys -> points (label, coord)
	vector<vector<vector<pair<int, Point2d>>>> corner_points(m_seed_count);

	for (int row = 5; row <= m_height-7; row++) {
		for (int col = 5; col <= m_width-7; col++) {
			vector<int> neighbors = get_neighbors(row, col);

			if (neighbors.size() > 2) {	//ecke
				if (!actual_corner(row, col)) continue;
				for (int i = 0; i < neighbors.size(); i++) {
					for (int j = 0; j < neighbors.size(); j++) {
						if (i == j || neighbors[i] < 0) continue;
						if (corner_points[neighbors[i]].size() != 0) {
							bool added = false;
							for (int k = 0; k < corner_points[neighbors[i]].size(); k++) {
								if (neighbors[j] == corner_points[neighbors[i]][k][0].first) {
									corner_points[neighbors[i]][k].push_back(make_pair(neighbors[j], Point2d(col, row)));
									added = true;
								}
							}
							if (!added) {
								vector<pair<int, Point2d>> temp(0);
								temp.push_back(make_pair(neighbors[j], Point2d(col, row)));
								corner_points[neighbors[i]].push_back(temp);
							}
						}
						else {
							vector<pair<int, Point2d>> temp(0);
							temp.push_back(make_pair(neighbors[j], Point2d(col, row)));
							corner_points[neighbors[i]].push_back(temp);
						}
					}
				}
			}
			else if (neighbors.size() == 2) {	// kante

				// 2. patch
				if (neighbors[1] < 0) continue;
				if (!edge_points[neighbors[1]].empty()) {
					bool added = false;
					for (int k = 0; k < edge_points[neighbors[1]].size(); k++) {
						if (neighbors[0] == edge_points[neighbors[1]][k][0].first) {
							edge_points[neighbors[1]][k].push_back(make_pair(neighbors[0], Point2d(col, row)));
							added = true;
						}
					}
					if (!added) {
						vector<pair<int, Point2d>> temp(0);
						temp.push_back(make_pair(neighbors[0], Point2d(col, row)));
						edge_points[neighbors[1]].push_back(temp);
					}
				}
				else {
					vector<pair<int, Point2d>> temp(0);
					temp.push_back(make_pair(neighbors[0], Point2d(col, row)));
					edge_points[neighbors[1]].push_back(temp);
				}

				// 1. patch
				if (neighbors[0] < 0) continue;
				if (!edge_points[neighbors[0]].empty()) {
					bool added = false;
					for (int k = 0; k < edge_points[neighbors[0]].size(); k++) {
						if (neighbors[1] == edge_points[neighbors[0]][k][0].first) {
							edge_points[neighbors[0]][k].push_back(make_pair(neighbors[1], Point2d(col, row)));
							added = true;
						}
					}
					if (!added) {
						vector<pair<int, Point2d>> temp(0);
						temp.push_back(make_pair(neighbors[1], Point2d(col, row)));
						edge_points[neighbors[0]].push_back(temp);
					}
				}
				else {
					vector<pair<int, Point2d>> temp(0);
					temp.push_back(make_pair(neighbors[1], Point2d(col, row)));
					edge_points[neighbors[0]].push_back(temp);
				}

			}
		}
	}

	for (int i = 0; i < corner_points.size(); i++) {
		printf("i: %d\n", i);
		vector<BezierCurve> curves;

		for (int j = 0; j < corner_points[i].size(); j++) {
			int edge_ind = -1;
			for (int k = 0; k < edge_points[i].size(); k++) {
				if (corner_points[i][j][0].first == edge_points[i][k][0].first) edge_ind = k;
			}

			// small edge without edge point
			if (edge_ind == -1) {
				if (corner_points[i][j].size() < 2) continue;
				vector<Point2d> temp2(corner_points[i][j].size());
				for (int k = 0; k < corner_points[i][j].size(); k++) {
					temp2[k] = corner_points[i][j][k].second;
				}
				pair<Point2d, Point2d> corners = furthest_distance(temp2);
				curves.emplace_back(BezierCurve(corners.first, corners.second, 1));
				continue;
			}

			vector<Point2d> temp(edge_points[i][edge_ind].size());
			for (int k = 0; k < edge_points[i][edge_ind].size(); k++) {
				temp[k] = edge_points[i][edge_ind][k].second;
			}
			
			if (corner_points[i][j].size() < 2) {
				printf("fehler\n");
				continue;
			}
			
			vector<Point2d> temp2(corner_points[i][j].size());
			for (int k = 0; k < corner_points[i][j].size(); k++) {
				temp2[k] = corner_points[i][j][k].second;
			}
			
			pair<Point2d, Point2d> corners = find_corners(temp2);

			if (corners.first.x == -1.f) {
				corners = furthest_distance(temp2);
			}
			
			Point2d first = corners.first;
			Point2d last = corners.second;

			vector<Point2d> temp3 = sort_edge(first, temp);
			temp3.push_back(last);

			curves.emplace_back(BezierCurve::fit_cubic(temp3));
		}
		if (!edge_points.empty())
			if(!curves.empty())
				patches.emplace_back(PatchRegion(0, { 0,0 }, curves, vector<BezierCurve>(), vector<BezierCurve>(), std::vector<BezierCurve>()));
	}


	return patches;
}

bool VoronoiTessellation::actual_corner(int row, int col)
{
	int center = pixels[row][col].label;

	if (row == 5 || row == m_height - 7 || col == 5 || col == m_width - 7)		return true;

	vector<int> n = { 
		pixels[row - 1][col - 1].label,
		pixels[row - 1][col    ].label,
		pixels[row - 1][col + 1].label,
		pixels[row    ][col + 1].label,
		pixels[row + 1][col + 1].label,
		pixels[row + 1][col    ].label,
		pixels[row + 1][col - 1].label,
		pixels[row    ][col - 1].label
	};

	for (int i = 0; i < n.size(); i++) {
		if (n[i] != center && n[(i + 1) % n.size()] != center && n[i] != n[(i + 1) % n.size()]) return true;
	}

	return false;
}

vector<Point2d> VoronoiTessellation::sort_edge(Point2d vertex, vector<Point2d> points) {
	vector<Point2d> ret;
	Point2d curr = vertex;
	
	int size = points.size() + 1;

	while (points.size() > 0) {
		ret.push_back(curr);

		for (int i = 0; i < points.size(); i++) {
			float dist = norm(curr - points[i]);
			
			if (dist < 3.0f) {
				points.erase(points.begin() + i);
			}
		}
		
		float min = 99999.f;
		int min_ind = -1;
		for (int i = 0; i < points.size(); i++) {
			float dist = norm(curr - points[i]);

			if (dist < min) {
				min = dist;
				min_ind = i;
			}
		}
		if (min_ind == -1) break;
		curr = points[min_ind];
		points.erase(points.begin() + min_ind);
	}

	return ret;
}

pair<Point2d, Point2d> VoronoiTessellation::find_corners(vector<Point2d> points) {
	bool* visited = new bool[points.size()];
	for (int v = 0; v < points.size(); v++)
		visited[v] = false;

	vector<vector<Point2d>> components;
	vector<Point2d>* comp = new vector<Point2d>();

	for (int i = 0; i < points.size(); i++) {
		if (visited[i] == false) {
			ccrecur(i, points, visited, comp);
			components.push_back(*comp);
			comp->clear();
		}
	}

	delete(visited);
	if (components.size() != 2) {
		return(make_pair(Point2d( -1,-1 ), Point2d( -1,-1 )));
	}

	Point2d first;
	Point2d second;

	float sumx = 0, sumy = 0;
	for (int i = 0; i < components[0].size(); i++) {
		sumx += components[0][i].x;
		sumy += components[0][i].y;
	}

	first = {sumx / components[0].size(), sumy / components[0].size()};

	sumx = 0;
	sumy = 0;
	for (int i = 0; i < components[1].size(); i++) {
		sumx += components[1][i].x;
		sumy += components[1][i].y;
	}

	second = { sumx / components[1].size(), sumy / components[1].size() };

	return make_pair(first, second);
}

void VoronoiTessellation::ccrecur(int v, vector<Point2d> points, bool visited[], vector<Point2d>* comp) {
	if (visited[v] == true) return;
	visited[v] = true;
	comp->push_back(points[v]);

	vector<int> adj = getAdj(v, points);

	for (int i = 0; i < adj.size(); i++) {
		if (!visited[adj[i]])
			ccrecur(adj[i], points, visited, comp);
	}
}

vector<int> VoronoiTessellation::getAdj(int v, vector<Point2d> points) {
	vector<int> adj;

	for (int i = 0; i < points.size(); i++) {
		if (norm(points[v] - points[i]) == 1.0f) {
			adj.push_back(i);
		}
	}

	return adj;
}

void VoronoiTessellation::mode_filter() {
	std::vector<std::vector<img_pixel_data>> new_pixels;
	new_pixels.resize(m_height, std::vector<img_pixel_data>(m_width, { -1, 999999, 0 }));

	for (int row = 1; row < pixels.size() - 1; row++) {
		for (int col = 1; col < pixels[0].size() - 1; col++) {

			vector<int> neighbors;
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					neighbors.push_back(pixels[row + i][col + j].label);
				}
			}

			sort(neighbors.begin(), neighbors.end());
			int max_count = 1, res = neighbors[0], curr_count = 1;

			for (int i = 1; i < neighbors.size(); i++) {
				if (neighbors[i] == neighbors[i - 1])
					curr_count++;
				else {
					if (curr_count > max_count) {
						max_count = curr_count;
						res = neighbors[i - 1];
					}
					curr_count = 1;
				}
			}

			if (curr_count > max_count)
			{
				max_count = curr_count;
				res = neighbors[neighbors.size() - 1];
			}

			
			new_pixels[row][col].label = res;
		}
	}
	pixels = new_pixels;
}

pair<Point2d, Point2d> VoronoiTessellation::furthest_distance(vector<Point2d> points) {
	int max_dist = 0, ind1, ind2;

	for (int i = 0; i < points.size(); i++) {
		for (int j = 0; j < points.size(); j++) {
			float dist = norm(points[i] - points[j]);

			if (dist > max_dist) {
				max_dist = dist;
				ind1 = i;
				ind2 = j;
			}
		}
	}

	return make_pair(points[ind1], points[ind2]);
}