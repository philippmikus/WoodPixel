#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "voronoi_tessellation.hpp"

#define EPSILON 1e-9

using namespace cv;

void VoronoiTessellation::poisson_disk_sampling(int iterations)
{
	std::vector<Point2f> seeds;
	for (size_t i = 0; i < m_seed_count; i++)
	{
		int x = rand() % m_width;
		int y = rand() % m_height;

		seeds.push_back(Point2f(x,y));
	}

	for (size_t i = 0; i < iterations; i++)
	{
		voronoi_tiles(seeds);
		seeds = lloyds_relaxation();
	}
}

void VoronoiTessellation::voronoi_tiles(std::vector<Point2f> seeds)
{
	Rect rect(0, 0, m_width, m_height);
	Subdiv2D subdiv(rect);

	for (std::vector<Point2f>::iterator it = seeds.begin(); it != seeds.end(); it++)
	{
		subdiv.insert(*it);
	}

	subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);
}

std::vector<Point2f> VoronoiTessellation::lloyds_relaxation()
{
	std::vector<Point2f> new_seeds;
	std::vector<Point2f> verts;
	std::vector<Vec2f> vectors;
	for (size_t i = 0; i < facets.size(); i++) 
	{
		verts.resize(facets[i].size());
		vectors.resize(facets[i].size());
		for (size_t j = 0; j < facets[i].size(); j++)
		{
			verts[j] = facets[i][j];
			vectors[j] = facets[i][j] - verts[0];
		}

		Point2f centroid(0.0f, 0.0f);
		double total_area = 0.0f;
		for (size_t j = 1; j < facets[i].size() - 1; j++)
		{
			double area = (vectors[i + 1][0] * vectors[i][1] - vectors[i + 1][1] * vectors[i][0]) / 2;
			total_area += area;
			centroid.x += area * (verts[0].x + verts[i].x + verts[i + 1].x) / 3;
			centroid.y += area * (verts[0].y + verts[i].y + verts[i + 1].y) / 3;
		}
		centroid.x /= total_area;
		centroid.y /= total_area;

		new_seeds.push_back(centroid);
	}

	return new_seeds;
}
