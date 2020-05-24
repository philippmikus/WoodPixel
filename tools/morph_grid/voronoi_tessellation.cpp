#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "voronoi_tessellation.hpp"

using namespace cv;

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
	float z = 1.0 / (clip.x * line.y - clip.y * line.x);

	return Point2f((x * line.x - y * clip.x) * z, (x * line.y - y * clip.y) * z);
}

void VoronoiTessellation::clip_facets()
{
	Point2f bbox[] = { {0,0}, {(float)m_width,0}, {(float)m_width,(float)m_height}, {0,(float)m_height} };
	int new_facet_size = 0;

	for (int i = 0; i < facets.size(); i++)
	{
		// Sutherland - Hodgman
		std::vector<Point2f> input_facet(32);
		std::vector<Point2f> new_facet(32);

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
			clip2 = bbox[(j+1) % 4];

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

std::vector<Point2f> VoronoiTessellation::poisson_disk_sampling(int iterations)
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
		clip_facets();
		seeds = lloyds_relaxation();
	}

	return seeds;
}

void VoronoiTessellation::voronoi_tiles(std::vector<Point2f> seeds)
{
	Rect rect(0, 0, m_width, m_height);
	Subdiv2D subdiv(rect);
	std::vector<int> indices;

	for (int i = 0; i < seeds.size(); i++)
	{
		subdiv.insert(seeds[i]);
	}

	subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);
}

std::vector<Point2f> VoronoiTessellation::lloyds_relaxation()
{
	std::vector<Point2f> new_seeds;
	std::vector<Point2f> vertices;
	std::vector<Point2f> vectors;

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
