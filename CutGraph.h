#ifndef CUTGRAPH_H_
#define CUTGRAPH_H_
#include "graph.h"
class CutGraph {
private:
	Graph<double, double, double> * graph;
public:
	CutGraph();
	CutGraph(int, int);
	int addVertex();
	double maxFlow();
	void addVertexWeights(int, double, double);
	void addEdges(int, int, double);
	bool isSourceSegment(int);
};
#endif
