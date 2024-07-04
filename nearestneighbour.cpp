// Description: Implementation of the kNearestNeighbour function
//Algorithm:
// 1. Create a priority queue of pairs of double and int to store <distance, index>.
// 2. Push the first k elements of the vector dataset into the priority queue.
// 3. For the rest of the elements, if the distance of the current element from the data vector is less than the distance of the top element of the priority queue, pop the top element and push the current element into the priority queue.
// 4. Create a new vector dataset and push the elements of the priority queue into the new vector dataset.
// 5. Return the new vector dataset.
// 6. The time complexity of this function is O(nlogk) where n is the size of the vector dataset and k is the number of nearest neighbours.
// 7. The space complexity of this function is O(k) where k is the number of nearest neighbours.
// 8. The function takes a vector dataset and a data vector as input and returns a vector dataset.
// 9. The function is called from the main function in test.cpp.
#include "VectorDataset.h"


VectorDataset kNearestNeighbour(VectorDataset& vd, DataVector& dv, int k)
{
    int dimension = vd.getdimension();
    int size = vd.getsize();
    priority_queue<pair<double, int>> pq;
    for (int i = 0; i < k; ++i)
    {
        pq.push(make_pair((vd[i].dist(dv)), i));
    }
    for (int i = k; i < size; ++i)
    {
        double distance = (vd[i].dist(dv));
        if (distance < pq.top().first)
        {
            pq.pop();
            pq.push(make_pair(distance, i));
        }
    }
    VectorDataset result(dimension);
    while (!pq.empty())
    {
        result.push_back(vd[pq.top().second]);
        pq.pop();
    }
    return result;
}