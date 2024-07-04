#include <iostream>
#include "TreeIndex.h"


int main()
{
    TreeIndex& ti = TreeIndex::GetInstance();
    KDTreeIndex& kdti = KDTreeIndex::GetInstance();
    //RPTreeIndex& rpti = RPTreeIndex::GetInstance();
    VectorDataset train(784);
    train.readFromFile("fmnist-train.csv");
    auto start1 = high_resolution_clock::now();
    kdti.MakeTreeKD(kdti.head, train);
    //rpti.MakeTreeRP(rpti.head, train);
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(stop1 - start1);
    //cout << "Time taken to make KDTree: " << duration1.count() << " milliseconds" << endl;
    cout << "Time taken to make RPTree: " << duration1.count() << " milliseconds" << endl;
    int k = 5;
    priority_queue<double> pq;
    DataVector query(784);
    VectorDataset test(784);
    test.readFromFile("fmnist-test.csv");
    auto start2 = high_resolution_clock::now();
    for (int i = 0; i < 100; ++i)
    {
        query = test[i];
        auto start = high_resolution_clock::now();
        pq = kdti.SearchKNN(kdti.head, query, k, train, pq);
        //pq = rpti.SearchKNN(rpti.head, query, k, train, pq);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        // cout << "distance of nearest neighbours are: ";
        // for (int j = 0; j < k; ++j)
        // {
        //     cout << pq.top().first << " ";
        //     pq.pop();
        // }
        cout << "Time taken by " << i << "th call: " << duration.count() << " milliseconds" << endl;
    }
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop2 - start2);
    cout << "Total time taken: " << duration2.count() << "milliseconds" << endl;
    return 0;
}

