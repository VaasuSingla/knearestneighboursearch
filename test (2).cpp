#include "VectorDataset.h"

int main(){
    VectorDataset train(784);
    auto start1 = high_resolution_clock::now();
    train.readFromFile("fmnist-train.csv");
    VectorDataset test(784);
    test.readFromFile("fmnist-test.csv");
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(stop1 - start1);
    cout << "Time taken to read from files: " << duration1.count() << " milliseconds" << endl;
    DataVector q;
    VectorDataset result(784);
    auto start2 = high_resolution_clock::now();
    for(int i = 0; i < 100; ++i){
        q = test[i];
        auto start = high_resolution_clock::now();
        result = kNearestNeighbour(train, q, 5);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout<<"distance of nearest neighbours are: ";
        for(int j = 0; j < 5; ++j){
            cout<<result[j].dist(q)<<" ";
        }
        cout << "\nTime taken by " << i << "th call: " << duration.count() << " milliseconds" << endl;
    }
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop2 - start2);
    cout << "Total time taken: " << duration2.count() << "milliseconds" << endl;
    return 0;
}
