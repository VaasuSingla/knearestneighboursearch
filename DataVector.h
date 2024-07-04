// Purpose : Header file for DataVector class
// Class   : DataVector
/*Methods:
  DataVector(int dimension = 0)                   - Constructor
  ~DataVector()                                   - Destructor
  DataVector(const DataVector& other)             - Copy Constructor
  DataVector & operator=(const DataVector &other) - Copy Assignment Operator
  void setDimension(int dimension)                - Set the dimension of the vector and clear the previous data
  DataVector operator+(const DataVector &other)   - Overloaded + operator
  DataVector operator-(const DataVector &other)   - Overloaded - operator
  double operator*(const DataVector &other)       - Overloaded * operator
  void print()                                    - Print the vector
  void assign(int index, int value)               - Assign value to a particular index
  double norm()                                   - Calculate the norm of the vector using the formula sqrt(v[0]^2 + v[1]^2 + ... + v[n-1]^2)
  double dist(const DataVector &other)            - Calculate the distance between two vectors using norm(v1 - v2)
*/
//Data Members:
//  vector <double> v - Vector to store the data
//  int dimension     - Dimension of the vector

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <queue>
#include <chrono>

using namespace std;
using namespace std::chrono;
class DataVector
{
    vector <double> v;
    int dimension;
    public:
    DataVector(int dimension = 0);
    ~DataVector();
    DataVector(const DataVector& other);
    DataVector & operator=(const DataVector &other);
    void setDimension(int dimension);
    DataVector operator+(const DataVector &other);
    DataVector operator-(const DataVector &other);
    double operator*(const DataVector &other);
    void print();
    void assign(int index, double value);
    double norm();
    double dist(const DataVector &other);
    int get(int index);
};
