//Purpose: Header file for VectorDataset class
//Class  : VectorDataset
/*Methods:
  VectorDataset(int dimension)                   - Constructor
  ~VectorDataset()                               - Destructor
  VectorDataset(const VectorDataset& other)      - Copy Constructor
  VectorDataset & operator=(const VectorDataset &other) - Copy Assignment Operator
  void readFromFile(string filename)             - Read the data from the file and store it in the vector
  int getsize()                                  - Return the size of the vector
  int getdimension()                             - Return the dimension of the vector
  int push_back(DataVector dv)                   - Push the DataVector dv to the vector
  DataVector operator[](int index)               - Return the DataVector at index
  void print()                                   - Print the vector
*/
//Data Members:
//  vector<DataVector> vd - Vector to store the DataVector
//  int size              - Size of the vector
//  int dimension         - Dimension of the vector
//
#include "DataVector.h"

class VectorDataset
{
    vector<DataVector> vd;
    int size;
    int dimension;

public:
    VectorDataset(int dimension)
    {
        size = 0;
        this->dimension = dimension;
    }

    ~VectorDataset()
    {
        vd.clear();
    }

    VectorDataset(const VectorDataset &other)
    {
        vd = other.vd;
        size = other.size;
        dimension = other.dimension;
    }

    VectorDataset &operator=(const VectorDataset &other)
    {
        vd = other.vd;
        size = other.size;
        dimension = other.dimension;
        return *this;
    }

    void readFromFile(string filename)
    {
        ifstream file(filename);
        string line;
        while (getline(file, line))
        {
            stringstream ss(line);
            DataVector dv(dimension);
            double value;
            int i = 0;
            while (ss >> value)
            {
                dv.assign(i, value);
                i++;
                if (ss.peek() == ',')
                    ss.ignore();
            }
            vd.push_back(dv);
            size++;
        }
        file.close();
    }

    int getsize()
    {
        return size;
    }

    int getdimension()
    {
        return dimension;
    }

    int push_back(DataVector dv)
    {
        vd.push_back(dv);
        size++;
        return size;
    }

    DataVector operator[](int index)
    {
        return vd[index];
    }

    void print()
    {
        for (int i = 0; i < size; i++)
        {
            vd[i].print();
        }
    }
};

VectorDataset kNearestNeighbour(VectorDataset& vd, DataVector& dv, int k);
