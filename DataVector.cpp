#include "DataVector.h"

DataVector::DataVector(int dimension)
{
    v.resize(dimension);
}

DataVector::~DataVector()
{
    v.clear();
}

DataVector::DataVector(const DataVector& other)
{
    v = other.v;
}

DataVector & DataVector::operator=(const DataVector &other)
{
    v = other.v;
    return *this;
}

void DataVector::setDimension(int dimension)
{
    v.clear();
    v.resize(dimension,0);
}

DataVector DataVector::operator+(const DataVector &other)
{
    DataVector result(v.size());
    for (int i = 0; i < v.size(); i++)
    {
        result.v[i] = v[i] + other.v[i];
    }
    return result;
}

DataVector DataVector::operator-(const DataVector &other)
{
    DataVector result(v.size());
    for (int i = 0; i < v.size(); i++)
    {
        result.v[i] = v[i] - other.v[i];
    }
    return result;
}

double DataVector::operator*(const DataVector &other)
{
    double result = 0;
    for (int i = 0; i < v.size(); i++)
    {
        result += v[i] * other.v[i];
    }
    return result;
}

void DataVector::print()
{
    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}

void DataVector::assign(int index, double value)
{
    v[index] = value;
}

double DataVector::norm()
{
    return sqrt((*this) * (*this));
}

double DataVector::dist(const DataVector &other)
{
    return ((*this - other).norm());
}

int DataVector::get(int index)
{
    return v[index];
}