#include "TreeIndex.h"

TreeIndex* TreeIndex::instance = nullptr;
KDTreeIndex* KDTreeIndex::instance = nullptr;
RPTreeIndex* RPTreeIndex::instance = nullptr;

TreeIndex& TreeIndex::GetInstance()
{
    if (instance == nullptr)
    {
        instance = new TreeIndex;
    }
    return *instance;
}

KDTreeIndex& KDTreeIndex::GetInstance()
{
    if (instance == nullptr)
    {
        instance = new KDTreeIndex;    
        instance -> head = new KDNode;
        instance -> head -> depth = 0;
        instance -> head -> left = nullptr;
        instance -> head -> right = nullptr;
    }
    return *(KDTreeIndex*)instance;
}

RPTreeIndex& RPTreeIndex::GetInstance()
{
    if (instance == nullptr)
    {
        instance = new RPTreeIndex;  
        instance -> head = new RPNode;
        instance -> head -> left = nullptr;
        instance -> head -> right = nullptr;
        instance -> head -> datavectors.resize(60001);
        for(int i = 0; i < 60001; i++){
            instance -> head -> datavectors[i] = i;
        }
    }
    return *(RPTreeIndex*)instance;
}

void KDTreeIndex::MakeTreeKD(KDNode* node, VectorDataset& vd)
{
    //cout << "depth:" << node -> depth << " size:" << node -> datavectors.size() << endl;
    if(node -> depth == 0){
        node -> datavectors.resize(vd.getsize());
        for(int i = 0; i < vd.getsize(); i++){
            node -> datavectors[i] = i;
        }
    }
    if (node->datavectors.size() < 100)
        return;

    pair<int, int> rule = ChooseRule(node, vd);
    //cout << "rule:" << rule.first << " " << rule.second << endl;
    node->median = rule.second;
    vector<int> left;
    vector<int> right;
    for (auto it : node->datavectors)
    {
        if (vd[it].get(rule.first) < rule.second)
        {
            left.push_back(it);
        }
        else
        {
            right.push_back(it);
        }
    }
    node->left = new KDNode;
    node->left->left = nullptr;
    node->left->right = nullptr;
    node->right = new KDNode;
    node->right->left = nullptr;
    node->right->right = nullptr;
    node->left->datavectors = left;
    node->right->datavectors = right;
    node->left->depth = node -> depth + 1;
    node->right->depth = node -> depth + 1;
    this->MakeTreeKD(node->left, vd);
    this->MakeTreeKD(node->right, vd);
}

pair<int, int> KDTreeIndex::ChooseRule(KDNode* node, VectorDataset& vd)
{
    int dimension = node -> depth % vd.getdimension();
    vector<int> v;
    for (auto it : node->datavectors)
    {
        v.push_back(vd[it].get(dimension));
    }
    auto m = v.begin() + v.size() / 2;
    nth_element(v.begin(), m, v.end());
    //cout << "dimension:" << dimension << " median:" << *m << endl;
    return make_pair(dimension, *m);
}

priority_queue<double> KDTreeIndex::SearchKNN(KDNode* node, DataVector& query, int k, VectorDataset& vd, priority_queue<double>& pq)
{
    if(node -> left == nullptr && node -> right == nullptr){
        for(auto it : node -> datavectors){
            pq.push(query.dist(vd[it]));
            if(pq.size() > k){
                pq.pop();
            }
        }
        return pq;
    }
    int dimension = node -> depth % vd.getdimension();
    if(query.get(dimension) < node -> median){
        if(node -> left != nullptr){
            pq = SearchKNN(node -> left, query, k, vd, pq);
        }
        if(pq.size() < k || abs(query.get(dimension) - node -> median) < pq.top()){
            if(node -> right != nullptr){
                pq = SearchKNN(node -> right, query, k, vd, pq);
            }
        }
    }
    else{
        if(node -> right != nullptr){
            pq = SearchKNN(node -> right, query, k, vd, pq);
        }
        if(pq.size() < k || abs(query.get(dimension) - node -> median) < pq.top()){
            if(node -> left != nullptr){
                pq = SearchKNN(node -> left, query, k, vd, pq);
            }
        }
    }
    return pq;
}

void RPTreeIndex::MakeTreeRP(RPNode* node, VectorDataset& dataset)
{
    if(node -> datavectors.size() < 2){
        return;
    }
    pair<double, DataVector> rule = makerule(node, dataset);
    node -> direction = rule.second;
    node -> delta = rule.first;
    vector<int> left;
    vector<int> right;
    for(auto it : node -> datavectors){
        if(dataset[it]*(node -> direction) < node -> delta){
            left.push_back(it);
        }
        else{
            right.push_back(it);
        }
    }
    //cout << left.size() << " " << right.size() << endl;
    node -> left = new RPNode;
    node -> left -> left = nullptr;
    node -> left -> right = nullptr;
    node -> right = new RPNode;
    node -> right -> left = nullptr;
    node -> right -> right = nullptr;
    node -> left -> datavectors = left;
    node -> right -> datavectors = right;
    this -> MakeTreeRP(node -> left, dataset);
    this -> MakeTreeRP(node -> right, dataset);
}

pair<double, DataVector> RPTreeIndex::makerule(RPNode* node, VectorDataset& dataset)
{
    DataVector direction(dataset.getdimension());
    for(int i = 0; i < dataset.getdimension(); i++)
        direction.assign(i, (rand() % 50));
    double norm = direction.norm();
    for(int i = 0; i < dataset.getdimension(); i++)
        direction.assign(i, direction.get(i)/norm);
    vector<double> projections;
    for(auto it : node -> datavectors){
        projections.push_back(dataset[it]*(direction));
    }
    auto m = projections.begin() + projections.size() / 2;
    nth_element(projections.begin(), m, projections.end());
    int maxdist = 0;
    DataVector q = dataset[node -> datavectors[0]];
    for(auto it : node -> datavectors){
        if(q.dist(dataset[it]) > maxdist){
            maxdist = q.dist(dataset[it]);
        }
    }
    double median = *m;
    median +=  (rand()%1000)/1000*6* maxdist / 28;
    int random = rand() % 2;
    if(random)median = -median;
    return make_pair(median, direction);
}

priority_queue<pair<double, int>> RPTreeIndex::SearchKNN(RPNode* node, DataVector& query, int k, VectorDataset& vd, priority_queue<pair<double, int>>& pq)
{
    if(node -> left == nullptr && node -> right == nullptr){
        for(auto it : node -> datavectors){
            pq.push(make_pair(query.dist(vd[it]), it));
            if(pq.size() > k){
                pq.pop();
            }
        }
        return pq;
    }
    if(query*(node -> direction) < node -> delta){
        if(node -> left != nullptr){
            pq = SearchKNN(node -> left, query, k, vd, pq);
        }
        if(pq.size() < k || abs(query*(node -> direction) - node -> delta) < pq.top().first){
            if(node -> right != nullptr){
                pq = SearchKNN(node -> right, query, k, vd, pq);
            }
        }
    }
    else{
        if(node -> right != nullptr){
            pq = SearchKNN(node -> right, query, k, vd, pq);
        }
        if(pq.size() < k || abs(query*(node -> direction) - node -> delta) < pq.top().first){
            if(node -> left != nullptr){
                pq = SearchKNN(node -> left, query, k, vd, pq);
            }
        }
    }
    return pq;
}
