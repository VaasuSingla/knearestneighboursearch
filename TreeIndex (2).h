#include "VectorDataset.h"
typedef struct Node { 
	vector<int> datavectors;
	int depth, median;
	Node* left; 
	Node* right; 
}KDNode;

typedef struct Node2 {
	vector<int> datavectors;
	DataVector direction;
	double delta;
	Node2* left;
	Node2* right;
}RPNode;

class TreeIndex {
	static TreeIndex* instance;
	protected: 
		TreeIndex() {} 
		~TreeIndex() {} 
	public: 
		static TreeIndex& GetInstance(); 
}; 
	
class KDTreeIndex : public TreeIndex {
	public:
		static KDTreeIndex& GetInstance();
		KDNode* head;
		void MakeTreeKD(KDNode* node, VectorDataset& vd);
		pair<int, int> ChooseRule(KDNode* node, VectorDataset& vd);
		priority_queue<double> SearchKNN(KDNode* node, DataVector& query, int k, VectorDataset& vd, priority_queue<double>& pq);
	private:
		static KDTreeIndex* instance;
		KDTreeIndex(){}; 
		~KDTreeIndex(){}; 
};

class RPTreeIndex : public TreeIndex { 
	public: 
		static RPTreeIndex& GetInstance();
		RPNode* head;
		void MakeTreeRP(RPNode* node, VectorDataset& dataset); 
		pair<double, DataVector> makerule(RPNode* node, VectorDataset& dataset);
		priority_queue<pair<double, int>> SearchKNN(RPNode* node, DataVector& query, int k, VectorDataset& vd, priority_queue<pair<double, int>>& pq);
	private:
		static RPTreeIndex* instance;
		RPTreeIndex(){}; 
		~RPTreeIndex(){};
		pair<int, DataVector> makerule();
};