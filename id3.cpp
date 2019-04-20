//This is a project for id3 algorithm

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <stack>
#include <map>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <cstdio>
 
using namespace std;

//Load the file data to matrix dataset
vector< vector<int> > inputFile(string filename){
    int i, k, data, lineID;
    vector< vector<int> > dataset;
    if(filename.empty()){
        return dataset;
    }
    freopen("lenses.data", "r", stdin);

    for(k = 0; k < 24; k++){
        vector<int> line;
        cin >> lineID;
        for(i = 0; i < 5; i++){
            cin >> data;
            line.push_back(data);
        }
        dataset.push_back(line);
    }
    fclose(stdin);
    return dataset;
}

//Caculate the ShannonEnt of dataset 
double calShannonEnt(const vector< vector<int> > &dataset){
    int labels[4];
    int k;
    double shannonEnt = 0.0;
    for(auto data : dataset){
        labels[data.size() - 1]++;
    }
    for(k = 1; k <= 3; k++){
        double pi = double(labels[k])/double(dataset.size());
        shannonEnt += (-pi * log10(pi)/log10(2));
    }
    return shannonEnt;
}

//Split a subDataset with value in matrix asix column
vector< vector<int> > splitDataset(const vector< vector<int> >& dataset, int axis, int value){
    vector< vector<int> > result;
    for (auto data : dataset){
        if (data[axis] != value)
            continue;
        vector<int> one;
        for (int i = 0; i < data.size(); i++){
            if (i == axis) continue;
            else one.push_back(data[i]);
        }
        result.push_back(one);
    }
    return result;
}

//Split the subDataset of labels
vector<int> splitLabel(const vector<int>& label, int axis){
    vector<int> result;
    for (int i = 0; i < label.size(); i++){
        if (i == axis) continue;
        else result.push_back(label[i]);
    }
    return result;
}

map<int, int>getFeatureClassify(vector< vector<int> > dataset, int axis){
    map<int, int> result;
    for(auto data: dataset){
        result[data[axis]]++;
    }
    return result;
}

//Caculate the infogain
double getInfogain(vector< vector<int> > dataset, int axis){
    auto baseShannonEnt = calShannonEnt(dataset);
    double targetShannonEntSum = 0.0;
    
    map<int, int> classify = getFeatureClassify(dataset, axis);

    auto it = classify.begin();
    for(it; it != classify.end(); it++){
        auto splitDatasets = splitDataset(dataset, axis, it -> first);
        auto pi = double(splitDatasets.size()) / double(dataset.size());
        auto shannonEnt = calShannonEnt(splitDatasets);
        targetShannonEntSum += pi * shannonEnt;
    }

    return baseShannonEnt - targetShannonEntSum;

}

//Find the best split feature
int getBestSplitFeature(vector< vector<int> > dataset){
    int i;
    double bestInfogain = 0.0;
    int bestFeatureAxis;
    int featureCount = dataset[0].size() - 1;

    for(i = 0; i < featureCount; i++){
        auto infogain = getInfogain(dataset, i);
        if (bestInfogain == 0){
            bestInfogain = infogain;
            bestFeatureAxis = i;
        } else if (infogain > bestInfogain){
            bestInfogain = infogain;
            bestFeatureAxis = i;
        }
    }
    return bestFeatureAxis;
}

//When there is only one label, choose the largest probability for the result
int getLargestProbability(vector< vector<int> > dataset, int axis, int value){
    map<int, int> result;
    for (auto data : dataset){
        if (data[axis] != value) continue;
        result[data[1]]++;
    }
    int maxx = 0;
    int v;
    auto it = result.begin();
    for (; it != result.end(); it++){
        if (it->second > maxx){
            maxx = it -> second;
            v = it -> first;
        }
    }
    return v;
}

struct node{
    int value;
    vector<node*> next;
    vector<int> property;
    node(int v = 0){
        value = v;
    }
};

node *createTree(vector< vector<int> > dataset, vector<int> labels){
    node *root = new node;
    if(dataset[0].size() == 2){
        root -> value = labels[0];
        auto featureClassify = getFeatureClassify(dataset, 0);
        auto it = featureClassify.begin();
        for (; it != featureClassify.end(); it++){
            //Choose the largest probability result
            int result = getLargestProbability(dataset, 0, it->first);
            node *Node = new node(result);
            root->next.push_back(Node);
            root->property.push_back(it->first);
        }
        return root;
    }

    int axis = getBestSplitFeature(dataset);
    root -> value = labels[axis];

    auto featureClassify = getFeatureClassify(dataset, axis);

    auto it = featureClassify.begin();
    for (; it != featureClassify.end(); it++){
        auto subDataset = splitDataset(dataset, axis, it->first);
        node* Node = nullptr;
 
        //if shannnonEnt == 0, the subDataset is in same class.
        if (calShannonEnt(subDataset) == 0){
            int result = subDataset[0][subDataset[0].size()-1];
            Node = new node(result);
 
        //if shannnonEnt != 0, the subDataset is in different class.
        }else {
            auto subLabel = splitLabel(labels, axis);
            Node = createTree(subDataset, subLabel);
        }
        root->next.push_back(Node);
        root->property.push_back(it->first);
    }
 
    return root;
}

//Test the id3 algorithm
int predict(node* tree, vector<int> labels, vector<int> data){
    while(tree->next.size()) {
        bool judge = true;
        for (int i = 0; i < labels.size(); i++) {
            if (labels[i] == tree -> value) {
                for (int j = 0; j < tree -> property.size(); j++) {
                    if (data[i] == tree -> property[j]){
                        tree = tree->next[j];
                        judge = false;
                        break;
                    }
                }
                break;
            }
        }
        if (judge) return 0;
    }
    return tree->value;
}

int main(){
    int i, j;
    vector< vector<int> > dataset = inputFile("license.data");
    int l[4] = {1, 2, 3, 4};
    vector<int> labels(l, l + 4);
    
    node *tree = createTree(dataset, labels);

    int d1[4] = {3, 2, 2, 2};  //3
    vector<int> data1(d1, d1 + 4);

    int d2[4] = {2, 2, 1, 2};   //2
    vector<int> data2(d2, d2 + 4);

    int d3[4] = {2, 1, 1, 1};   //3
    vector<int> data3(d3, d3 + 4);

    int d4[4] = {1, 2, 2, 1};    //3
    vector<int> data4(d4, d4 + 4);

    cout << predict(tree, labels, data1) << endl;
    cout << predict(tree, labels, data2) << endl;
    cout << predict(tree, labels, data3) << endl;
    cout << predict(tree, labels, data4) << endl;
}


 