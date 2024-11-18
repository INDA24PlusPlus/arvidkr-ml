#include <iostream>
#include <cmath>
#include <unordered_map>
#include <map>
#include <set>
#include <queue>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

#define all(x)  (x).begin(), (x).end()
#define rall(x)  (x).rbegin(), (x).rend()

using namespace std;

typedef long long ll;

vector<double> vectorAdd(const vector<double>& v1, const vector<double>& v2){
    vector<double> ret;
    if (v1.size() != v2.size()){
        cout << "ERROR vectorAdd, v1 and v2 have different sizes! v1:" << v1.size() << " v2:" << v2.size() << endl;
        return {-1000000000001};
    }
    for (int i = 0; i < v1.size(); i++)ret.push_back(v1[0]+v2[0]);
    return ret;
}

vector<double> matrixMult(const vector<double>& x, const vector<vector<double>>& weights){
    vector<double> ret(weights.size(), 0);
    if (weights[0].size() != x.size()){
        cout << "Error matrixMult:  weights[0].size():" << weights[0].size() << "  while   x.size():" << x.size() << endl; 
        return {-100000000001};
    }
    for (int i = 0; i < weights.size(); i++){
        for (int j = 0; j < weights[0].size(); j++){
            ret[i] += weights[i][j]*x[j];
        }
    }
    return ret;
}

vector<vector<double> > matrixTranspose(const vector<vector<double> >& M){
    vector<vector<double> > M1;
    for (int j = 0; j < M[0].size(); j++){
        M1.push_back({});
        for (int i = 0; i < M.size(); i++){
            M1[j].push_back(M[i][j]);
        }
    }
    return M1;
}

vector<double> innerProduct(vector<double> v1, vector<double> v2){
    vector<double> v3;
    for (int i = 0; i < v1.size(); i++)v3.push_back(v1[i]*v2[i]);
    return v3;
}

vector<vector<double> > gen_empty_NxM(int n, int m){
    vector<vector<double> > ret;
    for (int i = 0; i < m; i++){
        ret.push_back({});
        for (int j = 0; j < n; j++)ret[i].push_back(0.5);
    }
    return ret;
}

vector<double> gen_empty_N(int n){
    vector<double> ret;
    for (int i = 0; i < n; i++)ret.push_back(0.5);
    return ret;
}

double sig(double x){
    return 1.0/(1.0 + exp(-x));
}

double dsig(double x){
    return sig(x)*(1-sig(-x));
}

vector<double> sigmoid_it(vector<double> x){
    vector<double> ret;
    for (double e : x)ret.push_back(sig(e));
    return ret;
}


vector<double> softmax_it(vector<double> x){
    vector<double> ret;
    double sum = 0;
    for (double e : x)sum += exp(e);
    for (double e : x)ret.push_back(exp(e)/sum);
    return ret;
}

vector<double> activation(vector<double> x, int method){
    switch(method){
        case 0:
            return sigmoid_it(x);
            break;
        case 1:
            return softmax_it(x);
            break;
        case 2:
            return x;
            break;
        default:
            cout << "nothing implemented in activation! method:" << method << endl;
            return {-1};
    }
}

struct Layer {
    int layer_id; //0 -> fc, 1 -> activation, 2 -> softmax
    int n, m; //nuvarande storlek, nästa storlek
    vector<vector<double>> weights; //n x m matris
    vector<double> biases;
    vector<double> lastActivation;
    int activation_method;
    Layer(int activation_layer, int n1, int m1, vector<vector<double>> saved_weights, vector<double> saved_biases, int amethod){
        activation_method = amethod;
        layer_id = activation_layer;
        n = n1;
        m = m1;
        weights = saved_weights;
        biases = saved_biases;
    }

    vector<double> feedForward(vector<double> x){
        return lastActivation = activation(vectorAdd(matrixMult(x, weights), biases), activation_method);
    }
};


struct Network{
    vector<Layer> layers;
    Network(vector<pair<string, pair<int, int> > > v){
        string last = "";
        for (int i = v.size()-1; i >= 0; i--){
            if (v[i].first == "fc-layer"){
                int amethod = 2;
                if (last == "sigmoid")amethod = 0;
                else if (last == "softmax")amethod = 1;
                Layer L(0, v[i].second.first, v[i].second.second, gen_empty_NxM(v[i].second.first, v[i].second.second), gen_empty_N(v[i].second.second), amethod);
                layers.push_back(L);
            }
            else last = v[i].first;
        }
        reverse(all(layers));
    }

    vector<double> forward(vector<double> start){
        cout << layers.size() << endl;
        for (Layer& L : layers){
            start = L.feedForward(start);
            cout << L.activation_method << endl;
            //start = activation(start, 0);
            cout << "Steppp: ";
            for (auto e : start)cout << e << " ";
            cout << endl;
        }
        return start;
    }

    pair<pair<vector<vector<double>>, vector<double>>, vector<double>> back_propagation(Layer L, vector<double> Y, bool is_last, vector<double> init_cost, Layer last_layer){
        vector<vector<double>> weight_proposals = L.weights;
        vector<double> bias_proposals = L.biases; 

        if (is_last)for (int j = 0; j < L.lastActivation.size(); j++)init_cost[j] = 2*(L.lastActivation[j]-Y[j]);
    
        for (int j = 0; j < L.lastActivation.size(); j++)bias_proposals[j] = dsig(L.lastActivation[j])*init_cost[j];

        for (int k = 0; k < L.n; k++){
            for (int j = 0; j < L.lastActivation.size(); j++){
                weight_proposals[j][k] = last_layer.lastActivation[k]*dsig(L.lastActivation[j])*init_cost[j];
            }
        }
        
        return {{weight_proposals, bias_proposals}, init_cost};
    }

    void update(int layer_no, vector<vector<double>> weight_proposals, vector<double> bias_proposals){
        for (int i = 0; i < weight_proposals.size(); i++){
            for (int j = 0; j < weight_proposals[0].size(); j++){
                layers[layer_no].weights[i][j] -= weight_proposals[i][j];
            }
        }

        for (int i = 0; i < bias_proposals.size(); i++)layers[layer_no].biases[i] -= bias_proposals[i];
    }

};

int main(){
    //odd or not
    srand(42);
    vector<double> train_x;
    vector<pair<double, double> > train_y;
    for (int i = 0; i < 8; i++){
        train_x.push_back(i+1);
        train_y.push_back({i%2, (i+1)%2});
    }    

    vector<pair<string, pair<int, int> > > v;
    v.push_back({"fc-layer", {1, 8}});
    v.push_back({"sigmoid", {0,  0}});
    v.push_back({"fc-layer", {8, 8}});
    v.push_back({"sigmoid", {0,  0}});
    v.push_back({"fc-layer", {8, 2}});
    v.push_back({"softmax", {0,  0}});

    Network Net(v);

    vector<double> forward_ret = Net.forward({train_x[0]});
    cout << "forward_ret: ";
    for (auto e : forward_ret)cout << e << " ";
    cout << endl;



    return 0;
}