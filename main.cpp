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
    for (int i = 0; i < v1.size(); i++)ret.push_back(v1[i]+v2[i]);
    return ret;
}

vector<double> matrixMult(const vector<double>& x, const vector<vector<double>>& weights){
    vector<double> ret(weights.size(), 0);
    if (weights[0].size() != x.size()){
        cout << "Error matrixMult:  weights[0].size():" << weights[0].size() << "  while   x.size():" << x.size() << "  and weights.size():" << weights.size() << endl; 
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
        for (int j = 0; j < n; j++)ret[i].push_back(((double)(rand()%1000))/1000.0);
    }
    return ret;
}

vector<double> gen_empty_N(int n){
    vector<double> ret;
    for (int i = 0; i < n; i++)ret.push_back(((double)(rand()%1000))/1000.0);
    return ret;
}

double sig(double x){
    return min(1.0/(1.0 + exp(-x)), 1.0);
}

double dsig(double x){
    return min(sig(x)*(1-sig(-x)), x);
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
        for (int i = 0; i < layers.size(); i++){
            //cout << "in_forward: " << i << ": " << layers[i].biases[0] << endl;
            //for (auto e : start)cout << e << " ";
            //cout << endl;
            //cout << "i: " << i << "  start.size():" << start.size() << endl; 
            start = layers[i].feedForward(start);
        }
        
        return start;
    }

    pair<pair<vector<vector<double>>, vector<double>>, vector<double>> back_propagation(Layer L, vector<double> Y, bool is_last, vector<double> init_cost, Layer last_layer){
        vector<vector<double>> weight_proposals = L.weights;
        vector<double> bias_proposals = L.biases; 

        if (is_last)for (int j = 0; j < L.lastActivation.size(); j++)init_cost[j] = 2*(L.lastActivation[j]-Y[j]);
    
        for (int j = 0; j < L.lastActivation.size(); j++){
            bias_proposals[j] = dsig(L.lastActivation[j])*init_cost[j];
            if (abs(bias_proposals[j]) < 0.0001 || abs(bias_proposals[j]) > 1)bias_proposals[j] = 0;
        }

        for (int k = 0; k < L.n; k++){
            for (int j = 0; j < L.lastActivation.size(); j++){
                weight_proposals[j][k] = last_layer.lastActivation[k]*dsig(L.lastActivation[j])*init_cost[j];
                if (abs(weight_proposals[j][k]) < 0.0001 || abs(weight_proposals[j][k]) > 1)weight_proposals[j][k] = 0;
            }
        }
        
        //cout << "LW " << L.weights.size() << " " << L.weights[0].size() << endl;
        //vector<vector<double>> mT = matrixTranspose(L.weights);
        //cout << "mT " << mT.size() << " " << mT[0].size() << endl;


        init_cost = matrixMult(bias_proposals, matrixTranspose(L.weights));
        
        return {{weight_proposals, bias_proposals}, init_cost};
    }

    void update(int layer_no, vector<vector<double>> weight_proposals, vector<double> bias_proposals, double alpha){
        for (int i = 0; i < weight_proposals.size(); i++){
            for (int j = 0; j < weight_proposals[0].size(); j++){
                layers[layer_no].weights[i][j] -= alpha*weight_proposals[i][j];
            }
        }

        for (int i = 0; i < bias_proposals.size(); i++)layers[layer_no].biases[i] -= alpha*bias_proposals[i];
    }

};

void train(Network& Net, vector<vector<double>> train_x, vector<pair<double, double> > train_y, int batch_size, double learning_rate){
    vector<vector<double>> ac_x;
    vector<pair<double, double>> ac_y;
    for (int i = 0; i < batch_size; i++){
        int j = rand()%(train_x.size());
        ac_x.push_back(train_x[j]);
        ac_y.push_back(train_y[j]);
    }

    vector<pair<vector<vector<double>>, vector<double>>> suggestions;


    for (int i = 0; i < batch_size; i++){
        Net.forward(ac_x[i]);
        bool is_last = true;
        vector<double> init_cost = {0, 0};
        int counter = 0;
        for (int j = Net.layers.size()-1; j >= 0; j--){
            pair<pair<vector<vector<double>>, vector<double>>, vector<double>> p1 = Net.back_propagation(Net.layers[j], {ac_y[i].first, ac_y[i].second}, is_last, init_cost, Net.layers.back());
            init_cost = p1.second;
            if (i == 0){
                for (int k = 0; k < p1.first.first.size(); k++){
                    for (int l = 0; l < p1.first.first[0].size(); l++){
                        p1.first.first[k][l] /= batch_size;
                    }
                }
                for (int k = 0; k < p1.first.second.size(); k++)p1.first.second[k] /= batch_size;
                suggestions.push_back(p1.first);
            }
            else {
                for (int k = 0; k < p1.first.first.size(); k++){
                    for (int l = 0; l < p1.first.first[0].size(); l++){
                        suggestions[counter].first[k][l] += p1.first.first[k][l] / batch_size;
                    }
                }
                for (int k = 0; k < p1.first.second.size(); k++)suggestions[counter].second[k] += p1.first.second[k] / batch_size;
            }
            counter++;

            is_last = false;
        }
    }

    reverse(all(suggestions));

    //cout << "SUGGESTIONS" << endl << endl;

    int counter = 0;
    for (auto p : suggestions){
        Net.update(counter, p.first, p.second, learning_rate);

        counter++;
    }

}

double evaluate(Network& Net, vector<vector<double>> X, vector<vector<double>> Y){
    double sum = 0;
    int counter = 0;
    for (auto e : X){
        vector<double> returni = Net.forward(e);
        for (int i = 0; i < returni.size(); i++){
            sum += (returni[i]-Y[counter][i])*(returni[i]-Y[counter][i]);
        }
        counter++;
    }
    return ((sum)/((double)X.size()));
}

int main(){
    //odd or not
    srand(420);
    vector<vector<double>> train_x;
    vector<pair<double, double> > train_y;
    for (int i = 0; i < 8; i++){
        vector<double> temp = {0, 0, 0, 0, 0, 0, 0, 0};
        temp[i] = 1;
        train_x.push_back(temp);
        train_y.push_back({i%2, (i+1)%2});
    }    

    vector<pair<string, pair<int, int> > > v;
    v.push_back({"fc-layer", {8, 8}});
    v.push_back({"sigmoid", {0,  0}});
    v.push_back({"fc-layer", {8, 8}});
    v.push_back({"sigmoid", {0,  0}});
    v.push_back({"fc-layer", {8, 2}});
    v.push_back({"softmax", {0,  0}});

    Network Net(v);
    vector<double> forward_ret = Net.forward(train_x[0]);


    int amount_of_epochs = 10000, batch_size = 16;
    double learning_rate = 0.715;
    for (int i = 0; i < amount_of_epochs; i++){
        if (i%100 == 0){
            vector<vector<double>> temp_y;
            for (int j = 0; j < train_y.size(); j++)temp_y.push_back({train_y[j].first, train_y[j].second});
            cout << "After " << i << " epochs: " << evaluate(Net, train_x, temp_y) << endl;
        }
        train(Net, train_x, train_y, batch_size, learning_rate);
    }


    int counter = 0;
    for (vector<double> x : train_x){
        forward_ret = Net.forward(x);
        cout << ++counter << " " << forward_ret[0] << " " << forward_ret[1] << endl;
    }


    return 0;
}