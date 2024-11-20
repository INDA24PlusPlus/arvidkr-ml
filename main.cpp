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
#include <chrono>

//chrono::time_point<std::chrono::system_clock> start;
//start = std::chrono::system_clock::now();
//std::chrono::duration<double> elapsed_seconds = end - start;
//elapsed_second.count();

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

vector<double> weirdProduct(vector<double> v1, vector<double> v2){
    vector<double> v3;
    for (int i = 0; i < v1.size(); i++)v3.push_back(v1[i]*v2[i]);
    return v3;
}

vector<vector<double>> nmatr(vector<double>& v1, vector<double>& v2){
    vector<vector<double> > ret;
    for (int i = 0; i < v1.size(); i++){
        ret.push_back({});
        for (int j = 0; j < v2.size(); j++){
            ret[i].push_back(v2[j]*v1[i]);
        }
    }
    return ret;
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

vector<double> dsig(vector<double> x){
    vector<double> ret;
    for (auto e : x)ret.push_back(sig(e)*(1-sig(-e)));
    return ret;
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
        return lastActivation = vectorAdd(matrixMult(x, weights), biases);
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

    vector<pair<vector<double>, vector<double> > > forward(vector<double> X){
        vector<pair<vector<double>, vector<double> > > ret;
        for (int i = 0; i < layers.size(); i++){
            vector<double> temp_z = layers[i].feedForward(X);
            vector<double> temp_a = activation(temp_z, layers[i].activation_method);
            X = temp_a;
            ret.push_back({temp_z, temp_a});
        }
        return ret;
    }

    void update(int layer_no, vector<vector<double>>& weight_proposals, vector<double>& bias_proposals, double alpha){
        for (int i = 0; i < weight_proposals.size(); i++){
            for (int j = 0; j < weight_proposals[0].size(); j++){
                layers[layer_no].weights[i][j] -= alpha*weight_proposals[i][j];
            }
        }

        for (int i = 0; i < bias_proposals.size(); i++)layers[layer_no].biases[i] -= alpha*bias_proposals[i];
    }

    void back_propagation(vector<pair<vector<double>, vector<double>>>& from_forward, vector<double>& X, vector<double>& Y, double batch_size, double learning_rate){
        vector<double> dz;
        vector<vector<double>> weights;
        vector<pair<vector<vector<double> >, vector<double>>> ret;
        for (int i = layers.size()-1; i >= 0; i--){
            if (i == layers.size()-1)for (int j = 0; j < Y.size(); j++)dz.push_back(2*(from_forward[i].second[j]-Y[j]));
            else dz = weirdProduct(matrixMult(dz, matrixTranspose(weights)), dsig(from_forward[i].first));

            vector<vector<double>> dw;
            if (i == 0)dw = nmatr(dz, X);
            else dw = nmatr(dz, from_forward[i-1].second);

            vector<double> db = dz;
            weights = layers[i].weights;

            for (int i = 0; i < dw.size(); i++)for (int j = 0; j < dw[i].size(); j++)dw[i][j] /= batch_size;
            for (int i = 0; i < db.size(); i++)db[i] /= batch_size;

            update(i, dw, db, learning_rate);
        }
    }
    

};

void train(Network& Net, vector<vector<double>>& train_x, vector<vector<double> >& train_y, int batch_size, double learning_rate){
    vector<vector<double>> ac_x;
    vector<vector<double>> ac_y;
    for (int i = 0; i < batch_size; i++){
        int j = rand()%(train_x.size());
        ac_x.push_back(train_x[j]);
        ac_y.push_back(train_y[j]);
    }

    vector<pair<vector<vector<double>>, vector<double>>> suggestions;

    for (int i = 0; i < batch_size; i++){
        chrono::time_point<chrono::system_clock> starti = chrono::system_clock::now();
        vector<pair<vector<double>, vector<double>>> from_forward = Net.forward(ac_x[i]);
        chrono::time_point<chrono::system_clock> endi = chrono::system_clock::now();

        chrono::duration<double> elapsed_seconds = endi - starti;
        //cout << "forward_time: " << elapsed_seconds.count() << endl;
        
        Net.back_propagation(from_forward, ac_x[i], ac_y[i], batch_size, learning_rate);

        chrono::time_point<chrono::system_clock> endi2 = chrono::system_clock::now();

        elapsed_seconds = endi2 - endi;
        //cout << "back_time: " << elapsed_seconds.count() << endl;


    }

}

double evaluate(Network& Net, vector<vector<double>> X, vector<vector<double>> Y){
    double sum = 0;
    int counter = 0;
    for (auto e : X){
        vector<double> returni = Net.forward(e).back().second;
        for (int i = 0; i < returni.size(); i++){
            sum += (returni[i]-Y[counter][i])*(returni[i]-Y[counter][i]);
        }
        counter++;
    }
    return ((sum)/((double)X.size()));
}

double correcti(Network& Net, vector<vector<double>>& X, vector<vector<double>>& Y){
    double correct = 0;
    double total = 0;
    int counter = 0;
    for (auto e : X){
        vector<double> returni = Net.forward(e).back().second;
        /*cout << "Y: ";
        for (auto li : Y[counter])cout << li << " ";
        cout << endl;
        cout << "returni: ";
        for (auto li : returni)cout << li << " ";
        cout << endl;*/
        int guess = 0;
        double guess_s = 0, actual_s = 0;
        int actual = 0;
        for (int i = 0; i < returni.size(); i++){
            if (returni[i] > guess_s){
                guess_s = returni[i];
                guess = i;
            }
            if (Y[counter][i] > actual_s){
                actual_s = Y[counter][i];
                actual = i;
            }
        }

        correct += (guess == actual);
        total += 1;
        counter++;
    }
    return correct/total;
}

int main(){
    //MNIST
    srand(42);
    vector<vector<double>> train_x;
    vector<vector<double>> train_y;
    vector<vector<double>> test_x;
    vector<vector<double>> test_y;

    freopen("mnist_train.csv", "r", stdin);

    string line; 
    int counter = 0;
    while (cin >> line){
        if (line == "-1")break;
        train_x.push_back({});
        string s = "";
        bool is_first = true;
        for (int i = 0; i < line.size(); i++){
            if (line[i] == ','){
                if (is_first){
                    vector<double> temp_y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    temp_y[stoi(s)] = 1.0;
                    train_y.push_back(temp_y);
                    is_first = false;
                }
                else train_x[counter].push_back(stoi(s));
                s = "";
            }
            else {
                s += line[i];
            }
        }
        train_x[counter++].push_back(stoi(s));
    }

    line = "";
    counter = 0;

    while (cin >> line){
        test_x.push_back({});
        string s = "";
        bool is_first = true;
        for (int i = 0; i < line.size(); i++){
            if (line[i] == ','){
                if (is_first){
                    vector<double> temp_y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    temp_y[stoi(s)] = 1.0;
                    test_y.push_back(temp_y);
                    is_first = false;
                }
                else test_x[counter].push_back(stoi(s));
                s = "";
            }
            else {
                s += line[i];
            }
        }
        test_x[counter++].push_back(stoi(s));
    }


    vector<pair<string, pair<int, int> > > v;
    v.push_back({"fc-layer", {784, 10}});
    v.push_back({"sigmoid", {0,  0}});
    v.push_back({"fc-layer", {10, 10}});
    v.push_back({"softmax", {0,  0}});
    //v.push_back({"fc-layer", {32, 10}});
    //v.push_back({"softmax", {0,  0}});

    Network Net(v);

    cout << train_y[0][5] <<  " " << train_x[0].size() << endl;
    cout << test_y[0][7] <<  " " << test_x[0].size() << endl;

    int amount_of_epochs = 10000, batch_size = 128;
    double learning_rate = 0.15;
    for (int i = 0; i < amount_of_epochs; i++){
        if (i%1000 == 0){
            vector<vector<double>> temp_x;
            vector<vector<double>> temp_y;
            for (int j = 0; j < 1000; j++){
                temp_x.push_back(test_x[j]);
                temp_y.push_back(test_y[j]);
            }

            cout << "After " << i << " epochs: " << evaluate(Net, temp_x, temp_y) << "error" << " correctness:" << correcti(Net, temp_x, temp_y) << endl;
        }
        train(Net, train_x, train_y, batch_size, learning_rate);
    }



    return 0;
}