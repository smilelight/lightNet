/***************************************************************************
*
* Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
*
**************************************************************************/

/*
* @file main.cpp
* @author lidefang(lidefang@baidu.com)
* @date 2022/08/10
* @brief 测试c++实现mlp
*/

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <exception>
#include <sstream>
#include <numeric>

using namespace std;

#define Matrix std::vector<std::vector<float> > 
#define Tensor std::vector<float>

struct max_info
{
    int idx;
    float prob;

    std::string to_string() {
        return "{ idx: " + std::to_string(idx) + ", prob:" + std::to_string(prob)+ " }";
    }
};

max_info get_max_prob(Tensor x) {
    if (x.size() <= 0) {
        throw "get_max_prob size must > 0";
    }
    int max_idx = 0;
    for (size_t i = 1; i < x.size(); i++)
    {
        if (x[i] > x[max_idx]) {
            max_idx = i;
        }
    }
    return max_info{max_idx, x[max_idx]};
}


inline float sigmoid_func(float x) {
  return 1 / (1 + exp(-x));
}

inline float tanh_func(float x) {
    return tanh(x);
}

inline Tensor softmax(Tensor x) {
    if (x.size() <= 0) {
        return x;
    }
    Tensor y;
    for (size_t i = 0; i < x.size(); i++)
    {
        y.emplace_back(std::exp(x[i]));
    }
    
    float sum = std::accumulate(y.begin(), y.end(), 0);
    for (size_t i = 0; i < y.size(); i++)
    {
        y[i] /= sum;
    }
    return y;
}

std::string print_vec(Tensor x) {
    if (x.size() <= 0) {
        return "";
    }
    stringstream s;
    s << "{ " << std::to_string(x[0]);
    for (size_t i = 1; i < x.size(); i++)
    {
        s << ", " << std::to_string(x[i]);
    }
    s << " }";
    return s.str();
}

std::string print_shape(vector<int> shape) {
    if (shape.size() != 2) {
        throw "shape size not match 2";
    }
    return "( " + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + " )";
}

class NetException: public exception {
public:
    NetException(){}
    virtual ~NetException() {}

    NetException(std::string info) {
        this->info = info;
    }

    const char* what() const throw () {
        return info.c_str();
    }
private:
    std::string info;
};

class Net {
public:
    Net() {}
    virtual ~Net() {}

    virtual int size();
};

class Netron {
public:
    Netron(){}
    virtual ~Netron() {}

    Netron(Tensor weight, float bias, std::string activate_type="none") {
        this->weight = weight;
        this->bias = bias;
        this->activate_type = activate_type;
    }

    float forward(Tensor x);

    float activate(float x) {
        if (activate_type == "tanh") {
            return tanh_func(x);
        } else if (activate_type == "sigmoid") {
            return sigmoid_func(x);
        }
        return x;
    }

    void set_activate_type(std::string activate_type) {
        this->activate_type = activate_type;
    }

    const int size() {
        return weight.size();
    }

private:
    Tensor weight;
    float bias;
    std::string activate_type;
};

float Netron::forward(Tensor x) {
    if (x.size() != weight.size()) {
        return -1;
    }
    float sum = bias;
    for (size_t i = 0; i < x.size(); i++)
    {
        sum += x[i] * weight[i];
    }
    return activate(sum - 0.0);
}

class Linear {
public:
    Linear() {}
    virtual ~Linear() {}

    Linear(vector<Netron> netron_list, std::string activate_type="tanh") {
        if (netron_list.size() <= 0) {
            throw NetException("size <= 0");
        }
        this->activate_type = activate_type;
        int item_size = netron_list[0].size();
        netron_list[0].set_activate_type(activate_type);
        for (size_t i = 1; i < netron_list.size(); i++)
        {
            if (netron_list[i].size() != item_size) {
                throw NetException("size not match, " + std::to_string(netron_list[i].size()) + "!=" + std::to_string(item_size));
            }
            netron_list[i].set_activate_type(activate_type);
        }
        this->netron_list = netron_list;
    }

    float activate(float x) {
        return sigmoid_func(x);
    }

    const int size() {
        return netron_list.size();
    }

    Tensor forward(Tensor x);

    const vector<int> shape() {
        return {netron_list[0].size(), size()};
    }

private:
    vector<Netron> netron_list;
    std::string activate_type;
};

Tensor Linear::forward(Tensor x) {
    int input_size = shape()[0];
    if (input_size != x.size()) {
        throw NetException("size not match");
    }
    Tensor res;
    for (auto &netron : netron_list)
    {
        res.emplace_back(netron.forward(x));
    }

    return res;
    
}

class MLP {
public:
    MLP() {}
    virtual ~MLP() {}

    MLP(vector<Linear> layer_list) {
        if (layer_list.size() <= 0) {
            throw NetException("layer_size must > 0");
        }
        for (size_t i = 0; i < layer_list.size() - 1; i++)
        {
            auto pre_shape = layer_list[i].shape();
            auto next_shape = layer_list[i].shape();
            if (pre_shape[1] != next_shape[0]) {
                throw NetException("layer shape not match, please check " + to_string(pre_shape[1]) + "!=" + to_string(next_shape[0]));
            }
        }
        this->layer_list = layer_list;
    }

    const vector<int> shape() {
        return {layer_list[0].shape()[0], layer_list[layer_list.size() - 1].shape()[1]};
    }

    Tensor forward(Tensor input);

private:
    vector<Linear> layer_list;
};

Tensor MLP::forward(Tensor input) {
    Tensor res = input;
    for (auto &layer : layer_list)
    {
        res = layer.forward(res);
    }
    return res;
}

int main(int argc, char const *argv[])
{
    Tensor input = {3, 4, 2};
    Netron m = Netron({1, -1, 3}, 1, "tanh");
    Netron n = Netron({-1, 1, 5}, 1, "sigmoid");
    Netron o = Netron({-1, -1, 9}, 1);
    Linear l = Linear({m, n, o});

    Linear l2 = Linear({m, n, o});

    Linear l3 = Linear({m, n, o}, "none");

    float res = m.forward(input);
    cout << res << endl;
    
    auto ret = l.forward(input);
    cout << print_vec(ret) << endl;
    auto prob = softmax(ret);
    cout << print_vec(prob) << endl;
    cout << get_max_prob(prob).to_string() << endl;

    try
    {
        cout << "linear shape: " << print_shape(l.shape()) << endl;
        MLP mlp = MLP({l, l2, l3});
        cout << print_shape(mlp.shape()) << endl;
        auto f = mlp.forward(input);
        cout << print_vec(f) << endl;
    }
    catch(exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}

