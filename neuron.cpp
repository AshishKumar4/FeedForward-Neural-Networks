#include "iostream"
#include "fstream"
#include "vector"
#include "algorithm"
#include "main.h"
#include "IDX_Drivers\idx.cpp"
#include "random"
#include "chrono"

default_random_engine gen;

using namespace std;

double learning_rate = 0.005;

double lambda = 0;

double alpha = 0.815;

double beta = 0.1;

double theta = 0;

struct lbl_info
{
  int lbl;
  double avg;
};

vector<lbl_info> lblInfo;

//normal_distribution<double> dis_2(0, 1);

class neuron
{

public:
  vector<double> iweights;
  vector<double> deltaW;
  vector<neuron*> inputs;

  vector<double> oweights;
  vector<neuron*> outputs;
  int in_no;
  int out_no;

  double sum;
  double output;

  double Error;

  double bias;

  neuron()
  {
    in_no = 0;
    out_no = 0;
    sum = 0;
    output = 0;
    Error = 0;
    //srand(time(0));
  //  bias = dis_2(gen);//(double)(rand()%10000)/1000;
  }

  double result()
  {
    sum = 0;//bias;
    for(int i = 0; i < in_no ; i++)
    {
      sum += iweights[i]*inputs[i]->output;
    }
    if(in_no)
      output = Func(sum);
    return output;
  }

  double result_s()
  {
    sum = 0;
    for(int i = 0; i < in_no ; i++)
    {
      sum += iweights[i]*inputs[i]->output;
    }
    return sum;
  }

  void link(neuron* n, double weight)
  {
    ++in_no;
    deltaW.push_back(0);
    inputs.push_back(n);
    iweights.push_back(weight);
    n->outputs.push_back(this);
    n->oweights.push_back(weight);
    ++n->out_no;
  }
};

class neuralNet
{
public:
  vector<vector<neuron*>> layers;
  int nl;
  vector<int> lsz;

  neuralNet(int n, int* opt)
  {
    nl = n;
    vector<neuron*> tmp2;
    for(int j = 0; j < opt[0]; j++)
    {
      neuron* nn = new neuron();
      tmp2.push_back(nn);
    }
    layers.push_back(tmp2);

    for(int i = 1; i < n; i++)
    {
      vector<neuron*> tmp;
      normal_distribution<double> dis(0, 1/(pow(opt[i-1], 0.5)));
      for(int j = 0; j < opt[i]; j++)
      {
        neuron* nn = new neuron();
        for(int k = 0; k < layers[i-1].size(); k++)
        {
          //cout<<"{"<<dis(gen)<<"}";
          nn->link(layers[i-1][k], dis(gen));
        }
        tmp.push_back(nn);
      }
      layers.push_back(tmp);
    }
  }

  void input(int arr[])
  {
    for(int i = 0; i < layers[0].size(); i++)
    {
      layers[0][i]->output = arr[i];
    }
  }

  void input(idx_content im)
  {
    int d = 0;
    for(int i = 0; i < layers[0].size(); i++)
    {
      d += im.values[i];
    }
    double e = (double)d;
    e /= (28*28);
  //  cout<<"=>"<<e;
    for(int i = 0; i < layers[0].size(); i++)
    {
      layers[0][i]->output = tanh((double)(((double)im.values[i] - e))/(e*_gamma));//2*(FuncNeural((double)im.values[i]/1000) - 0.5);
    //  cout<<"=>"<<layers[0][i]->output;
    }
  }

  void output()
  {
    for(int i = 1; i < layers.size(); i++)
    {
      for(int j = 0; j < layers[i].size(); j++)
      {
        layers[i][j]->result();
      }
    }
  }

  int output(int lbl)
  {
    //cout<<"["<<lbl<<"]";
    output();
    int c = 0;
    double d = 0;
    double td = 0;
    for(int i = 0; i < layers[layers.size()-1].size(); i++)
    {
      td += pow(phi, layers[layers.size()-1][i]->sum);
  //    cout<<"=>{"<<layers[layers.size()-1][i]->sum<<"}";
    }

    for(int i = 0; i < layers[layers.size()-1].size(); i++)
    {
      layers[layers.size()-1][i]->output = pow(phi, layers[layers.size()-1][i]->sum)/(td);
      if(i == lbl)
        layers[layers.size()-1][i]->Error = layers[layers.size()-1][i]->output - 1;
      else
        layers[layers.size()-1][i]->Error = layers[layers.size()-1][i]->output;

      if(d  < layers[layers.size()-1][i]->output)
      {
        d = layers[layers.size()-1][i]->output;
        c = i;
      }
    //  cout<<"->"<<layers[layers.size()-1][i]->output;
    }
    //cout<<"=> "<<c;
    return c;
  }

  void backProp()
  {
    for(int i = layers.size() - 2; i > 0; i--)
    {
      for(int j = 0; j < layers[i].size(); j++)
      {
        layers[i][j]->Error = 0;
        for(int k = 0; k < layers[i+1].size(); k++)
        {
          layers[i][j]->Error += (layers[i+1][k]->Error * layers[i][j]->oweights[k]);
        }
        layers[i][j]->Error *= 1 - (layers[i][j]->output*layers[i][j]->output);//FuncDerivative(layers[i][j]->sum);
      }
    }
  }

  void changeWeights()
  {
    for(int i = 1; i < layers.size(); i++)
    {
      for(int j = 0; j < layers[i].size(); j++)
      {
        for(int k = 0; k < layers[i-1].size(); k++)
        {
        //  layers[i][j]->deltaW[k] = learning_rate*((layers[i][j]->Error*layers[i-1][k]->output));// + (lambda*layers[i][j]->iweights[k]));
          layers[i][j]->iweights[k] -= learning_rate*((layers[i][j]->Error*layers[i-1][k]->output));//layers[i][j]->deltaW[k];
          //layers[i][j]->bias -= learning_rate*layers[i][j]->Error;
        }
      }
    }
  }

  void clearProp()
  {
    for(int i = 1; i < layers.size(); i++)
    {
      for(int j = 0; j < layers[i].size(); j++)
      {
        for(int k = 0; k < layers[i-1].size(); k++)
        {
          layers[i][j]->Error = 0;
        }
      }
    }
  }

  void addWeightNoise()
  {
    for(int i = 1; i < layers.size(); i++)
    {
      for(int j = 0; j < layers[i].size(); j++)
      {
        normal_distribution<double> dis(0, 1/(powf(layers[i-1].size(), 0.5)));
        for(int k = 0; k < layers[i][j]->iweights.size(); k++)
        {
          layers[i][j]->iweights[k] += (dis(gen)*theta);
        //cout<<(dis(gen))<<"=>";
        }
      //  cout<<layers[i-1][j]->iweights.size()<<"=>";
      }
    }
    theta *= beta;
  }
};
