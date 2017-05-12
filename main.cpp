#include "iostream"
#include "fstream"
#include "vector"
#include "algorithm"
#include "neuron.cpp"
#include "stdio.h"
//#include "IDX_Drivers/idx.cpp"

int main()
{
  idx_labels* lbl = new idx_labels("digits/trainlabel.bin");
  idx_img* imgs = new idx_img("digits/trainimg.bin", 60000);

  idx_labels* lbl2 = new idx_labels("digits/testlabel.bin");
  idx_img* imgs2 = new idx_img("digits/testimg.bin", 60000);


  int n = 100;
  int opt[3] = {28*28, n, 10};

  neuralNet nn(3, opt);
  vector<double> results;
  nn.clearProp();
  double ttpp = 0;
  for(int k = 0; k < 40; k++)
  {
    for(int i = 0; i < 60000; i++)
    {
      nn.input(imgs->imgs[i]);//(d_train[i]);
      int dd = nn.output(lbl->labels.values[i]);

      nn.backProp();
      nn.changeWeights();
    }
//
    learning_rate *= alpha;
  /*  lambda += ttpp;
    ttpp *= 5;*/
    double c = 0;
    for(int i = 0; i < 10000; i++)
    {
      for(int j = 0; j < 1; j++)
      {//(imgs->imgs[i]);//
        nn.input(imgs2->imgs[i]);//(d_test[i]);
        int dd = nn.output(lbl2->labels.values[i]);
        if((int)dd == (int)lbl2->labels.values[i])
        {
          cout<<dd<<"=>";
          ++c;
          break;
        }
        cout<<endl;
      }
    }
//    nn.addWeightNoise();
    c /= 100;
  //  cout<<"\n{"<<c<<" % }->";
    results.push_back(c);
    for(int i = 0; i < results.size(); i++)
    {
      cout<<"{"<<results[i]<<" % }->";
    }
  }
  /*
  for(int i = 0; i < results.size(); i++)
  {
    cout<<"{"<<results[i]<<" % }->";
  }*/
  printf("\n Tanh n = %d, SOFTMAX phi = %f rate = 0.005, lambda = %f theta = %f, beta = %f, better averaged convergence = e, backprop continuous, aplha = %f, _gamma = %f",
     n, phi, lambda, theta, beta, alpha, _gamma);
  return 0;
}
