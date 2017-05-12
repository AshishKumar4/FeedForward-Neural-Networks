#include "fstream"
#include "idx.h"
//#include "stdafx.h"
#include "iostream"
#include "string.h"
#include "fstream"
#include "../main.h"


using namespace std;

void HighToLowEndian(uint32_t &d)
{
	uint32_t a;
	unsigned char *dst = (unsigned char *)&a;
	unsigned char *src = (unsigned char *)&d;

	dst[0] = src[3];
	dst[1] = src[2];
	dst[2] = src[1];
	dst[3] = src[0];

	d = a;
}

class idx_content
{
public:
	uint8_t* values;
};

class data_input
{
public:
	double* data;
};

class idx_file
{
protected:
	fstream* file;

	uint32_t magic;

public:
	uint32_t n_items;

	idx_file(char* fname)
	{
		fstream f;
		file = new fstream(fname, ios::binary | ios::in);
		if (file->is_open()) cout << "File Read" << endl;
		else cout << "File Not READ" << endl;

		file->seekg(0);

		file->read((char*)&magic, 4);

		HighToLowEndian(magic);

		file->read((char*)&n_items, 4);

		HighToLowEndian(n_items);

		cout << magic << endl << n_items;
	}

};

class idx_labels : public idx_file
{
public:

	idx_content labels;
	idx_labels(char* fname) : idx_file(fname)
	{
		labels.values = new uint8_t[n_items];

		file->read((char*)labels.values, n_items);
		cout << "\t " << fname << " File readed successfully. Number of labels: " << n_items << "\n";
	}
};

class idx_img : public idx_file
{
public:
	uint32_t rows;
	uint32_t columns;

	idx_content* imgs;
	int n_loaded;

	idx_img(char* fname, int n) : idx_file(fname)
	{
		imgs = new idx_content[n_items];

		file->read((char*)&rows, 4);
		HighToLowEndian(rows);
		file->read((char*)&columns, 4);
		HighToLowEndian(columns);

		int n_size = rows*columns;
		file->seekg(16);

		for (int i = 0; i < n; i++)
		{
			imgs[i].values = new uint8_t[n_size];
			file->read((char*)imgs[i].values, n_size);
		}
		n_loaded = n;
		cout << "\t " << fname << " File readed successfully. Number of images: " << n_items << ", Loaded images: "<<n<<"\n";
		cout << rows << " " << columns << endl;
	}

	idx_img(char* fname) : idx_file(fname)
	{
		idx_img(fname, n_items);
	}
};

void imgConverter(idx_content im, data_input* dd)
{
  int d = 0;
  for(int i = 0; i < 28*28; i++)
  {
    d += im.values[i];
  }
  double e = (double)d;
  e /= (28*28);

	dd->data = new double[28*28];
  for(int i = 0; i < 28*28; i++)
  {
    dd->data[i] = tanh((double)(((double)im.values[i] - e))/(e*_gamma));//2*(FuncNeural((double)im.values[i]/1000) - 0.5);
  }
}
