#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <sstream>


using namespace std;

int GetRandomInt(int min, int max)
{
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_int_distribution<int> int_distribution;

	int_distribution = std::uniform_int_distribution<int>(min, max);
	return int_distribution(generator);
}

float GetRandomFloat(float min, float max)
{
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_real_distribution<float> float_distribution;

	float_distribution = std::uniform_real_distribution<float>(min, max);
	return float_distribution(generator);
}

void generateFile(int number)
{
	
	fstream my_file;
	for (int i = 0; i < number; i++) {
		string name = "cornell_files/cornell" + to_string(i) + ".txt";
		my_file.open(name, ios::out);
		if (!my_file) {
			cout << "File not created!";
			break;
		}
		else {
			//write materials
			int emittance = GetRandomInt(1, 15);
			my_file << "MATERIAL 0\n";
			my_file << "RGB         1 1 1\n";
			my_file << "SPECEX      0\n";
			my_file << "SPECRGB     0 0 0\n";
			my_file << "REFL        0\n";
			my_file << "REFR        0\n";
			my_file << "REFRIOR     0\n";
			my_file << "EMITTANCE   " + to_string(emittance) + "\n";
			my_file << "\n";

			my_file << "MATERIAL 1\n";
			my_file << "RGB         .98 .98 .98\n";
			my_file << "SPECEX      0\n";
			my_file << "SPECRGB     0 0 0\n";
			my_file << "REFL        0\n";
			my_file << "REFR        0\n";
			my_file << "REFRIOR     0\n";
			my_file << "EMITTANCE   0\n";
			my_file << "\n";

			my_file << "MATERIAL 2\n";
			my_file << "RGB         .85 .35 .35\n";
			my_file << "SPECEX      0\n";
			my_file << "SPECRGB     0 0 0\n";
			my_file << "REFL        0\n";
			my_file << "REFR        0\n";
			my_file << "REFRIOR     0\n";
			my_file << "EMITTANCE   0\n";
			my_file << "\n";

			my_file << "MATERIAL 3\n";
			my_file << "RGB         .35 .85 .35\n";
			my_file << "SPECEX      0\n";
			my_file << "SPECRGB     0 0 0\n";
			my_file << "REFL        0\n";
			my_file << "REFR        0\n";
			my_file << "REFRIOR     0\n";
			my_file << "EMITTANCE   0\n";
			my_file << "\n";

			my_file << "MATERIAL 4\n";
			my_file << "RGB         1 1 1\n";
			my_file << "SPECEX      0\n";
			my_file << "SPECRGB     0 0 0\n";
			my_file << "REFL        0\n";
			my_file << "REFR        0\n";
			my_file << "REFRIOR     0\n";
			my_file << "EMITTANCE   0\n";
			my_file << "\n";

			int num_materials = GetRandomInt(6, 50);
			for (int n = 5; n < num_materials; n++) {
				float r = GetRandomFloat(0, 1);
				float g = GetRandomFloat(0, 1);
				float b = GetRandomFloat(0, 1);
				int refl = GetRandomInt(0, 1);
				int refr = GetRandomInt(0, 1);
				float refrior = refr == 0 ? 0 : GetRandomFloat(0.f, 0.25f);
				my_file << "MATERIAL " + to_string(n) + "\n";
				my_file << "RGB         " + to_string(r) + " " + to_string(g) + " " + to_string(b) + "\n";
				my_file << "SPECEX      0\n";
				my_file << "SPECRGB     0 0 0\n";
				my_file << "REFL        " + to_string(refl) + "\n";
				my_file << "REFR        " + to_string(refr) + "\n";
				my_file << "REFRIOR     " + to_string(refrior) + "\n";
				my_file << "EMITTANCE   0\n";
				my_file << "\n";
			}
			//write camera 
			my_file << "CAMERA\n";
			my_file << "RES         800 800\n";
			my_file << "FOVY        45\n";
			my_file << "ITERATIONS  5000\n";
			my_file << "DEPTH       8\n";
			my_file << "FILE        cornellNUM\n";
			my_file << "EYE         0.0 5 7.5\n";
			my_file << "LOOKAT      0 5 0\n";
			my_file << "UP          0 1 0\n";
			my_file << "FOCAL       5\n";
			my_file << "LENSE       2\n";
			my_file << "\n";

			//write ceiling light
			my_file << "OBJECT 0\n";
			my_file << "cube\n";
			my_file << "material 0\n";
			my_file << "TRANS       0 10 0\n";
			my_file << "ROTAT       0 0 0\n";
			my_file << "SCALE       3 .3 3\n";
			my_file << "\n";

			//write walls
			my_file << "OBJECT 1\n";
			my_file << "cube\n";
			my_file << "material 1\n";
			my_file << "TRANS       0 0 0\n";
			my_file << "ROTAT       0 0 0\n";
			my_file << "SCALE       10 .01 10\n";
			my_file << "\n";

			my_file << "OBJECT 2\n";
			my_file << "cube\n";
			my_file << "material 1\n";
			my_file << "TRANS       0 10 0\n";
			my_file << "ROTAT       0 0 90\n";
			my_file << "SCALE       .01 10 10\n";
			my_file << "\n";

			my_file << "OBJECT 3\n";
			my_file << "cube\n";
			my_file << "material 1\n";
			my_file << "TRANS       0 5 -5\n";
			my_file << "ROTAT       0 90 0\n";
			my_file << "SCALE       .01 10 10\n";
			my_file << "\n";

			my_file << "OBJECT 4\n";
			my_file << "cube\n";
			my_file << "material 2\n";
			my_file << "TRANS       -5 5 0\n";
			my_file << "ROTAT       0 0 0\n";
			my_file << "SCALE       .01 10 10\n";
			my_file << "\n";

			my_file << "OBJECT 5\n";
			my_file << "cube\n";
			my_file << "material 3\n";
			my_file << "TRANS       5 5 0\n";
			my_file << "ROTAT       0 0 0\n";
			my_file << "SCALE       .01 10 10\n";
			my_file << "\n";

			
			for (int t = 6; t < num_materials; t++) {

				//write objects
				int obj_type = GetRandomInt(0, 1);
				float transX = GetRandomFloat(-4, 4);
				float transY = GetRandomFloat(-4, 4);
				float transZ = GetRandomFloat(-0.5, 5);

				float rotX = GetRandomFloat(0, 90);
				float rotY = GetRandomFloat(0, 90);
				float rotZ = GetRandomFloat(0, 90);

				float scale = GetRandomFloat(0, 5.5);
				string type = obj_type == 0 ? "cube\n" : "sphere\n";

				my_file << "OBJECT " + to_string(t) + "\n";
				my_file << type;
				my_file << "material " + to_string(t) + "\n";
				my_file << "TRANS       " + to_string(transX) + " " + to_string(transY) + " " + to_string(transZ) + "\n";
				my_file << "ROTAT       " + to_string(rotX) + " " + to_string(rotY) + " " + to_string(rotZ) + "\n";;
				my_file << "SCALE       " + to_string(scale) + " " + to_string(scale) + " " + to_string(scale) + "\n";
				my_file << "\n";

			}

			my_file.close();
		}
	}
}

int main()
{
	generateFile(500);
	return 0;
}