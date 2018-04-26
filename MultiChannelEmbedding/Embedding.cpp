#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "Task.hpp"
#include <omp.h>
#include <sys/time.h>

// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	omp_set_num_threads(6);

	Model* model = nullptr;

	cout << "this is real test0!" << endl;
	if (argc == 1)
	{
		cout << "no params!" << endl;
		return 0;
	}

	/* TransG case*/
	if (atoi(argv[1]) == 1)
	{
		//creating model
		switch (atoi(argv[2]))
		{
		case 1:
			model = new TransG(FB15K_, LinkPredictionTail, report_path, atoi(argv[3]), atof(argv[4]), atof(argv[5]), atoi(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9]), atoi(argv[10]));
			break;
		case 2:
			model = new TransG(WN18_, LinkPredictionTail, report_path, atoi(argv[3]), atof(argv[4]), atof(argv[5]), atoi(argv[6]), atof(argv[7]), atof(argv[8]), atoi(argv[9]), atoi(argv[10]));
			break;
		default:
			cout << "wrong data params!" << endl;
			return 0;
			break;
		}

		//calculating training time
		struct timeval after, before;
		gettimeofday(&before, NULL);

		model->run(atoi(argv[14]));

		gettimeofday(&after, NULL);
		cout << "training training_data time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;

		//testing
		switch (atoi(argv[11]))
		{
		case 0:
			model->test(false);
			break;
		case 1:
			model->test(true, atoi(argv[12]), atoi(argv[13]), atoi(argv[15]), atoi(argv[16]), atoi(argv[17]));
			break;
		default:
			cout << "wrong test stage params!" << endl;
			return 0;
			break;
		}
	}

	/* TransE case */
	else if (atoi(argv[1]) == 2)
	{
		//creating model
		switch (atoi(argv[2]))
		{
		case 1:
			model = new TransE(FB15K_, LinkPredictionTail, report_path, atoi(argv[3]), atof(argv[4]), atof(argv[5]), atof(argv[6]), atof(argv[7]), atoi(argv[8]));
			break;
		case 2:
			model = new TransE(WN18_, LinkPredictionTail, report_path, atoi(argv[3]), atof(argv[4]), atof(argv[5]), atof(argv[6]), atof(argv[7]), atoi(argv[8]));
			break;
		default:
			cout << "wrong data params!" << endl;
			return 0;
			break;
		}

		//calculating training time
		struct timeval after, before;
		gettimeofday(&before, NULL);

		model->run(atoi(argv[12]));

		gettimeofday(&after, NULL);
		cout << "training training_data time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;

		//testing
		switch (atoi(argv[9]))
		{
		case 0:
			model->test(false);
			break;
		case 1:
			model->test(true, atoi(argv[10]), atoi(argv[11]), atoi(argv[13]), atoi(argv[14]), atoi(argv[15]));
			break;
		default:
			cout << "wrong test stage params!" << endl;
			return 0;
			break;
		}
	}
} 
