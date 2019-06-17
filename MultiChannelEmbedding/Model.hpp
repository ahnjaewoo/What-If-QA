#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include <boost/progress.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <ctime>
#include <thread>
#include <chrono>
using namespace std;
using namespace arma;

class Model
{
public:
	const DataModel&	data_model;
	const TaskType		task_type;
	const bool			be_deleted_data_model;

private:
	vector<vec> embedding_entity;
	vector<vec>	embedding_relation;
	vector<vector<vec>> embedding_clusters;
	vector<vec> weights_clusters;
	vector<int> size_clusters;

public:
	double mean;
	double hits;
	double fmean;
	double fhits;
	double rmrr;
	double fmrr;
	double total;
	vector<double> rrank;
	vector<double> frank;
	double real_hit;
	double lreal_hit;

public:
	ModelLogging&		logging;

public:
	int	epos;
    int rmean_instances[6];

public:
	vector<vector<pair<pair<int, int>, int>>> subgraph;
	vector<vector<pair<pair<int, int>, int>>> dev_subgraph;
	vector<vector<int>> cut_pos_subgraph;
	int gradient_mode;
    ofstream fout;

public:
	Model(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path)
		:data_model(*(new DataModel(dataset))), task_type(task_type),
		logging(*(new ModelLogging(logging_base_path))),
		be_deleted_data_model(true)
	{
		epos = 0;
        for (auto i = 0; i < 6; i++)
            rmean_instances[i] = 0;
		best_triplet_result = 0;
		std::cout << "Ready" << endl;

		logging.record() << "\t[Dataset]\t" << dataset.name;
		logging.record() << TaskTypeName(task_type);
	}

	Model(const Dataset& dataset,
		const string& file_zero_shot,
		const TaskType& task_type,
		const string& logging_base_path)
		:data_model(*(new DataModel(dataset))), task_type(task_type),
		logging(*(new ModelLogging(logging_base_path))),
		be_deleted_data_model(true)
	{
		epos = 0;
		best_triplet_result = 0;
		std::cout << "Ready" << endl;

		logging.record() << "\t[Dataset]\t" << dataset.name;
		logging.record() << TaskTypeName(task_type);
	}

	Model(const DataModel* data_model,
		const TaskType& task_type,
		ModelLogging* logging)
		:data_model(*data_model), logging(*logging), task_type(task_type),
		be_deleted_data_model(false)
	{
		epos = 0;
		best_triplet_result = 0;
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) = 0;
	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) = 0;
	virtual double prob_triplets_subgraph(const pair<pair<int, int>, int>& triplet, vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s, vector<int>& size_clusters_s) = 0;
	virtual void deep_copy_for_subgraph(vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s, vector<vector<vec>>& embedding_clusters_s,
		vector<vec>& weights_clusters_s, vector<int>& size_clusters_s) = 0;
	virtual void get_delta_unit(vector<vec>& embedding_relation_s, vector<vector<vec>>& embedding_clusters_s, vector<int>& size_clusters_s) = 0;
	virtual void train_triplet_subgraph(const pair<pair<int, int>, int>& triplet, vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s, vector<int>& size_clusters_s,
		vector<pair<pair<int, int>, int>> subgraph) = 0;
	virtual void train_triplet_subgraph_BM(const pair<pair<int, int>, int>& triplet, vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s, vector<int>& size_clusters_s,
		vector<pair<pair<int, int>, int>> subgraph, vector<int> cut_pos) = 0;


public:
	virtual void train(bool last_time = false)
	{
		++epos;

#pragma omp parallel for
		for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
		{
			train_triplet(*i);
		}
	}

	virtual void train_and_test_subgraph(int total_epos)
	{
		//parameter initialization
		mean = 0;
		hits = 0;
		fmean = 0;
		fhits = 0;
		rmrr = 0;
		fmrr = 0;
		real_hit = 0;
		lreal_hit = 0;
		total = data_model.data_test_true.size();
		rrank.resize(data_model.data_test_true.size());
		frank.resize(data_model.data_test_true.size());
        
        
        // time_t t = time(0);   // get time now
        // struct tm * now = localtime( & t  );
        // stringstream ss;

        // ss << "subgraph_" << (now->tm_year + 1900) << '_' << (now->tm_mon + 1) << '_' <<  now->tm_mday << '_'  <<  now->tm_hour << '_' <<  now->tm_min << '_' <<  now->tm_sec << ".csv";
        // fout.open(ss.str());
        // fout << "# of condition triples, # of subgraph triples, rmean, frmean, real hit, time(s), subgraph training iteration" << endl;

		cout << "\nTraining subgraph & testing test-set.." << endl;
		boost::progress_display	cons_bar(data_model.data_test_true.size());

		if (gradient_mode == 0)
		{
			for (auto i = 0; i < data_model.data_test_true.size(); i++)
			{
				++cons_bar;
                struct timeval after_single_query, before_single_query;
                gettimeofday(&before_single_query, NULL);

				//deep copy
				deep_copy_for_subgraph(embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters);

				//train
				for (auto tot = 0; tot < total_epos; tot++)
				{
#pragma omp parallel for
					for (auto j = subgraph[i].begin(); j != subgraph[i].end(); j++)
					{
						train_triplet_subgraph(*j, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters, subgraph[i]);
					}
				}

				//test : link prediction
				test_link_prediction_subgraph(i);

				//test : triplet classification
				test_triplet_classification_subgraph(i);
                
                gettimeofday(&after_single_query, NULL);
                // fout <<  after_single_query.tv_sec + after_single_query.tv_usec/1000000.0 - before_single_query.tv_sec - before_single_query.tv_usec/1000000.0 << ", " << total_epos << endl;
			}

            // for (auto j=0; j<6; j++)
            // {
            //     cout << "number of instances whose rmean = " << j << "is: " << rmean_instances[j] << endl;
            // }

			print_final_test_link_prediction_subgraph();
			print_final_test_triplet_classification_subgraph();
		}
		else if (gradient_mode == 1)
		{
			get_delta_unit(embedding_relation, embedding_clusters, size_clusters);
            srand((unsigned int)time(0));
			for (auto i = 0; i < data_model.data_test_true.size(); i++)
			{
				++cons_bar;
                struct timeval after_single_query, before_single_query;
                gettimeofday(&before_single_query, NULL);

				//deep copy
				deep_copy_for_subgraph(embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters);
				
                //train
                double ratio = double(subgraph[i].size()) / dev_subgraph[i].size();
                if (ratio >= 1.0) ratio = 1.0;
				for (auto tot = 0; tot < total_epos; tot++)
				{
#pragma omp parallel for
					for (auto j = subgraph[i].begin(); j != subgraph[i].end(); j++)
					{
						train_triplet_subgraph(*j, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters, subgraph[i]);
					}
#pragma omp parallel for
					for (auto j = dev_subgraph[i].begin(); j != dev_subgraph[i].end(); j++)
					{
						if (double(rand() / RAND_MAX) > ratio) continue;
						train_triplet_subgraph_BM(*j, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters, subgraph[i], cut_pos_subgraph[i]);
					}
				}

				//test : link prediction
				test_link_prediction_subgraph(i);

				//test : triplet classification
				test_triplet_classification_subgraph(i);
			    
                gettimeofday(&after_single_query, NULL);
                // fout <<  after_single_query.tv_sec + after_single_query.tv_usec/1000000.0 - before_single_query.tv_sec - before_single_query.tv_usec/1000000.0 << ", " << total_epos << endl;
			}

			print_final_test_link_prediction_subgraph();
			print_final_test_triplet_classification_subgraph();
		}
		else
		{
			cout << "mode error!" << endl;
			return;
		}

	}

	void run(int total_epos)
	{
		logging.record() << "\t[Epos]\t" << total_epos;

		--total_epos;
		cout << "Training train-set.." << endl;
		boost::progress_display	cons_bar(total_epos);
		while (total_epos-- > 0)
		{
			++cons_bar;
			train();

			if (task_type == TripletClassification)
				test_triplet_classification();
		}

		train(true);
	}

public:
	double		best_triplet_result;
	double		best_link_mean;
	double		best_link_hitatten;
	double		best_link_deviation;
	double		best_link_fmean;
	double		best_link_fhitatten;
	double		best_link_fdeviation;

	void reset()
	{
		best_triplet_result = 0;
		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_deviation = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;
		best_link_fdeviation = 0;
	}

	void initialize_cut(set<string> A, vector<int>& cut)
	{
		for (int i = 0; i < cut.size(); i++)
		{
			int count_cut = 0;
			for (auto a : A)
			{
				int e1 = data_model.entity_name_to_id.at(a);
				int e2 = i;

				if (data_model.rel_finder.find(make_pair(e1, e2)) != data_model.rel_finder.end() ||
					data_model.rel_finder.find(make_pair(e2, e1)) != data_model.rel_finder.end())
					count_cut++;
			}
			cut[i] = count_cut;
		}
	}

	void update_cut(set<string> diff, vector<int>& cut, string best_vertex)
	{
		for (string v : diff)
		{
			if (v != best_vertex) continue;

			int e1 = data_model.entity_name_to_id.at(v);
			int e2 = data_model.entity_name_to_id.at(best_vertex);
			if (data_model.rel_finder.find(make_pair(e1, e2)) != data_model.rel_finder.end() ||
				data_model.rel_finder.find(make_pair(e2, e1)) != data_model.rel_finder.end())
				cut[e1] += 1;
			break;
		}
	}

	void bipartite(int budget, int q_in_subgraph = 1, int method = 0)
	{
		vector<int> b(data_model.data_test_true.size());

		cout << "Bipartition and construction of subgraph.." << endl;

		if (method == 1)
		{
			boost::progress_display	cons_bar(data_model.data_test_true.size());
#pragma omp parallel for
			for (auto i = 0; i < data_model.data_test_true.size(); ++i)
			{
				++cons_bar;

				set<string> A;
				if (q_in_subgraph)
				{
					A.insert(data_model.entity_id_to_name[data_model.data_test_true[i].first.first]);
					A.insert(data_model.entity_id_to_name[data_model.data_test_true[i].first.second]);
				}

				for (auto j = 0; j < data_model.data_condition[i].size(); ++j)
				{
					A.insert(data_model.entity_id_to_name[data_model.data_condition[i][j].first.first]);
					A.insert(data_model.entity_id_to_name[data_model.data_condition[i][j].first.second]);
				}
				b[i] = budget;
				set<string> V = data_model.set_entity;
				cut_pos_subgraph[i].resize(data_model.set_entity.size(), 0);
				initialize_cut(A, cut_pos_subgraph[i]);
				while (b[i]--)
				{
					int best_scr = numeric_limits<int>::max();
					string best_vertex;
					//update vertex set V
					set<string> diff;
					set_difference(V.begin(), V.end(), A.begin(), A.end(), inserter(diff, diff.begin()));
					for (string v : diff) {
						if (data_model.count_entity.find(v) == data_model.count_entity.end()) continue;
						if (-data_model.count_entity.at(v) >= best_scr) continue;
						int scr = -cut_pos_subgraph[i][data_model.entity_name_to_id.at(v)];
						if (scr < best_scr)
						{
							best_scr = scr;
							best_vertex = v;
						}
					}
					if (best_scr >= 0) break;
					A.insert(best_vertex);

					update_cut(diff, cut_pos_subgraph[i], best_vertex);
				}

				//constructing subgraph + condition
				for (auto j : A)
				{
					for (auto k : A)
					{
						if (j == k) continue;
						int e1 = data_model.entity_name_to_id.at(j);
						int e2 = data_model.entity_name_to_id.at(k);
                        if (data_model.rel_finder.find(make_pair(e1, e2)) != data_model.rel_finder.end()) {
                            subgraph[i].push_back(make_pair(make_pair(e1, e2), data_model.rel_finder.at(make_pair(e1, e2))));
                            // codes for use case: constructing sub graph without crucial clue triples in it
                            // if ((e1 != data_model.data_test_true[i].first.second || e2 != data_model.data_test_true[i].first.first) && (e1 != data_model.data_test_true[i].first.first || e2 != data_model.data_test_true[i].first.second))
                            // {   
                            //     subgraph[i].push_back(make_pair(make_pair(e1, e2), data_model.rel_finder.at(make_pair(e1, e2))));
                            // }
                            // else {
                            //     cout << "oh my god there's a crucial clue in subgraph [" << i << "]" << endl;
                            // }
                        }
					}
				}
				for (auto j = 0; j < data_model.data_condition[i].size(); ++j)
				{
					subgraph[i].push_back(data_model.data_condition[i][j]);	//containing conditional part C
				}

                // printing subgraph size and triples
                if (0)
                // if (i == 0 || i == 4) 
                {
                    cout << "subgraph [" << i << "]: num of triples: " << subgraph[i].size() << endl;
                    for (auto j : subgraph[i])
                    {
                        cout << data_model.entity_id_to_name[j.first.first] << ", " << data_model.relation_id_to_name[j.second] << ", " << data_model.entity_id_to_name[j.first.second] << endl;
                    }
                }

				//constructing cutting edge subgraph
				if (gradient_mode == 1) {
					set<string> diff;
					set_difference(data_model.set_entity.begin(), data_model.set_entity.end(), A.begin(), A.end(), inserter(diff, diff.begin()));
					for (auto t : data_model.data_train)
					{
						int e1 = t.first.first;
						int e2 = t.first.second;
						string st1 = data_model.entity_id_to_name[e1];
						string st2 = data_model.entity_id_to_name[e2];
						if (A.find(st1) != A.end() && diff.find(st2) != diff.end())
							dev_subgraph[i].push_back(t);
						if (A.find(st2) != A.end() && diff.find(st1) != diff.end())
							dev_subgraph[i].push_back(t);
					}
				}
			}
		}
		else if (method == 0)
		{
			srand((unsigned int)time(0));
			boost::progress_display	cons_bar(data_model.data_test_true.size());
#pragma omp parallel for
			for (auto i = 0; i < data_model.data_test_true.size(); ++i)
			{
				++cons_bar;

				set<string> A;
				if (q_in_subgraph)
				{
					A.insert(data_model.entity_id_to_name[data_model.data_test_true[i].first.first]);
					A.insert(data_model.entity_id_to_name[data_model.data_test_true[i].first.second]);
				}

				for (auto j = 0; j < data_model.data_condition[i].size(); ++j)
				{
					A.insert(data_model.entity_id_to_name[data_model.data_condition[i][j].first.first]);
					A.insert(data_model.entity_id_to_name[data_model.data_condition[i][j].first.second]);
				}

				cut_pos_subgraph[i].resize(data_model.set_entity.size(), 0);
				while (A.size() < budget)
				{
					//insert entity
					A.insert(data_model.entity_id_to_name[rand() % data_model.set_entity.size()]);
				}

				//constructing subgraph + condition
				for (auto j : A)
				{
					for (auto k : A)
					{
						if (j == k) continue;
						int e1 = data_model.entity_name_to_id.at(j);
						int e2 = data_model.entity_name_to_id.at(k);
						if (data_model.rel_finder.find(make_pair(e1, e2)) != data_model.rel_finder.end())
							subgraph[i].push_back(make_pair(make_pair(e1, e2), data_model.rel_finder.at(make_pair(e1, e2))));
					}
				}
				for (auto j = 0; j < data_model.data_condition[i].size(); ++j)
				{
					subgraph[i].push_back(data_model.data_condition[i][j]);	//containing conditional part C
				}

				//constructing cutting edge subgraph
				if (gradient_mode == 1) {
					set<string> diff;
					set_difference(data_model.set_entity.begin(), data_model.set_entity.end(), A.begin(), A.end(), inserter(diff, diff.begin()));
					for (auto t : data_model.data_train)
					{
						int e1 = t.first.first;
						int e2 = t.first.second;
						string st1 = data_model.entity_id_to_name[e1];
						string st2 = data_model.entity_id_to_name[e2];
						if (A.find(st1) != A.end() && diff.find(st2) != diff.end())
							dev_subgraph[i].push_back(t);
						if (A.find(st2) != A.end() && diff.find(st1) != diff.end())
							dev_subgraph[i].push_back(t);
					}
				}
			}
		}
		else
		{
			cout << "bipartite method error!" << endl;
			return;
		}

	}

	void test(bool subgraph_task = true, int mode = 0, int budget = 10, int subgraph_epos = 500, int q_in_subgraph = 1, int bipartite_method = 0, int hit_rank = 10)
	{
        logging.record() << "\t[Budget]\t" << budget;
        logging.record() << "\t[Subgraph Epos]\t" << subgraph_epos;

		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_deviation = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;
		best_link_fdeviation = 0;

		if (subgraph_task) {

			struct timeval after, before;
			gettimeofday(&before, NULL);

			//0. initialization
			initialize_subgraph(mode);

			//1. condition vector들과 관련된 training의 triple들 떼어내기 (subgraph)
			// make a sub-graph containing verices and edges of conditions
			bipartite(budget, q_in_subgraph, bipartite_method);

			//2. subgraph + condition 학습
			//triplet learning
            
            // writing num of condition & subgraph triples in csv file
            // ofstream file;
            // file.open("subgraph.csv");
            // file << "# of condition triples, # of subgraph triples,\n";
            // for (auto i=0; i<subgraph.size();i++)
            // {
            //     // cout << "# of condition triples: " << data_model.data_condition[i].size() << endl; 
            //     // cout << "# of subgraph triples: " << subgraph[i].size() << endl;

            //     file << data_model.data_condition[i].size() << ", " << subgraph[i].size() << endl;
            // }

			//3 - 1. 그리고 밑에 있는 test task 수행
			//test_link_prediction per test set
			train_and_test_subgraph(subgraph_epos);

			gettimeofday(&after, NULL);
            logging.record() << "testing subgraph test_data time: " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds";
            cout << "testing subgraph test_data time: " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            logging.record() << "End of the test";
		}
		else {

			struct timeval after, before;
                        gettimeofday(&before, NULL);

			//3 - 2. 그리고 밑에 있는 test task 수행
			//test_link_prediction per test set
			if (task_type == LinkPredictionHead || task_type == LinkPredictionTail || task_type == LinkPredictionRelation)
				test_link_prediction(hit_rank);
			if (task_type == LinkPredictionHeadZeroShot || task_type == LinkPredictionTailZeroShot || task_type == LinkPredictionRelationZeroShot)
				test_link_prediction_zeroshot(hit_rank);
			else
				test_triplet_classification();

			gettimeofday(&after, NULL);
			cout << "testing non subgraph test_data time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;
		}
	}

	void initialize_subgraph(int mode)
	{
		gradient_mode = mode;
		subgraph.resize(data_model.data_test_true.size());
		dev_subgraph.resize(data_model.data_test_true.size());
		cut_pos_subgraph.resize(data_model.data_test_true.size());
	}

public:
	void test_triplet_classification()
	{
		real_hit = 0;
		lreal_hit = 0;
		for (auto r = 0; r < data_model.set_relation.size(); ++r)
		{
			vector<pair<double, bool>>	threshold_dev;
			for (auto i = data_model.data_dev_true.begin(); i != data_model.data_dev_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), true));
			}
			for (auto i = data_model.data_dev_false.begin(); i != data_model.data_dev_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), false));
			}

			sort(threshold_dev.begin(), threshold_dev.end());

			double threshold;
			double vari_mark = 0;
			int tot = 0;
			int hit = 0;
			for (auto i = threshold_dev.begin(); i != threshold_dev.end(); ++i)
			{
				if (i->second == false)
					++hit;
				++tot;

				if (vari_mark <= 2 * hit - tot + data_model.data_dev_true.size())
				{
					vari_mark = 2 * hit - tot + data_model.data_dev_true.size();
					threshold = i->first;
				}
			}


			double lreal_total = 0;
			for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				++lreal_total;
				if (prob_triplets(*i) > threshold)
					++real_hit, ++lreal_hit;
			}

			for (auto i = data_model.data_test_false.begin(); i != data_model.data_test_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				++lreal_total;
				if (prob_triplets(*i) <= threshold)
					++real_hit, ++lreal_hit;
			}
		}

		std::cout << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size());
		best_triplet_result = max(
			best_triplet_result,
			real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size()));
		std::cout << ", Best = " << best_triplet_result << endl;

		logging.record() << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size())
			<< ", Best = " << best_triplet_result;

		std::cout.flush();
	}

	void test_triplet_classification_subgraph(int idx)
	{
		pair<pair<int, int>, int> triplet = data_model.data_test_true[idx];
		pair<pair<int, int>, int> triplet_f = data_model.data_test_true[idx];
		getNegativeTriplet(triplet_f, idx);

		int r = triplet.second;
		vector<pair<double, bool>> threshold_dev;
		for (auto i = data_model.data_dev_true.begin(); i != data_model.data_dev_true.end(); ++i)
		{
			if (i->second != r)
				continue;
			threshold_dev.push_back(make_pair(prob_triplets_subgraph(*i, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters), true));
		}

		for (auto i = data_model.data_dev_false.begin(); i != data_model.data_dev_false.end(); ++i)
		{
			if (i->second != r)
				continue;
			threshold_dev.push_back(make_pair(prob_triplets_subgraph(*i, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters), false));
		}

		sort(threshold_dev.begin(), threshold_dev.end());

		double threshold;
		double vari_mark = 0;
		int tot = 0;
		int hit = 0;
		for (auto i = threshold_dev.begin(); i != threshold_dev.end(); ++i)
		{
			if (i->second == false)
				++hit;
			++tot;
			if (vari_mark <= 2 * hit - tot + data_model.data_dev_true.size())
			{
				vari_mark = 2 * hit - tot + data_model.data_dev_true.size();
				threshold = i->first;
			}
		}
        double original_real_hit = real_hit;
		if (prob_triplets_subgraph(triplet, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters) > threshold)
			++real_hit, ++lreal_hit;

		if (prob_triplets_subgraph(triplet_f, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters) <= threshold)
			++real_hit, ++lreal_hit;
        // fout << real_hit - original_real_hit << ", ";
	}

	void print_final_test_triplet_classification_subgraph()
	{
		int tot = data_model.data_test_true.size();
		std::cout << epos << "\t Accuracy = "
			<< real_hit / (2 * tot);
		best_triplet_result = max(
			best_triplet_result,
			real_hit / (2 * tot));
		std::cout << ", Best = " << best_triplet_result << endl;

		logging.record() << epos << "\t Accuracy = "
			<< real_hit / (2 * tot)
			<< ", Best = " << best_triplet_result;

		std::cout.flush();
	}
	void test_link_prediction(int hit_rank = 10, const int part = 0)
	{
		//parameter initialization
		mean = 0;
		hits = 0;
		fmean = 0;
		fhits = 0;
		rmrr = 0;
		fmrr = 0;
		total = data_model.data_test_true.size();
		rrank.resize(data_model.data_test_true.size());
		frank.resize(data_model.data_test_true.size());

		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };

		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++arr_total[data_model.relation_type[i->second]];
		}

		int cnt = 0;
		deep_copy_for_subgraph(embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters);

		cout << "Testing query-test-set.." << endl;
		boost::progress_display cons_bar(data_model.data_test_true.size() / 100);

//#pragma omp parallel for
		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++cnt;
			if (cnt % 100 == 0)
			{
				++cons_bar;
			}

			pair<pair<int, int>, int> t = *i;
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*i);
			int head = t.first.first;
			int relation = t.second;
			int tail = t.first.second;

			if (task_type == LinkPredictionRelation || part == 2)
			{
				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{
					t.second = j;

					if (score_i >= prob_triplets(t))
						continue;
                    
					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
						++frmean;
				}
			}
			else
			{
				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{
					if (task_type == LinkPredictionHead || part == 1)
						t.first.first = j;
					else
						t.first.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;
                    // cout << data_model.entity_id_to_name[j] << endl;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
						//if (frmean > hit_rank)
						//	break;
					}
				}
			}

#pragma omp critical
			{
				if (frmean < hit_rank)
					++arr_mean[data_model.relation_type[i->second]];

				mean += rmean;
				fmean += frmean;
				rmrr += 1.0 / (rmean + 1);
				fmrr += 1.0 / (frmean + 1);
				int idx = distance(data_model.data_test_true.begin(), i);
				rrank[idx] = rmean;
				frank[idx] = frmean;

				if (rmean < hit_rank)
					++hits;
				if (frmean < hit_rank)
					++fhits;
			}
			/* CODES IN ORDER TO ACHIEVE SPECIFIC RESULTS COMPARING PARTIALLY UPDATED METHOD'S THOSE WITH BASELINE'S THOSE*/
			// cout << "===============================================================================" << endl;
	        //        cout << cnt << ": '" << data_model.entity_id_to_name[head] << " " << data_model.relation_id_to_name[relation] << " " << data_model.entity_id_to_name[tail] << "'- rmean: " << rmean <<", frmean: " << frmean << endl;
        	        /*embedding_entity[head].print("head: ");
                	embedding_entity[tail].print("tail: ");
	                embedding_relation[relation].print("relation: ");*/
			//vec error = embedding_entity[head] + embedding_relation[relation] - embedding_entity[tail];
	                //cout << "baseline energy: " << sum(abs(error)) << endl;
        	        //cout << "===============================================================================" << endl;
		}

		std::cout << endl;
		for (auto i = 1; i <= 4; ++i)
		{
			std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
			logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		}
		logging.record();

		print_final_test_link_prediction_subgraph();
	}

	void getNegativeTriplet(pair<pair<int, int>, int>& triplet, int i)
	{
		srand((unsigned int)time(0));
		int selectHead = rand() % 2;

		set<int> A;
		for (auto t : subgraph[i]) {
			A.insert(t.first.first);
			A.insert(t.first.second);
		}
		for (auto v : A) {
			if (selectHead) triplet.first.first = v;
			else triplet.first.second = v;
			bool uniq = true;
			for (auto t : subgraph[i]) {
				if (triplet == t) {
					uniq = false;
					break;
				}
			}
			if (!uniq) continue;
			if (triplet == data_model.data_test_true[i]) continue;
			if (triplet.first.first == triplet.first.second) continue;
			break;
		}
	}

	void test_link_prediction_subgraph(int idx, int hit_rank = 10, const int part = 0)
	{
		pair<pair<int, int>, int> t = data_model.data_test_true[idx];

		int frmean = 0;
		int rmean = 0;
		double score_i = prob_triplets_subgraph(t, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters);
		int head = t.first.first;
		int relation = t.second;
		int tail = t.first.second;

		if (task_type == LinkPredictionRelation || part == 2)
		{
			for (auto j = 0; j != data_model.set_relation.size(); ++j)
			{
				t.second = j;

				if (score_i >= prob_triplets_subgraph(t, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters))
					continue;

				++rmean;

				if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					++frmean;
			}
		}
		else
		{
			for (auto j = 0; j != data_model.set_entity.size(); ++j)
			{
				if (task_type == LinkPredictionHead || part == 1)
					t.first.first = j;
				else
					t.first.second = j;

				if (score_i >= prob_triplets_subgraph(t, embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters))
					continue;

				++rmean;

				if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
				{
					++frmean;
					//if (frmean > hit_rank)
					//	break;
				}
			}
		}

        // fout << data_model.data_condition[idx].size() << ", " << subgraph[idx].size() << ", " << rmean << ", " << frmean << ", ";
        // finding subgraph instances whose rank is smaller 5th rank
        // if (rmean <= 5)
        // {
        //      rmean_instances[rmean] += 1;
        // }
        
        if (0)
        {
            cout << "False" << endl;
		    // ofstream fout("./result.csv");
		    // /* CODES IN ORDER TO ACHIEVE SPECIFIC RESULTS COMPARING PARTIALLY UPDATED METHOD'S THOSE WITH BASELINE'S THOSE*/
		    // cout << "===============================================================================" << endl;
		    // cout << "'" << data_model.entity_id_to_name[head] << " " << data_model.relation_id_to_name[relation] << " " << data_model.entity_id_to_name[tail] << "'- rmean: " << rmean <<", frmean: " << frmean  << endl;
		    // fout << "'" << data_model.entity_id_to_name[head] << " " << data_model.relation_id_to_name[relation] << " " << data_model.entity_id_to_name[tail] << "'- rmean: " << rmean <<", frmean: " << frmean << endl; 
            // fout << "AFTER PARTIAL EMBEDDING(QUESTION)" << endl;
		    // fout << "head, ";
		    // for (int i=0; i < embedding_entity[head].size(); i++)
            //     fout << embedding_entity[head](i) << ", ";
		    // fout << "\n" << "relation, ";
		    // for (int i=0; i < embedding_relation[relation].size(); i++)
            //     fout << embedding_relation[relation](i) << ", ";
		    // fout << "\n" << "tail, ";
		    // for (int i=0; i < embedding_entity[tail].size(); i++) 
            //     fout << embedding_entity[tail](i) << ", ";
		    // fout << endl;
		    // // embedding_entity[head].print("head: ");
		    // // embedding_entity[tail].print("tail: ");
		    // // embedding_relation[relation].print("relation: ");
		    // vec error = embedding_entity[head] + embedding_relation[relation] - embedding_entity[tail];
		    // cout << "partial embedding query energy: " << sum(abs(error)) << endl;
		    // fout << "partial embedding query energy: " << sum(abs(error)) << endl;

		    // // partial embedding condition energies
		    // fout << "AFTER PARTIAL EMBEDDING(CONDITIONS)" << endl;
		    // for (auto &v : data_model.data_condition[idx]) {
            //     fout << "head, ";
		    // 	for (int i=0; i < embedding_entity[v.first.first].size(); i++) fout << embedding_entity[v.first.first](i) << ", ";
            //     fout << "\n" << "relation, ";
            //     for (int i=0; i < embedding_relation[v.second].size(); i++) fout << embedding_relation[v.second](i) << ", ";
            //     fout << "\n" << "tail, ";
            //     for (int i=0; i < embedding_entity[v.first.second].size(); i++) fout << embedding_entity[v.first.second](i) << ", ";
            //     fout << endl;
            //     cout << "'" << data_model.entity_id_to_name[v.first.first] << " " << data_model.relation_id_to_name[v.second] << " " << data_model.entity_id_to_name[v.first.second] << "'" << endl;
            //     fout << "'" << data_model.entity_id_to_name[v.first.first] << " " << data_model.relation_id_to_name[v.second] << " " << data_model.entity_id_to_name[v.first.second] << "'" << endl;
            //     error = embedding_entity[v.first.first] + embedding_relation[v.second] - embedding_entity[v.first.second];
            //     cout << "partial embedding condition energy: " << sum(abs(error)) << endl;
            //     fout << "partial embedding condition energy: " << sum(abs(error)) << endl;
            // }

		    // // partial embedding subgraph energies
            // fout << "AFTER PARTIAL EMBEDDING(SUBGRAPH)" << endl;
            //     for (auto &v : subgraph[idx]) {
            //     fout << "head, ";
            //     for (int i=0; i < embedding_entity[v.first.first].size(); i++) fout << embedding_entity[v.first.first](i) << ", ";
            //     fout << "\n" << "relation, ";
            //     for (int i=0; i < embedding_relation[v.second].size(); i++) fout << embedding_relation[v.second](i) << ", ";
            //     fout << "\n" << "tail, ";
            //     for (int i=0; i < embedding_entity[v.first.second].size(); i++) fout << embedding_entity[v.first.second](i) << ", ";
            //     fout << endl;
            //     // cout << "'" << data_model.entity_id_to_name[v.first.first] << " " << data_model.relation_id_to_name[v.second] << " " << data_model.entity_id_to_name[v.first.second] << "'" << endl;
            //     fout << "'" << data_model.entity_id_to_name[v.first.first] << " " << data_model.relation_id_to_name[v.second] << " " << data_model.entity_id_to_name[v.first.second] << "'" << endl;
            //     error = embedding_entity[v.first.first] + embedding_relation[v.second] - embedding_entity[v.first.second];
            //     // cout << "partial embedding subgraph energy: " << sum(abs(error)) << endl;
            //     fout << "partial embedding subgraph energy: " << sum(abs(error)) << endl;
            // }

		    // deep_copy_for_subgraph(embedding_entity, embedding_relation, embedding_clusters, weights_clusters, size_clusters);
		    // fout << "BEFORE PARTIAL EMBEDDING(QUESITON)" << endl;
            // fout <<  "head, ";
            // for (int i=0; i < embedding_entity[head].size(); i++)
            //     fout << embedding_entity[head](i) << ", ";
            // fout << "\n" << "relation, ";
            // for (int i=0; i < embedding_relation[relation].size(); i++)
            //     fout << embedding_relation[relation](i) << ", ";
            // fout << "\n" << "tail, ";
            // for (int i=0; i < embedding_entity[tail].size(); i++)
            //     fout << embedding_entity[tail](i) << ", ";
            // fout << endl;
            // // embedding_entity[head].print("head: ");
            // // embedding_entity[tail].print("tail: ");
            // // embedding_relation[relation].print("relation: ");
		    // error = embedding_entity[head] + embedding_relation[relation] - embedding_entity[tail];
            // cout << "baseline query energy: " << sum(abs(error)) << endl;
		    // fout << "baseline query energy: " << sum(abs(error)) << endl;
		    // 
		    // fout << "BEFORE PARTIAL EMBEDDING(CONDITIONS)" << endl;
		    // for (auto &v : data_model.data_condition[idx]) {
            //     fout <<  "head, ";
            //     for (int i=0; i < embedding_entity[v.first.first].size(); i++) fout << embedding_entity[v.first.first](i) << ", ";
            //     fout << "\n" << "relation, ";
            //     for (int i=0; i < embedding_relation[v.second].size(); i++) fout << embedding_relation[v.second](i) << ", ";
            //     fout << "\n" << "tail, ";
            //     for (int i=0; i < embedding_entity[v.first.second].size(); i++) fout << embedding_entity[v.first.second](i) << ", ";
            //     fout << endl;
            //     cout << "'" << data_model.entity_id_to_name[v.first.first] << " " << data_model.relation_id_to_name[v.second] << " " << data_model.entity_id_to_name[v.first.second] << "'" << endl;
            //     fout << "'" << data_model.entity_id_to_name[v.first.first] << " " << data_model.relation_id_to_name[v.second] << " " << data_model.entity_id_to_name[v.first.second] << "'" << endl;
            //     error = embedding_entity[v.first.first] + embedding_relation[v.second] - embedding_entity[v.first.second];
            //     cout << "baseline condition energy: " << sum(abs(error)) << endl;
            //     fout << "baseline condition energy: " << sum(abs(error)) << endl;
            // }

		    // fout << "BEFORE PARTIAL EMBEDDING(SUBGRAPH)" << endl;
            // for (auto &v : subgraph[idx]) {
            //     fout <<  "head, ";
            //     for (int i=0; i < embedding_entity[v.first.first].size(); i++) fout << embedding_entity[v.first.first](i) << ", ";
            //     fout << "\n" << "relation, ";
            //     for (int i=0; i < embedding_relation[v.second].size(); i++) fout << embedding_relation[v.second](i) << ", ";
            //     fout << "\n" << "tail, ";
            //     for (int i=0; i < embedding_entity[v.first.second].size(); i++) fout << embedding_entity[v.first.second](i) << ", ";
            //     fout << endl;
            //     // cout << "'" << data_model.entity_id_to_name[v.first.first] << " " << data_model.relation_id_to_name[v.second] << " " << data_model.entity_id_to_name[v.first.second] << "'" << endl;
            //     fout << "'" << data_model.entity_id_to_name[v.first.first] << " " << data_model.relation_id_to_name[v.second] << " " << data_model.entity_id_to_name[v.first.second] << "'" << endl;
            //     error = embedding_entity[v.first.first] + embedding_relation[v.second] - embedding_entity[v.first.second];
            //     // cout << "baseline subgraph energy: " << sum(abs(error)) << endl;
            //     fout << "baseline subgraph energy: " << sum(abs(error)) << endl;
            // }
		    // fout.close();
		    // cout << "===============================================================================" << endl;
        }
#pragma omp critical
		{
			mean += rmean;
			fmean += frmean;
			rmrr += 1.0 / (rmean + 1);
			fmrr += 1.0 / (frmean + 1);
			rrank[idx] = rmean;
			frank[idx] = frmean;

			if (rmean < hit_rank)
				++hits;
			if (frmean < hit_rank)
				++fhits;
		}
	}

	void test_link_prediction_zeroshot(int hit_rank = 10, const int part = 0)
	{
		reset();

		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double total = data_model.data_test_true.size();

		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };

		cout << endl;

		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			if (i->first.first >= data_model.zeroshot_pointer
				&& i->first.second >= data_model.zeroshot_pointer)
			{
				++arr_total[3];
			}
			else if (i->first.first < data_model.zeroshot_pointer
				&& i->first.second >= data_model.zeroshot_pointer)
			{
				++arr_total[2];
			}
			else if (i->first.first >= data_model.zeroshot_pointer
				&& i->first.second < data_model.zeroshot_pointer)
			{
				++arr_total[1];
			}
			else
			{
				++arr_total[0];
			}
		}

		cout << "0 holds " << arr_total[0] << endl;
		cout << "1 holds " << arr_total[1] << endl;
		cout << "2 holds " << arr_total[2] << endl;
		cout << "3 holds " << arr_total[3] << endl;

		int cnt = 0;

#pragma omp parallel for
		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++cnt;
			if (cnt % 100 == 0)
			{
				std::cout << cnt << ',';
				std::cout.flush();
			}

			pair<pair<int, int>, int> t = *i;
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*i);

			if (task_type == LinkPredictionRelationZeroShot || part == 2)
			{
				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{
					t.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
						++frmean;
				}
			}
			else
			{
				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{
					if (task_type == LinkPredictionHeadZeroShot || part == 1)
						t.first.first = j;
					else
						t.first.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
					}
				}
			}

#pragma omp critical
			{
				if (frmean < hit_rank)
				{
					if (i->first.first >= data_model.zeroshot_pointer
						&& i->first.second >= data_model.zeroshot_pointer)
					{
						++arr_mean[3];
					}
					else if (i->first.first < data_model.zeroshot_pointer
						&& i->first.second >= data_model.zeroshot_pointer)
					{
						++arr_mean[2];
					}
					else if (i->first.first >= data_model.zeroshot_pointer
						&& i->first.second < data_model.zeroshot_pointer)
					{
						++arr_mean[1];
					}
					else
					{
						++arr_mean[0];
					}
				}

				mean += rmean;
				fmean += frmean;
				if (rmean < hit_rank)
					++hits;
				if (frmean < hit_rank)
					++fhits;
			}
		}

		std::cout << endl;
		for (auto i = 0; i < 4; ++i)
		{
			std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
			logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		}
		logging.record();

		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;
		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;
	}

	void print_final_test_link_prediction_subgraph()
	{
		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_deviation = getDeviation(rrank, best_link_mean);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);
		best_link_fdeviation = getDeviation(frank, best_link_fmean);


		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestMRR = " << rmrr / total << endl;
		std::cout << "Raw.BestHARMONICMEANS = " << total / rmrr << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		std::cout << "Raw.BestDEVIATIONS = " << best_link_deviation << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestMRR = " << total / rmrr;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;
		logging.record() << "Raw.BestDEVIATIONS = " << best_link_deviation;

		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestMRR= " << fmrr / total << endl;
		std::cout << "Filter.BestHARMONICMEANS = " << total / fmrr << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		std::cout << "Filter.BestDEVIATIONS = " << best_link_fdeviation << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestMRR= " << total / fmrr;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;
		logging.record() << "Filter.BestDEVIATIONS = " << best_link_fdeviation;

		std::cout.flush();
	}
public:
	double getDeviation(vector<double> rank, double mean)
	{
		double sum = 0;
		for (int i = 0; i<rank.size(); i++)
		{
			sum += pow((rank[i] - mean), 2);
		}

		return sqrt(sum / (rank.size() - 1));
	}
public:
	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		return;
	}

	virtual void draw(const string& filename, const int radius,
		const int id_head, const int id_relation)
	{
		return;
	}

	virtual void report(const string& filename) const
	{
		return;
	}
public:
	~Model()
	{
		logging.record();
		if (be_deleted_data_model)
		{
			delete &data_model;
			delete &logging;
		}
	}

public:
	int count_entity() const
	{
		return data_model.set_entity.size();
	}

	int count_relation() const
	{
		return data_model.set_relation.size();
	}

	const DataModel& get_data_model() const
	{
		return data_model;
	}

public:
	virtual void save(const string& filename)
	{
		cout << "BAD";
	}

	virtual void load(const string& filename)
	{
		cout << "BAD";
	}

	virtual vec entity_representation(int entity_id) const
	{
		cout << "BAD";
		return NULL;
	}

	virtual vec relation_representation(int relation_id) const
	{
		cout << "BAD";
		return NULL;
	}
};
 
 
