#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include "Model.hpp"
#include <cmath>
#include <fstream>

class TransE
	:public Model
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation;

public:
	const int	dim;
	const double	alpha;
	const double	training_threshold;

public:
	double			log_base;
	int				multify;

public:
	TransE(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double Log_base = 10.0,
		int Multify = 2)
		:Model(dataset, task_type, logging_base_path),
		dim(dim), alpha(alpha), training_threshold(training_threshold)
	{
		logging.record() << "\t[Name]\tTransE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;

		log_base = Log_base;
		multify = Multify;

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem) {elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });

		embedding_relation.resize(count_relation());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem) {elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
	}

	TransE(const Dataset& dataset,
		const string& file_zero_shot,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:Model(dataset, file_zero_shot, task_type, logging_base_path),
		dim(dim), alpha(alpha), training_threshold(training_threshold)
	{
		logging.record() << "\t[Name]\tTransE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem) {elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim);});

		embedding_relation.resize(count_relation());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem) {elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim);});
	}

	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -sum(abs(error));
	}

	virtual double prob_triplets_subgraph(const pair<pair<int, int>, int>& triplet, vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s, vector<int>& size_clusters_s)
	{
		vec error = embedding_entity_s[triplet.first.first]
			+ embedding_relation_s[triplet.second]
			- embedding_entity_s[triplet.first.second];

		return -sum(abs(error));
	}

	virtual void deep_copy_for_subgraph(vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s, vector<vector<vec>>& embedding_clusters_s,
		vector<vec>& weights_clusters_s, vector<int>& size_clusters_s)
	{
		embedding_entity_s = embedding_entity;
		embedding_relation_s = embedding_relation;
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);
	}

	virtual void train_triplet_subgraph(const pair<pair<int, int>, int>& triplet, vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s, vector<int>& size_clusters_s,
		vector<pair<pair<int, int>, int>> subgraph)
	{
		vec& head = embedding_entity_s[triplet.first.first];
		vec& tail = embedding_entity_s[triplet.first.second];
		vec& relation = embedding_relation_s[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets_subgraph(triplet, embedding_entity_s, embedding_relation_s, embedding_clusters_s, weights_clusters_s, size_clusters_s) - prob_triplets_subgraph(triplet_f, embedding_entity_s, embedding_relation_s, embedding_clusters_s, weights_clusters_s, size_clusters_s) > training_threshold)
			return;

		vec& head_f = embedding_entity_s[triplet_f.first.first];
		vec& tail_f = embedding_entity_s[triplet_f.first.second];
		vec& relation_f = embedding_relation_s[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);

	}

	virtual void train_triplet_subgraph_BM(const pair<pair<int, int>, int>& triplet, vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s, vector<int>& size_clusters_s,
		vector<pair<pair<int, int>, int>> subgraph, vector<int> cut_pos, vector<int> cut_tot, vector<int> cut_tot_rel)

	{
		vec& head = embedding_entity_s[triplet.first.first];
		vec& tail = embedding_entity_s[triplet.first.second];
		vec& relation = embedding_relation_s[triplet.second];

		if (prob_triplets_subgraph(triplet, embedding_entity_s, embedding_relation_s, embedding_clusters_s, weights_clusters_s, size_clusters_s) - prob_triplets(triplet) < 0.001)
			return;

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);
	}

	virtual void relation_reg(int i, int j, double factor)
	{
		if (i == j)
			return;

		embedding_relation[i] -= alpha * factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[j];
		embedding_relation[j] -= alpha * factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[i];
	}

	virtual void entity_reg(int i, int j, double factor)
	{
		if (i == j)
			return;

		embedding_entity[i] -= alpha * factor * sign(as_scalar(embedding_entity[i].t()*embedding_entity[j])) * embedding_entity[j];
		embedding_entity[j] -= alpha * factor * sign(as_scalar(embedding_entity[i].t()*embedding_entity[j])) * embedding_entity[i];
	}

public:
	virtual vec entity_representation(int entity_id) const
	{
		return embedding_entity[entity_id];
	}

	virtual vec relation_representation(int relation_id) const
	{
		return embedding_relation[relation_id];
	}

	virtual void save(const string& filename) override
	{
		ofstream fout(filename, ios::binary);
		storage_vmat<double>::save(embedding_entity, fout);
		storage_vmat<double>::save(embedding_relation, fout);
		fout.close();
	}

	virtual void load(const string& filename) override
	{
		ifstream fin(filename, ios::binary);
		storage_vmat<double>::load(embedding_entity, fin);
		storage_vmat<double>::load(embedding_relation, fin);
		fin.close();
	}
};

class TransG
	:public Model
{
protected:
	vector<vec>				embedding_entity;
	vector<vector<vec>>		embedding_clusters;
	vector<vec>				weights_clusters;
	vector<int>		size_clusters;

protected:
	const int				n_cluster;
	const double			alpha;
	const bool				single_or_total;
	const double			training_threshold;
	const int			dim;
	const bool				be_weight_normalized;
	const int			step_before;
	const double			normalizor;

protected:
	double			CRP_factor;

public:
	double			log_base;
	int				multify;

public:
	TransG(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		int n_cluster,
		double CRP_factor,
		double Log_base = 10.0,
		int Multify = 2,
		int step_before = 10,
		bool sot = false,
		bool be_weight_normalized = true)
		:Model(dataset, task_type, logging_base_path), dim(dim), alpha(alpha),
		training_threshold(training_threshold), n_cluster(n_cluster), CRP_factor(CRP_factor),
		single_or_total(sot), be_weight_normalized(be_weight_normalized), step_before(step_before),
		normalizor(1.0 / pow(3.1415, dim / 2))
	{
		logging.record() << "\t[Name]\tTransM";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Cluster Counts]\t" << n_cluster;
		logging.record() << "\t[CRP Factor]\t" << CRP_factor;

		log_base = Log_base;
		multify = Multify;

		if (be_weight_normalized)
			logging.record() << "\t[Weight Normalized]\tTrue";
		else
			logging.record() << "\t[Weight Normalized]\tFalse";

		if (sot)
			logging.record() << "\t[Single or Total]\tTrue";
		else
			logging.record() << "\t[Single or Total]\tFalse";

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem) {elem = randu(dim, 1); });

		embedding_clusters.resize(count_relation());
		for (auto &elem_vec : embedding_clusters)
		{
			elem_vec.resize(30);
			for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem) {elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
		}

		weights_clusters.resize(count_relation());
		for (auto & elem_vec : weights_clusters)
		{
			elem_vec.resize(21);
			elem_vec.fill(0.0);
			for (auto i = 0; i<n_cluster; ++i)
			{
				elem_vec[i] = 1.0 / n_cluster;
			}
		}

		size_clusters.resize(count_relation(), n_cluster);
		this->CRP_factor = CRP_factor / data_model.data_train.size() * count_relation();
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		if (single_or_total == false)
			return training_prob_triplets(triplet);

		double	mixed_prob = 1e-100;
		for (int c = 0; c<size_clusters[triplet.second]; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
			mixed_prob = max(mixed_prob, fabs(weights_clusters[triplet.second][c])
				* exp(-sum(abs(error_c))));
		}

		return mixed_prob;
	}

	virtual double prob_triplets_subgraph(const pair<pair<int, int>, int>& triplet, vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s, vector<int>& size_clusters_s)
	{
		if (single_or_total == false)
			return training_prob_triplets_subgraph(triplet, size_clusters_s, embedding_entity_s, embedding_clusters_s, weights_clusters_s);

		double	mixed_prob = 1e-100;
		for (int c = 0; c<size_clusters_s[triplet.second]; ++c)
		{
			vec error_c = embedding_entity_s[triplet.first.first] + embedding_clusters_s[triplet.second][c]
				- embedding_entity_s[triplet.first.second];
			mixed_prob = max(mixed_prob, fabs(weights_clusters_s[triplet.second][c])
				* exp(-sum(abs(error_c))));
		}

		return mixed_prob;
	}

	virtual double training_prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		double	mixed_prob = 1e-100;
		for (int c = 0; c<size_clusters[triplet.second]; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
			mixed_prob += fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c)));
		}

		return mixed_prob;
	}

	virtual double training_prob_triplets_subgraph(const pair<pair<int, int>, int>& triplet, vector<int>& size_clusters_s, vector<vec>& embedding_entity_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s)
	{
		double	mixed_prob = 1e-100;
		for (int c = 0; c<size_clusters_s[triplet.second]; ++c)
		{
			vec error_c = embedding_entity_s[triplet.first.first] + embedding_clusters_s[triplet.second][c]
				- embedding_entity_s[triplet.first.second];
			mixed_prob += fabs(weights_clusters_s[triplet.second][c]) * exp(-sum(abs(error_c)));
		}

		return mixed_prob;
	}

	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		mat	record(radius*6.0 + 10, radius*6.0 + 10);
		record.fill(255);
		for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
		{
			if (i->second == id_relation)
			{
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]),
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1])) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
			}
		}

		string relation_name = data_model.relation_id_to_name[id_relation];
		record.save(filename + replace_all(relation_name, "/", "_") + ".ppm", pgm_binary);
	}

	/* train_cluster_once, train_triplet �Լ����� overloading �ϸ� �ɵ�! */
	virtual void train_cluster_once(
		const pair<pair<int, int>, int>& triplet,
		const pair<pair<int, int>, int>& triplet_f,
		int cluster, double prob_true, double prob_false, double factor)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_clusters[triplet.second][cluster];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_clusters[triplet_f.second][cluster];

		double prob_local_true = exp(-sum(abs(head + relation - tail)));
		double prob_local_false = exp(-sum(abs(head_f + relation_f - tail_f)));

		weights_clusters[triplet.second][cluster] +=
			factor / prob_true * prob_local_true * sign(weights_clusters[triplet.second][cluster]);
		weights_clusters[triplet_f.second][cluster] -=
			factor / prob_false * prob_local_false  * sign(weights_clusters[triplet_f.second][cluster]);

		head -= factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		tail += factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		relation -= factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		head_f += factor * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]);
		tail_f -= factor * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false  * fabs(weights_clusters[triplet.second][cluster]);
		relation_f += factor * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(relation_f) > 1.0)
			relation_f = normalise(relation_f);
	}


	virtual void train_cluster_once_subgraph(
		const pair<pair<int, int>, int>& triplet,
		const pair<pair<int, int>, int>& triplet_f,
		int cluster, double prob_true, double prob_false, double factor,
		vector<int>& size_clusters_s, vector<vec>& embedding_entity_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s,
		vector<pair<pair<int, int>, int>> subgraph)
	{
		vec& head = embedding_entity_s[triplet.first.first];
		vec& tail = embedding_entity_s[triplet.first.second];
		vec& relation = embedding_clusters_s[triplet.second][cluster];
		vec& head_f = embedding_entity_s[triplet_f.first.first];
		vec& tail_f = embedding_entity_s[triplet_f.first.second];
		vec& relation_f = embedding_clusters_s[triplet_f.second][cluster];

		double prob_local_true = exp(-sum(abs(head + relation - tail)));
		double prob_local_false = exp(-sum(abs(head_f + relation_f - tail_f)));

		weights_clusters_s[triplet.second][cluster] +=
			factor / prob_true * prob_local_true * sign(weights_clusters_s[triplet.second][cluster]);
		weights_clusters_s[triplet_f.second][cluster] -=
			factor / prob_false * prob_local_false  * sign(weights_clusters_s[triplet_f.second][cluster]);

		head -= factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		tail += factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		relation -= factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		head_f += factor * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]);
		tail_f -= factor * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false  * fabs(weights_clusters[triplet.second][cluster]);
		relation_f += factor * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(relation_f) > 1.0)
			relation_f = normalise(relation_f);
	}

	virtual void train_cluster_once_subgraph_BM(
		const pair<pair<int, int>, int>& triplet,
		const pair<pair<int, int>, int>& triplet_f,
		int cluster, double prob_true, double prob_false, double factor,
		vector<int>& size_clusters_s, vector<vec>& embedding_entity_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s,
		vector<pair<pair<int, int>, int>> subgraph, vector<int> cut_pos, vector<int> cut_tot,
		vector<int> cut_tot_rel)
	{
		vec& head = embedding_entity_s[triplet.first.first];
		vec& tail = embedding_entity_s[triplet.first.second];
		vec& relation = embedding_clusters_s[triplet.second][cluster];

		double prob_local_true = exp(-sum(abs(head + relation - tail)));

		weights_clusters_s[triplet.second][cluster] +=
			factor / prob_true * prob_local_true * sign(weights_clusters_s[triplet.second][cluster]);

		head -= factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		tail += factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		relation -= factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];

		if (!head.is_finite())
			cout << "d";

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		double prob_true = training_prob_triplets(triplet);
		double prob_false = training_prob_triplets(triplet_f);

		if (prob_true / prob_false > exp(training_threshold))
			return;

		for (int c = 0; c<size_clusters[triplet.second]; ++c)
		{
			train_cluster_once(triplet, triplet_f, c, prob_true, prob_false, alpha);
		}

		double prob_new_component = CRP_factor * exp(-sum(abs(head - tail)));

		if (randu() < prob_new_component / (prob_new_component + prob_true)
			&& size_clusters[triplet.second] < 20
			&& epos >= step_before)
		{
#pragma omp critical
			{
				weights_clusters[triplet.second][size_clusters[triplet.second]] = CRP_factor;
				embedding_clusters[triplet.second][size_clusters[triplet.second]] = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); //tail - head;
				++size_clusters[triplet.second];
			}
		}

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);

		if (be_weight_normalized)
			weights_clusters[triplet.second] = normalise(weights_clusters[triplet.second]);
	}

	void getNegativeTriplet(pair<pair<int, int>, int>& triplet, vector<pair<pair<int, int>, int>> subgraph, int subgraph_idx)
	{
		srand((unsigned int)time(0));
		int selectHead = rand() % 2;

		set<int> A;
		for (auto t : subgraph) {
			A.insert(t.first.first);
			A.insert(t.first.second);
		}
		for (auto v : A) {
			if (selectHead) triplet.first.first = v;
			else triplet.first.second = v;
			bool uniq = true;
			for (auto t : subgraph) {
				if (triplet == t) {
					uniq = false;
					break;
				}
			}
			if (!uniq) continue;
			if (triplet == data_model.data_test_true[subgraph_idx]) continue;
			if (triplet.first.first == triplet.first.second) continue;
			break;
		}
	}

	virtual void train_triplet_subgraph(const pair<pair<int, int>, int>& triplet, vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s, vector<int>& size_clusters_s,
		vector<pair<pair<int, int>, int>> subgraph)
	{
		vec& head = embedding_entity_s[triplet.first.first];
		vec& tail = embedding_entity_s[triplet.first.second];

		if (!head.is_finite())
			cout << "d";

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		double prob_true = training_prob_triplets_subgraph(triplet, size_clusters_s, embedding_entity_s, embedding_clusters_s, weights_clusters_s);
		double prob_false = training_prob_triplets_subgraph(triplet_f, size_clusters_s, embedding_entity_s, embedding_clusters_s, weights_clusters_s);

		if (prob_true / prob_false > exp(training_threshold))
			return;

		for (int c = 0; c<size_clusters_s[triplet.second]; ++c)
		{
			train_cluster_once_subgraph(triplet, triplet_f, c, prob_true, prob_false, alpha, size_clusters_s, embedding_entity_s, embedding_clusters_s, weights_clusters_s, subgraph);
		}

		double prob_new_component = CRP_factor * exp(-sum(abs(head - tail)));

		if (randu() < prob_new_component / (prob_new_component + prob_true)
			&& size_clusters_s[triplet.second] < 20
			&& epos >= step_before)
		{
#pragma omp critical
			{
				weights_clusters_s[triplet.second][size_clusters_s[triplet.second]] = CRP_factor;
				embedding_clusters_s[triplet.second][size_clusters_s[triplet.second]] = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); //tail - head;
				++size_clusters_s[triplet.second];
			}
		}

		vec& head_f = embedding_entity_s[triplet_f.first.first];
		vec& tail_f = embedding_entity_s[triplet_f.first.second];

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);

		if (be_weight_normalized)
			weights_clusters_s[triplet.second] = normalise(weights_clusters_s[triplet.second]);
	}

	virtual void train_triplet_subgraph_BM(const pair<pair<int, int>, int>& triplet, vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s,
		vector<vector<vec>>& embedding_clusters_s, vector<vec>& weights_clusters_s, vector<int>& size_clusters_s,
		vector<pair<pair<int, int>, int>> subgraph, vector<int> cut_pos, vector<int> cut_tot, vector<int> cut_tot_rel)
	{
		vec& head = embedding_entity_s[triplet.first.first];
		vec& tail = embedding_entity_s[triplet.first.second];

		if (!head.is_finite())
			cout << "d";

		double prob_true = training_prob_triplets_subgraph(triplet, size_clusters_s, embedding_entity_s, embedding_clusters_s, weights_clusters_s);
		double prob_origin = training_prob_triplets_subgraph(triplet, size_clusters, embedding_entity, embedding_clusters, weights_clusters);

		if (prob_true / prob_origin < exp(0.001))
			return;

		pair<pair<int, int>, int> triplet_f = triplet;
		data_model.sample_false_triplet(triplet, triplet_f);
		double prob_false = training_prob_triplets_subgraph(triplet_f, size_clusters_s, embedding_entity_s, embedding_clusters_s, weights_clusters_s);

		for (int c = 0; c<size_clusters_s[triplet.second]; ++c)
		{
			train_cluster_once_subgraph_BM(triplet, triplet_f, c, prob_true, prob_false, multify * alpha, size_clusters_s, embedding_entity_s, embedding_clusters_s, weights_clusters_s, subgraph, cut_pos, cut_tot, cut_tot_rel);
		}

		double prob_new_component = CRP_factor * exp(-sum(abs(head - tail)));

		if (randu() < prob_new_component / (prob_new_component + prob_true)
			&& size_clusters_s[triplet.second] < 20
			&& epos >= step_before)
		{
#pragma omp critical
			{
				weights_clusters_s[triplet.second][size_clusters_s[triplet.second]] = CRP_factor;
				embedding_clusters_s[triplet.second][size_clusters_s[triplet.second]] = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); //tail - head;
				++size_clusters_s[triplet.second];
			}
		}

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (be_weight_normalized)
			weights_clusters_s[triplet.second] = normalise(weights_clusters_s[triplet.second]);
	}

public:
	virtual void deep_copy_for_subgraph(vector<vec>& embedding_entity_s, vector<vec>& embedding_relation_s, vector<vector<vec>>& embedding_clusters_s,
		vector<vec>& weights_clusters_s, vector<int>& size_clusters_s)
	{
		embedding_entity_s = embedding_entity;
		embedding_clusters_s = embedding_clusters;
		weights_clusters_s = weights_clusters;
		size_clusters_s = size_clusters;
	}
	virtual void report(const string& filename) const
	{
		if (task_type == TransM_ReportClusterNumber)
		{
			for (auto i = 0; i<count_relation(); ++i)
			{
				cout << data_model.relation_id_to_name[i] << ':';
				cout << size_clusters[i] << endl;
			}
			return;
		}
		else if (task_type == TransM_ReportDetailedClusterLabel)
		{
			vector<bitset<32>>	counts_component(count_relation());
			ofstream fout(filename.c_str());
			for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
			{
				int pos_cluster = 0;
				double	mixed_prob = 1e-8;
				for (int c = 0; c<n_cluster; ++c)
				{
					vec error_c = embedding_entity[i->first.first]
						+ embedding_clusters[i->second][c]
						- embedding_entity[i->first.second];
					if (mixed_prob < exp(-sum(abs(error_c))))
					{
						pos_cluster = c;
						mixed_prob = exp(-sum(abs(error_c)));
					}
				}

				counts_component[i->second][pos_cluster] = 1;
				fout << data_model.entity_id_to_name[i->first.first] << '\t';
				fout << data_model.relation_id_to_name[i->second] << "==" << pos_cluster << '\t';
				fout << data_model.entity_id_to_name[i->first.second] << '\t';
				fout << endl;
			}
			fout.close();

			for (auto i = 0; i != counts_component.size(); ++i)
			{
				fout << data_model.relation_id_to_name[i] << ":";
				fout << counts_component[i].count() << endl;
			}
		}
	}
};
 
