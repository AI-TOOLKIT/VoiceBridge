/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/
#include "ExamplesUtil.h"

int concurentThreadsSupported = std::thread::hardware_concurrency();

/*
	Simple checking if the input files have newer date than the trained model file. Can be overwritten by the
	FORCE_RETRAIN_MODEL parameter.
*/
bool NeedToRetrainModel(fs::path project_model_dir, 
	fs::path project_input_dir, 
	fs::path project_waves_dir, 
	fs::path project_config,	//the config file for the model training
	fs::path project_base_model_dir) //the model dir on which the final.mdl in the project_model_dir depends on; if not defined then not checked
{
	//check if the model exists
	if (!fs::exists(project_model_dir / "final.mdl"))
		return true;
	//get model's creation time
	time_t model_creation_time = fs::last_write_time(project_model_dir / "final.mdl");
	time_t base_model_creation_time;
	if (project_base_model_dir != "") {
		if (!fs::exists(project_base_model_dir / "final.mdl")) return true;
		base_model_creation_time = fs::last_write_time(project_base_model_dir / "final.mdl");
		if (base_model_creation_time > model_creation_time)
			return true;
	}
	//compare the creation date of each file int he following directories to the creation date of the model (final.mdl)
	std::vector<fs::path> _input = GetFilesInDir(project_input_dir);
	std::vector<fs::path> _wav = GetFilesInDir(project_waves_dir);
	for (fs::path p : _input)
		if (model_creation_time < fs::last_write_time(p))
			return true;
	for (fs::path p : _wav)
		if (model_creation_time < fs::last_write_time(p))
			return true;
	//check the config file for training the model; options could have changed
	if (model_creation_time < fs::last_write_time(project_config))
		return true;

	return false;
}

/*
	Simple checking if we need to decode the test set. It checks if the trained model is newer than lat.1 which is made during decoding.
*/
bool NeedToDecode(fs::path model, fs::path decode_dir)
{
	fs::path lat1(decode_dir / "lat.1");
	if (!fs::exists(lat1)) return true;
	//check if the model exists
	if (!fs::exists(model))
		return false; //this should never happen here
	//get model's creation time
	time_t model_creation_time = fs::last_write_time(model);
	//compare the creation date of each file int he following directories to the creation date of the model (final.mdl)	
	if (model_creation_time > fs::last_write_time(lat1))
		return true;

	return false;
}
