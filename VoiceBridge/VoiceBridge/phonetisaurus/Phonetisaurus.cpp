/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
Based on Phonetisaurus: Copyright (c) [2012-], Josef Robert Novak, All rights reserved.
*/
#include "Phonetisaurus.h"
#include "..\mitlm\mitlm.h"

int G2Pfst(int argc, char* argv[]);
int Arpa2Wfst(int argc, char* argv[]);
int Align(int argc, char* argv[]);

/*
	Pronunciation generator
*/
namespace Phonetisaurus {
	/*
		Trains an FST model from a reference dictionary. The model can then be used to acquire the pronunciation 
		for any word in the same language. The pronunciation for the words which are in the reference dictinory
		can be acquired from the model or can be taken from the reference dictionary (option).
		refDictionary: path to the ref dictionary
		outModel: path to the output model
		ngramOrder: the ngram order to use for training the model; higher values will provide a higher accuracy
					but will train slower and will produce a larger model file.
	*/
	VOICEBRIDGE_API int TrainModel(fs::path refDictionary, fs::path outModel, int ngramOrder)
	{
		fs::path basepath(refDictionary.parent_path());
		/*
			Convert dictionary to Phonetisaurus format.
			NOTE: Phonetisaurus relies on a special format in which the word must be separated from the phonems by
				  a TAB character. All other elements are space separated. If this requirement is not met then the
				  software will not output the pronunciation!
		*/
		fs::ifstream ifs(refDictionary);
		if (!ifs) {
			LOGTW_ERROR << "Could not open " << refDictionary.string();
			return -1;
		}
		fs::ofstream ofs(basepath / "dict.p", std::ios::binary | std::ios::out);
		if (!ofs) {
			LOGTW_ERROR << "Could not open output file cmudict.dict.p";
			return -1;
		}
		std::string line;
		static const boost::regex rexp("\\([0-9]+\\)");
		while (std::getline(ifs, line)) {
			//remove space from the start and tail.
			boost::algorithm::trim(line);
			std::vector<std::string> _w;
			//use regex to remove (n) from repeated words //TODO:... should repeated words be kept?				
			line = boost::regex_replace(line, rexp, "");
			//there can be a comment in the line: remove the comment
			size_t pos = line.find_first_of('#', 0);
			if(pos != std::string::npos)
				line = line.substr(0, pos-1);
			//split the line
			strtk::parse(line, " ", _w, strtk::split_options::compress_delimiters);
			ofs << ConvertToCaseUtf8(_w[0], false) << "\t"; //NOTE: word lower case!
			int c = 1;
			for (int i = 1; i < _w.size(); i++) {
				if (c > 1) ofs << " ";
				ofs << ConvertToCaseUtf8(_w[i], true); //NOTE: phonems upper case!
				c++;
			}
			ofs << "\n";
		}
		ofs.flush(); ofs.close();

		/* TRAIN MODEL: */
		std::vector<std::string> options = {
		"--input_A=" + (basepath / "dict.p").string(),
		"--ofile_A=" + (basepath / "dict.a").string(),
		"--seq2_del_A=true"
		//DEBUG: ,"--write_model_A="+(basepath / "cmudict.dict.a2").string()
		};
		StrVec2Arg args(options);
		if(Align(args.argc(), args.argv()) < 0) return -1;

		//--------------------------------------------------------
		std::vector<std::string> options2 = {
			"-o", std::to_string(ngramOrder),
			"-t", (basepath / "dict.a").string(),
			"-wl", (basepath / "dict.arpa").string()
		};
		StrVec2Arg args2(options2);
		if(EstimateNgram(args2.argc(), args2.argv()) < 0) return -1;
		//--------------------------------------------------------
		std::vector<std::string> options3 = {
			"-lm_W=" + (basepath / "dict.arpa").string(),
			"-ofile_W=" + outModel.string()
		};
		StrVec2Arg args3(options3);
		if(Arpa2Wfst(args3.argc(), args3.argv()) < 0) return -1;
		//--------------------------------------------------------
		
		//for DEBUG purposes :
		/*
			TEST MODEL WITH A LIST OF WORDS - Apply the G2P model to a word list
		*/
		/*
		std::vector<std::string> options4 = {
			"--model_P=" + (basepath / "cmudict.dict.model").string(),
			"--wordlist_P=" + (basepath / "test_wordlist.txt").string()
			//Could adjust but not used at the moment: ,"pmass_P=99" ,"nbest_P=3" ,"beam_P=5000"
			//DEBUG: ,"--write_fsts_P=true"
		};
		StrVec2Arg args4(options4);
		G2Pfst(args4.argc(), args4.argv());
		*/
		//--------------------------------------------------------

		/*
		//for DEBUG purposes:
		fst::FstInfo info;
		fstinfo((basepath / "cmudict.dict.model").string(), "any", "auto", false, false, &info);
		//
		std::vector<std::string> optionsI = {
			"--save_isymbols="+(basepath / "isyms").string(),
			"--save_osymbols="+(basepath / "osyms").string(),
			(basepath / "cmudict.dict.model").string(),
		};
		StrVec2Arg argsI(optionsI);
		fstsymbols(argsI.argc(), argsI.argv());
		*/

		//cleanup
		try	{
			if (fs::exists(basepath / "dict.p")) fs::remove(basepath / "dict.p");
			if (fs::exists(basepath / "dict.a")) fs::remove(basepath / "dict.a");
			if (fs::exists(basepath / "dict.arpa")) fs::remove(basepath / "dict.arpa");

		}
		catch (const std::exception& e)
		{
			LOGTW_WARNING << "Could not remove temporary files. Reason: " << e.what();
		}
		return 0;
	}

	/*
		Get the pronunciation for a list of words by using the trained model and the ref dictionary
		refDictionary: path to the ref dictionary
		model: path to the output model
		intxtfile: the path to the text file containing the list of words for which you want the pronunciation
		outtxtfile: output text file
		forceModel: if true the model will be used for getting the pronunciation; if false the reference dictionary
					will be used for existing words for getting the pronunciation.
	*/
	VOICEBRIDGE_API int GetPronunciation(fs::path refDictionary, fs::path model, fs::path intxtfile, fs::path outtxtfile, bool forceModel)
	{
		/*
			Apply the G2P model to a word list
		*/
		std::vector<std::string> options = {
		"--model_P=" + model.string(),
		"--refdict_P=" + refDictionary.string(),
		"--forcemodel_P=" + bool_as_text(forceModel),
		"--wordlist_P=" + intxtfile.string(),
		"--outfile_P=" + outtxtfile.string()
		//Could adjust but not used at the moment: ,"pmass_P=99" ,"nbest_P=3" ,"beam_P=5000"
		//DEBUG: ,"--write_fsts_P=true"
		};
		StrVec2Arg args(options);
		if (G2Pfst(args.argc(), args.argv()) < 0) return -1;

		return 0;
	}

}
