/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : Copyright 2015 Johns Hopkins University (Author: Yenda Trmal <jtrmal@gmail.com>), Apache 2.0.
*/

#include "kaldi-win\scr\kaldi_scr.h"

static std::string rjustify(std::string str, int maxlen) {
	return str.insert(0, maxlen - str.length(), ' ');
}
static std::string ljustify(std::string str, int maxlen) {
	return str.append(maxlen - str.length(), ' ');
}
static void print_header(fs::ofstream * f_out, int SPK_WIDTH, int WIDTH)
{
	std::stringstream sstr;
	sstr << rjustify("SPEAKER", SPK_WIDTH) << " id  " << ljustify("#SENT", WIDTH) << " " << ljustify("#WORD", WIDTH) << " "
		<< ljustify("Corr", WIDTH) << " " << ljustify("Sub", WIDTH) << " " << ljustify("Ins", WIDTH) << " " 
		<< ljustify("Del", WIDTH) << " " << ljustify("Err", WIDTH) << " " << ljustify("S.Err", WIDTH);
	if (f_out!=NULL && *f_out) {
		*f_out << sstr.str() << "\n";
	}
	else {
		LOGTW_INFO << sstr.str();
	}
}
static void format_print_raw (fs::ofstream * f_out, int SPK_WIDTH, int WIDTH,
	std::string spk, int sent, int word, int c, int s, int i, int d, int err, int serr) 
{
	std::stringstream sstr;
	sstr << rjustify(spk, SPK_WIDTH) << " raw " << ljustify(std::to_string(sent), WIDTH) << " " 
		<< ljustify(std::to_string(word), WIDTH) << " " << ljustify(std::to_string(c), WIDTH) << " " 
		<< ljustify(std::to_string(s), WIDTH) << " " << ljustify(std::to_string(i), WIDTH) << " " 
		<< ljustify(std::to_string(d), WIDTH) << " " << ljustify(std::to_string(err), WIDTH) << " " 
		<< ljustify(std::to_string(serr), WIDTH);
	if (f_out != NULL && *f_out) {
		*f_out << sstr.str() << "\n";
	}
	else {
		LOGTW_INFO << sstr.str();
	}
}
static void format_print_sys (fs::ofstream * f_out, int SPK_WIDTH, int WIDTH,
	std::string spk, int sent, int word, double c, double s, double i, double d, double err, double serr){
	std::stringstream sstr;
	sstr << rjustify(spk, SPK_WIDTH) << " sys " 
		<< ljustify(std::to_string(sent), WIDTH) << " "	<< ljustify(std::to_string(word), WIDTH) << " " 
		<< ljustify(to_string_with_precision<double>(c, 2), WIDTH) << " "	//%.2f
		<< ljustify(to_string_with_precision<double>(s, 2), WIDTH) << " "	//%.2f
		<< ljustify(to_string_with_precision<double>(i, 2), WIDTH) << " "	//%.2f
		<< ljustify(to_string_with_precision<double>(d, 2), WIDTH) << " "	//%.2f
		<< ljustify(to_string_with_precision<double>(err, 2), WIDTH) << " "	//%.2f
		<< ljustify(to_string_with_precision<double>(serr, 2), WIDTH);		//%.2f
	if (f_out != NULL && *f_out) {
		*f_out << sstr.str() << "\n";
	}
	else {
		LOGTW_INFO << sstr.str();
	}
}


/*
	This program aggregates the per-utterance output from WerPerUttDetails(). 
	It cares only about the "#csid" field (counts of Corr, Sub, Ins and Del);

	Input:
		UTT-A #csid 3 1 1 1
	Output:
		SPEAKER         id       #SENT      #WORD       Corr        Sub        Ins        Del        Err      S.Err
		A               raw          1          5          3          1          1          1          3          1
		A               sys          1          5      60.00      20.00      20.00      20.00      60.00     100.00
		SUM             raw          1          5          3          1          1          1          3          1
		SUM             sys          1          5      60.00      20.00      20.00      20.00      60.00     100.00
*/
int WerPerSpkDetails(
	fs::path in,
	fs::path utt2spk,
	fs::path out,
	int WIDTH, // Width of the fields (with exception of the SPK ID field)
	int SPK_WIDTH // Width of the first field (spk ID field)
)
{
	StringTable t_in;
	if (ReadStringTable(in.string(), t_in) < 0) return -1;
	//prepare output file
	fs::ofstream f_out(out, fs::ofstream::binary | fs::ofstream::out);
	if (!f_out) {
		LOGTW_ERROR << "Could not write to file " << out.string();
		return -1;
	}
	using UMAPSI = std::unordered_map<std::string, int>;
	using UMAPSS = std::unordered_map<std::string, std::string>;
	using MAPSUMAPSI = std::map<std::string, UMAPSI>; //NOTE: automatically sorted by the first key and unique keys!
	//
	MAPSUMAPSI PERSPK_STATS;
	UMAPSI elem;
	elem.emplace("C", 0);
	//
	StringTable t_utt2spk;
	if (ReadStringTable(utt2spk.string(), t_utt2spk) < 0) return -1;
	//
	UMAPSS UTTMAP;
	for (StringTable::const_iterator it(t_utt2spk.begin()), it_end(t_utt2spk.end()); it != it_end; ++it)
	{
		if ((*it).size() != 2) {
			LOGTW_ERROR << " There should be 2 columns in the utt2spk file!";
			return -1;
		}
		UTTMAP.emplace((*it)[0], (*it)[1]);
		if (SPK_WIDTH < (*it)[1].length()) SPK_WIDTH = (*it)[1].length();
	}
	//initialize PERSPK_STATS
	for (auto & pair : UTTMAP)
	{
		MAPSUMAPSI::iterator itm = PERSPK_STATS.find(pair.second); //normaly all should be unique but...
		if (itm == PERSPK_STATS.end()) {
			PERSPK_STATS.emplace(pair.second, UMAPSI({ { std::string("C") , 0 } }));
			PERSPK_STATS.emplace(pair.second, UMAPSI({ { std::string("S") , 0 } }));
			PERSPK_STATS.emplace(pair.second, UMAPSI({ { std::string("I") , 0 } }));
			PERSPK_STATS.emplace(pair.second, UMAPSI({ { std::string("D") , 0 } }));
			PERSPK_STATS.emplace(pair.second, UMAPSI({ { std::string("SENT") , 0 } }));
			PERSPK_STATS.emplace(pair.second, UMAPSI({ { std::string("SERR") , 0 } }));
		}
	}
	//
	for (StringTable::const_iterator it(t_in.begin()), it_end(t_in.end()); it != it_end; ++it) {
		if ((*it)[1] != "#csid") continue;
		if ((*it).size() != 6) {
			LOGTW_ERROR << "Incompatible entry in " << in.string();
			return -1;
		}

		int c = StringToNumber<int>((*it)[2], -1);
		int s = StringToNumber<int>((*it)[3], -1);
		int i = StringToNumber<int>((*it)[4], -1);
		int d = StringToNumber<int>((*it)[5], -1);
		if (c < 0 || s < 0 || i < 0 || d < 0) {
			LOGTW_ERROR << "Entry expected to be a number in " << in.string();
			return -1;
		}
		std::string UTT = (*it)[0];
		//make sure that it is UTF8
		UTT = ConvertToUTF8(UTT);
		UMAPSS::iterator itm = UTTMAP.find(UTT);
		if (itm == UTTMAP.end()) {
			LOGTW_ERROR << "Utterance " << UTT << " can not be found in utt2spk.";
			return -1;
		}
		std::string SPK = UTTMAP[UTT];
		//collect stats
		PERSPK_STATS[SPK]["C"] += c;
		PERSPK_STATS[SPK]["S"] += s;
		PERSPK_STATS[SPK]["I"] += i;
		PERSPK_STATS[SPK]["D"] += d;
		PERSPK_STATS[SPK]["SENT"] += 1;
		if (s + i + d != 0)
			PERSPK_STATS[SPK]["SERR"] += 1;
	}

	int C = 0;
	int S = 0;
	int I = 0;
	int D = 0;
	int SENT = 0;
	int WORD = 0;
	int ERR = 0;
	int SERR = 0;

	//NOTE: we print everything twice, first to the screen and then to the output file!
	print_header(NULL, SPK_WIDTH, WIDTH);
	print_header(&f_out, SPK_WIDTH, WIDTH);

	for (auto & pair : PERSPK_STATS)
	{
		std::string SPK(pair.first);
		int c = PERSPK_STATS[SPK]["C"];
		int s = PERSPK_STATS[SPK]["S"];
		int i = PERSPK_STATS[SPK]["I"];
		int d = PERSPK_STATS[SPK]["D"];
		int sent = PERSPK_STATS[SPK]["SENT"];
		int word = c + s + d;
		int err = s + d + i;
		int serr = PERSPK_STATS[SPK]["SERR"]; // 0;

		C += c; S += s; I += i; D += d;
		SENT += sent; SERR += serr;
		double w = 1.0 * word;

		//NOTE: we print everything twice, first to the screen and then to the output file!
		//screen
		format_print_raw(NULL, SPK_WIDTH, WIDTH, SPK, sent, word, c, s, i, d, err, serr);
		if (w != 0) {
			format_print_sys(NULL, SPK_WIDTH, WIDTH, SPK, sent, word, 100.0 * c / w, 100.0 * s / w,
				100.0 * i / w, 100.0 * d / w, 100.0 * err / w, 100.0 * serr / sent);
		}
		//output file
		//NOTE: we print everything twice, first to the screen and then to the output file!
		format_print_raw(&f_out, SPK_WIDTH, WIDTH, SPK, sent, word, c, s, i, d, err, serr);
		if (w != 0) {
			format_print_sys(&f_out, SPK_WIDTH, WIDTH, SPK, sent, word, 100.0 * c / w, 100.0 * s / w,
				100.0 * i / w, 100.0 * d / w, 100.0 * err / w, 100.0 * serr / sent);
		}
	}

	WORD = C + S + D;
	ERR = S + D + I;
	double W = 1.0 * WORD;

	//NOTE: we print everything twice, first to the screen and then to the output file!
	//screen
	format_print_raw(NULL, SPK_WIDTH, WIDTH, "SUM", SENT, WORD, C, S, I, D, ERR, SERR);
	if (W != 0) {
		format_print_sys(NULL, SPK_WIDTH, WIDTH, "SUM", SENT, WORD, 100.0 * C / W, 100.0 * S / W,
				100.0 * I / W, 100.0 * D / W, 100.0 * ERR / W, 100.0 * SERR / SENT);
	}
	//output file
	//NOTE: we print everything twice, first to the screen and then to the output file!
	format_print_raw(&f_out, SPK_WIDTH, WIDTH, "SUM", SENT, WORD, C, S, I, D, ERR, SERR);
	if (W != 0) {
		format_print_sys(&f_out, SPK_WIDTH, WIDTH, "SUM", SENT, WORD, 100.0 * C / W, 100.0 * S / W,
			100.0 * I / W, 100.0 * D / W, 100.0 * ERR / W, 100.0 * SERR / SENT);
	}

	f_out.flush(); f_out.close();
	return 0;
}
