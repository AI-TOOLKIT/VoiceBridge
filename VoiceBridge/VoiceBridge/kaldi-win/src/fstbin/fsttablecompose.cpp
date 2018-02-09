/*
Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.0 - January 9, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.

Based on : openfst
*/
// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

#include <fst/script/compose.h>
#include <fst/script/getters.h>

#include "fst_ext.h"

/*
cd ~/tmpdir
while true; do
fstrand  | fstarcsort --sort_type=olabel > 1.fst; fstrand | fstarcsort > 2.fst
fstcompose 1.fst 2.fst > 3a.fst
fsttablecompose 1.fst 2.fst > 3b.fst
fstequivalent --random=true 3a.fst 3b.fst || echo "Test failed"
echo -n "."
done

*/
using namespace kaldi;
using namespace fst;
using fst::script::VectorFstClass;

/*
Composition algorithm [between two FSTs of standard type, in tropical semiring] that is more efficient for certain 
cases-- in particular, where one of the FSTs (the left one, if --match-side=left) has large out-degree
*/
int fsttablecompose(std::string fst1_in_str, std::string fst2_in_str, std::string fst_out_str,
	std::string match_side,			//Side of composition to do table match, one of: \"left\" or \"right\"
	std::string composeFilter,		//Composition filter to use, one of: \"alt_sequence\", \"auto\", \"match\", \"sequence\"
	bool bConnect)					//If true, trim FST before output.
{
	try {
		using kaldi::int32;
		/*
		fsttablecompose should always give equivalent results to compose, but it is more efficient for certain kinds of inputs.
		In particular, it is useful when, say, the left FST has states that typically either have epsilon olabels, or
		one transition out for each of the possible symbols (as the olabel).  The same with the input symbols of the right-hand FST
		is possible.
		*/
		TableComposeOptions opts;
		std::string match_side = "left";
		std::string compose_filter = "sequence";
		opts.connect = bConnect;

		if (match_side == "left") {
			opts.table_match_type = MATCH_OUTPUT;
		}
		else if (match_side == "right") {
			opts.table_match_type = MATCH_INPUT;
		}
		else {
			LOGTW_ERROR << "Invalid match-side option: " << match_side;
			return -1;
		}

		if (compose_filter == "alt_sequence") {
			opts.filter_type = ALT_SEQUENCE_FILTER;
		}
		else if (compose_filter == "auto") {
			opts.filter_type = AUTO_FILTER;
		}
		else  if (compose_filter == "match") {
			opts.filter_type = MATCH_FILTER;
		}
		else  if (compose_filter == "sequence") {
			opts.filter_type = SEQUENCE_FILTER;
		}
		else {
			LOGTW_ERROR << "Invalid compose-filter option: " << compose_filter;
			return -1;
		}

		// Note: the "table" in is_table_1 and similar variables has nothing
		// to do with the "table" in "fsttablecompose"; is_table_1 relates to
		// whether we are dealing with a single FST or a whole set of FSTs.
		bool is_table_1 =
			(ClassifyRspecifier(fst1_in_str, NULL, NULL) != kNoRspecifier),
			is_table_2 =
			(ClassifyRspecifier(fst2_in_str, NULL, NULL) != kNoRspecifier),
			is_table_out =
			(ClassifyWspecifier(fst_out_str, NULL, NULL, NULL) != kNoWspecifier);
		if (is_table_out != (is_table_1 || is_table_2)) {
			LOGTW_ERROR << "Incompatible combination of archives and files";
			return -1;
		}

		if (!is_table_1 && !is_table_2) { // Only dealing with files...
			VectorFst<StdArc> *fst1 = ReadFstKaldi(fst1_in_str);

			VectorFst<StdArc> *fst2 = ReadFstKaldi(fst2_in_str);

			// Checks if <fst1> is olabel sorted and <fst2> is ilabel sorted.
			if (fst1->Properties(fst::kOLabelSorted, true) == 0) {
				LOGTW_WARNING << " The first FST is not olabel sorted.";
			}
			if (fst2->Properties(fst::kILabelSorted, true) == 0) {
				LOGTW_WARNING << " The second FST is not ilabel sorted.";
			}

			VectorFst<StdArc> composed_fst;

			TableCompose(*fst1, *fst2, &composed_fst, opts);

			delete fst1;
			delete fst2;

			WriteFstKaldi(composed_fst, fst_out_str);
			return 0;
		}
		else if (!is_table_1 && is_table_2
			&& opts.table_match_type == MATCH_OUTPUT) {
			// second arg is an archive, and match-side=left (default).
			TableComposeCache<Fst<StdArc> > cache(opts);
			VectorFst<StdArc> *fst1 = ReadFstKaldi(fst1_in_str);
			SequentialTableReader<VectorFstHolder> fst2_reader(fst2_in_str);
			TableWriter<VectorFstHolder> fst_writer(fst_out_str);
			int32 n_done = 0;

			// Checks if <fst1> is olabel sorted.
			if (fst1->Properties(fst::kOLabelSorted, true) == 0) {
				LOGTW_WARNING << " The first FST is not olabel sorted.";
			}
			for (; !fst2_reader.Done(); fst2_reader.Next(), n_done++) {
				VectorFst<StdArc> fst2(fst2_reader.Value());
				VectorFst<StdArc> fst_out;
				TableCompose(*fst1, fst2, &fst_out, &cache);
				fst_writer.Write(fst2_reader.Key(), fst_out);
			}
			LOGTW_INFO << "Composed " << n_done << " FSTs.";
			return (n_done != 0 ? 0 : 1);
		}
		else if (is_table_1 && is_table_2) {
			SequentialTableReader<VectorFstHolder> fst1_reader(fst1_in_str);
			RandomAccessTableReader<VectorFstHolder> fst2_reader(fst2_in_str);
			TableWriter<VectorFstHolder> fst_writer(fst_out_str);
			int32 n_done = 0, n_err = 0;
			for (; !fst1_reader.Done(); fst1_reader.Next()) {
				std::string key = fst1_reader.Key();
				if (!fst2_reader.HasKey(key)) {
					LOGTW_WARNING << " No such key " << key << " in second table.";
					n_err++;
				}
				else {
					const VectorFst<StdArc> &fst1(fst1_reader.Value()),
						&fst2(fst2_reader.Value(key));
					VectorFst<StdArc> result;
					TableCompose(fst1, fst2, &result, opts);
					if (result.NumStates() == 0) {
						LOGTW_WARNING << " Empty output for key " << key;
						n_err++;
					}
					else {
						fst_writer.Write(key, result);
						n_done++;
					}
				}
			}
			LOGTW_INFO << "Successfully composed " << n_done << " FSTs, errors or "
				<< "empty output on " << n_err;
		}
		else {
			LOGTW_ERROR << "The combination of tables/non-tables and match-type that you "
				<< "supplied is not currently supported.  Either implement this, "
				<< "ask the maintainers to implement it, or call this program "
				<< "differently.";
			return -1;
		}
	}
	catch (const std::exception &e) {
		LOGTW_ERROR << e.what();
		return -1;
	}
	return 0;
}

