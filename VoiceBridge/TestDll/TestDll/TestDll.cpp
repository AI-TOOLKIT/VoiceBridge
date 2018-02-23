/*
	Copyright 2017-present Zoltan Somogyi (AI-TOOLKIT), All Rights Reserved
	You may use this file only if you agree to the software license:
	AI-TOOLKIT Open Source Software License - Version 2.1 - February 22, 2018:
	https://ai-toolkit.blogspot.com/p/ai-toolkit-open-source-software-license.html.
	Also included with the source code distribution in AI-TOOLKIT-LICENSE.txt.
*/

#include "ExamplesUtil.h"

//test cases
int TestYesNo();
int TestLibriSpeech();

static int concurentThreadsSupported = std::thread::hardware_concurrency();


int main() 
{
	/*The Yes-No example is a very simple and fast to run example which is recommended for starting to learn VoiceBridge */

	TestYesNo();

	/*The LibriSpeech example is a real life full speech recognition example. You can very easily adapt the code for your
	own speech recognition tasks, you only need to change a very few number of lines/parameters. It is a ready to use
	recipe.*/

	//TestLibriSpeech(); 
}



