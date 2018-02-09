==========================================================================
    Windows DLL for Kaldi Copyright 2017-2018 Zoltan Somogyi (AI-TOOLKIT)
==========================================================================

TODO:
	

Low Priority:
	- Check if should add back the multiple pronunciations to the lexicon generated in the beginning. Not
	  sure yet if this has any good effect on the accuracy.

	- Use festival's Text module to normalize text; e.g. numbers will be converted to text 1 = 'one', dates will be
	  converted to text, etc. At the moment it is English only but it would not be difficult to make it multilingual.
	  This should be a pre-processing step for getting the language transcriptions for a WAV file/sentence.
	  The input transcriptions should only contain normalized text!
	  
	  [The Festival Speech Synthesis System, Centre for Speech Technology Research, University of Edinburgh, UK
		free to use in commercial software also; MIT like license.]



PROGRAMMING INFO:
	- 'boost::regex_match' can not be used for testing for a partial match because it only returns true if the whole
				word is matched! 

COMPILE:
	- Do not switch on whole program optimization because then the linker will not have enough virtual space to link
	  everything together. Whole programm optimization results only in 3-4% speed improvement.
	
