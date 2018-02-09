
INFO
====

LOGGING:
	The VoiceBridge logging mechanism works as follows:
	- use LOGTW_INFO, LOGTW_WARNING, LOGTW_ERROR, LOGTW_FATALERROR, LOGTW_DEBUG to write to the global log file and to the
		screen in case of a console app (std::cout).
	- number formating must be set for each invocation of the log object or macro. Example: LOGTW_INFO << "double " << std::fixed << std::setprecision(2) << 25.2365874;
	- old Kaldi messages: 
		- KALDI_ERR		=> goes to LOGTW_ERROR
		- KALDI_WARN	=> goes to LOGTW_WARNING
		- KALDI_LOG		=> goes to local log file or to LOGTW_INFO if local log is not defined
		- KALDI_VLOG(v) => goes to local log file or to LOGTW_INFO if local log is not defined
		  , where local log file means the log file to store spesific information from special modules 
		  e.g. training.
