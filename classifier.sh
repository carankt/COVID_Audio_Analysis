# this bash file will run the classifier on each feature of feature set with all its functionals and give the performance of all those feats in a text file

# NOTE: PLS CHANGE THE OUTPUT TEXT FILE WHEN YOU RUN THE CODE EACH TIME
# Basically change the modality and output text file

listsdir=LISTS
datadir=data
featsdir=feats

echo "==== Audio signals ====="
# mention the directory name where you wanted to put your results

# Breathing
mkdir -p breathing_feats

# Cough
mkdir -p cough_feats

# Speech
mkdir -p speech_feats

# lr linearSVM rbfSVM we can use all these three classifier to see how the performance go

for classifier in lr;do 
	
	# Breathing
	python local/classifier_with_feats.py breathing $classifier $datadir $featsdir/breathing.csv breathing_feats
	
	# Cough
	python local/classifier_with_feats.py cough $classifier $datadir $featsdir/cough.csv cough_feats
	
	# Speech
	python local/classifier_with_feats.py speech  $classifier $datadir $featsdir/speech.csv speech_feats
done


