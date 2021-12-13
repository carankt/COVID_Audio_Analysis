#!/bin/bash
# in this you have to give the path the folder where your final 
# summary matrix are there

folder_name="/Users/rohitkumar/Desktop/Course_Work/ASP/COSWARA/MuDiCov/speech_feats/results_lr/"

ls $folder_name | grep summary | sort > temp


while read -r line;
do
echo $line
cat $folder_name$line
echo " "
done < temp > performance_metrix_speech.txt

rm temp
