#!/bin/bash -eux

REFERENCEDIR="tests/results/gpu_error_reference_files"
RESULTSDIR="tests/results/gpu_error_results"

mkdir -p "$REFERENCEDIR"
mkdir -p "$RESULTSDIR"
mkdir -p models/

wget --no-clobber -P models/ https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s5832081920-d3223508649.bin.gz
wget --no-clobber -P models/ https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
wget --no-clobber -P models/ https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s8326494464-d4628051565.bin.gz
wget --no-clobber -P models/ https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz
wget --no-clobber -P models/ https://media.katagotraining.org/uploaded/networks/models_extra/b5c192nbt-v16test.bin.gz

MODEL1=tests/models/run4-s67105280-d24430742-b6c96.txt.gz  # version 3
MODEL2=tests/models/grun50-b6c96-s156348160-d118286860.txt.gz  # version 4
MODEL3=tests/models/g103-b6c96-s103408384-d26419149.txt.gz  # version 5
MODEL4=tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz  # version 8
MODEL5=models/kata1-b18c384nbt-s5832081920-d3223508649.bin.gz  # version 11
MODEL6=models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz  # version 14
MODEL7=models/kata1-b28c512nbt-s8326494464-d4628051565.bin.gz  # version 15
MODEL8=models/b18c384nbt-humanv0.bin.gz # human SL model
MODEL9=models/b5c192nbt-v16test.bin.gz  # version 16 model very slightly trained

MODELBASE1=$(basename "$MODEL1")
MODELBASE2=$(basename "$MODEL2")
MODELBASE3=$(basename "$MODEL3")
MODELBASE4=$(basename "$MODEL4")
MODELBASE5=$(basename "$MODEL5")
MODELBASE6=$(basename "$MODEL6")
MODELBASE7=$(basename "$MODEL7")
MODELBASE8=$(basename "$MODEL8")
MODELBASE9=$(basename "$MODEL9")

./katago testgpuerror -model "$MODEL1" -config configs/gtp_example.cfg -boardsize 9 \
         -override-config "requireMaxBoardSize=True" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE1"_size9.txt | tee "$RESULTSDIR"/"$MODELBASE1"_size9.txt
./katago testgpuerror -model "$MODEL1" -config configs/gtp_example.cfg -boardsize 19 \
         -override-config "requireMaxBoardSize=False, maxBatchSize=16" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE1"_size19.txt | tee "$RESULTSDIR"/"$MODELBASE1"_size19.txt

./katago testgpuerror -model "$MODEL2" -config configs/gtp_example.cfg -boardsize 13 \
         -override-config "requireMaxBoardSize=False" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE2"_size13.txt | tee "$RESULTSDIR"/"$MODELBASE2"_size13.txt
./katago testgpuerror -model "$MODEL2" -config configs/gtp_example.cfg -boardsize 19 \
         -override-config "requireMaxBoardSize=True, maxBatchSize=19" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE2"_size19.txt | tee "$RESULTSDIR"/"$MODELBASE2"_size19.txt

./katago testgpuerror -model "$MODEL3" -config configs/gtp_example.cfg -boardsize 9 \
         -override-config "requireMaxBoardSize=False, maxBatchSize=32" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE3"_size9.txt | tee "$RESULTSDIR"/"$MODELBASE3"_size9.txt
./katago testgpuerror -model "$MODEL3" -config configs/gtp_example.cfg -boardsize 19 \
         -override-config "requireMaxBoardSize=False, maxBatchSize=2" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE3"_size19.txt | tee "$RESULTSDIR"/"$MODELBASE3"_size19.txt

./katago testgpuerror -model "$MODEL4" -config configs/gtp_example.cfg -boardsize 9 \
         -override-config "requireMaxBoardSize=True, maxBatchSize=3" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE4"_size9.txt | tee "$RESULTSDIR"/"$MODELBASE4"_size9.txt
./katago testgpuerror -model "$MODEL4" -config configs/gtp_example.cfg -boardsize 13 \
         -override-config "requireMaxBoardSize=False, maxBatchSize=27" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE4"_size13.txt | tee "$RESULTSDIR"/"$MODELBASE4"_size13.txt
./katago testgpuerror -model "$MODEL4" -config configs/gtp_example.cfg -boardsize 19 \
         -override-config "requireMaxBoardSize=False" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE4"_size19.txt | tee "$RESULTSDIR"/"$MODELBASE4"_size19.txt
./katago testgpuerror -model "$MODEL4" -config configs/gtp_example.cfg -boardsize 10x14 \
         -override-config "requireMaxBoardSize=True, maxBatchSize=13" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE4"_size10x14.txt | tee "$RESULTSDIR"/"$MODELBASE4"_size10x14.txt
./katago testgpuerror -model "$MODEL4" -config configs/gtp_example.cfg -boardsize rectangle \
         -override-config "requireMaxBoardSize=False" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE4"_sizerect.txt | tee "$RESULTSDIR"/"$MODELBASE4"_sizerect.txt

./katago testgpuerror -model "$MODEL4" -config configs/gtp_example.cfg -boardsize 13 \
         -override-config "requireMaxBoardSize=False,maxBoardXSizeForNNBuffer=18,maxBoardYSizeForNNBuffer=19" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE4"_size13_rectbuffer.txt | tee "$RESULTSDIR"/"$MODELBASE4"_size13_rectbuffer.txt

./katago testgpuerror -model "$MODEL5" -config configs/gtp_example.cfg -boardsize rectangle \
         -override-config "requireMaxBoardSize=False" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE5"_sizerect.txt | tee "$RESULTSDIR"/"$MODELBASE5"_sizerect.txt
./katago testgpuerror -model "$MODEL5" -config configs/gtp_example.cfg -boardsize 19 \
         -override-config "requireMaxBoardSize=True, maxBatchSize=16" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE5"_size19.txt | tee "$RESULTSDIR"/"$MODELBASE5"_size19.txt

./katago testgpuerror -model "$MODEL6" -config configs/gtp_example.cfg -boardsize 9 \
         -override-config "requireMaxBoardSize=True" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE6"_size9.txt | tee "$RESULTSDIR"/"$MODELBASE6"_size9.txt
./katago testgpuerror -model "$MODEL6" -config configs/gtp_example.cfg -boardsize 13 \
         -override-config "requireMaxBoardSize=False, maxBatchSize=28" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE6"_size13.txt | tee "$RESULTSDIR"/"$MODELBASE6"_size13.txt
./katago testgpuerror -model "$MODEL6" -config configs/gtp_example.cfg -boardsize 19 \
         -override-config "requireMaxBoardSize=False, maxBatchSize=8" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE6"_size19.txt | tee "$RESULTSDIR"/"$MODELBASE6"_size19.txt
./katago testgpuerror -model "$MODEL6" -config configs/gtp_example.cfg -boardsize 10x14 \
         -override-config "requireMaxBoardSize=False, maxBatchSize=15" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE6"_size10x14.txt | tee "$RESULTSDIR"/"$MODELBASE6"_size10x14.txt
./katago testgpuerror -model "$MODEL6" -config configs/gtp_example.cfg -boardsize rectangle \
         -override-config "requireMaxBoardSize=False" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE6"_sizerect.txt | tee "$RESULTSDIR"/"$MODELBASE6"_sizerect.txt

./katago testgpuerror -model "$MODEL7" -config configs/gtp_example.cfg -boardsize 9 \
         -override-config "requireMaxBoardSize=False, maxBatchSize=4" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE7"_size9.txt | tee "$RESULTSDIR"/"$MODELBASE7"_size9.txt
./katago testgpuerror -model "$MODEL7" -config configs/gtp_example.cfg -boardsize 13 \
         -override-config "requireMaxBoardSize=True, maxBatchSize=29" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE7"_size13.txt | tee "$RESULTSDIR"/"$MODELBASE7"_size13.txt
./katago testgpuerror -model "$MODEL7" -config configs/gtp_example.cfg -boardsize 19 \
         -override-config "requireMaxBoardSize=True" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE7"_size19.txt | tee "$RESULTSDIR"/"$MODELBASE7"_size19.txt
./katago testgpuerror -model "$MODEL7" -config configs/gtp_example.cfg -boardsize 10x14 \
         -override-config "requireMaxBoardSize=True, maxBatchSize=5" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE7"_size10x14.txt | tee "$RESULTSDIR"/"$MODELBASE7"_size10x14.txt
./katago testgpuerror -model "$MODEL7" -config configs/gtp_example.cfg -boardsize rectangle \
         -override-config "requireMaxBoardSize=False" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE7"_sizerect.txt | tee "$RESULTSDIR"/"$MODELBASE7"_sizerect.txt

./katago testgpuerror -model "$MODEL7" -config configs/gtp_example.cfg -boardsize 9 \
         -override-config "requireMaxBoardSize=False,maxBoardXSizeForNNBuffer=16,maxBoardYSizeForNNBuffer=11, maxBatchSize=9" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE7"_size9_rectbuffer.txt | tee "$RESULTSDIR"/"$MODELBASE7"_size9_rectbuffer.txt

./katago testgpuerror -model "$MODEL8" -config configs/gtp_example.cfg -boardsize rectangle \
         -override-config "requireMaxBoardSize=False, maxBatchSize=11" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE8"_sizerect.txt | tee "$RESULTSDIR"/"$MODELBASE8"_sizerect.txt
./katago testgpuerror -model "$MODEL8" -config configs/gtp_example.cfg -boardsize 19 \
         -override-config "requireMaxBoardSize=True" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE8"_size19.txt | tee "$RESULTSDIR"/"$MODELBASE8"_size19.txt

./katago testgpuerror -model "$MODEL9" -config configs/gtp_example.cfg -boardsize rectangle \
         -override-config "requireMaxBoardSize=False, maxBatchSize=11" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE9"_sizerect.txt | tee "$RESULTSDIR"/"$MODELBASE9"_sizerect.txt
./katago testgpuerror -model "$MODEL9" -config configs/gtp_example.cfg -boardsize 19 \
         -override-config "requireMaxBoardSize=True" \
         -reference-file "$REFERENCEDIR"/"$MODELBASE9"_size19.txt | tee "$RESULTSDIR"/"$MODELBASE9"_size19.txt


