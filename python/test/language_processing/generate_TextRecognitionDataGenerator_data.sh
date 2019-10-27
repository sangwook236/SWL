#!/usr/bin/env bash

if [ $# -eq 2 ]
then
	if [ "$1" = "kr" ]
	then
		LANG=kr
		CMD="run_sangwook.py -l ${LANG}"

		#FONT_SIZES=48
		FONT_SIZES=(32 48 64 80 96)
	else
		LANG=en
		CMD=run.py

		#FONT_SIZES=24
		FONT_SIZES=(16 24 32 40 48)
	fi
	VERSION=$(printf %02d $2)

	TRAIN_DATA_DIR=text_line_samples_${LANG}_train_v${VERSION}
	TEST_DATA_DIR=text_line_samples_${LANG}_test_v${VERSION}

	for FONT_SIZE in ${FONT_SIZES[*]}
	do
		for WORDS in {1..3}
		do
			python ${CMD} -c 10000 -f ${FONT_SIZE} -w ${WORDS} -b 1 -t 8 --name_format 2 --output_dir ${TRAIN_DATA_DIR}/dic_h${FONT_SIZE}_w${WORDS}
			python ${CMD} -c 10000 -f ${FONT_SIZE} -w ${WORDS} -rs -b 1 -t 8 --name_format 2 --output_dir ${TRAIN_DATA_DIR}/rs_h${FONT_SIZE}_w${WORDS}
			python ${CMD} -c 1000 -f ${FONT_SIZE} -w ${WORDS} -b 1 -t 8 --name_format 2 --output_dir ${TEST_DATA_DIR}/dic_h${FONT_SIZE}_w${WORDS}
			python ${CMD} -c 1000 -f ${FONT_SIZE} -w ${WORDS} -rs -b 1 -t 8 --name_format 2 --output_dir ${TEST_DATA_DIR}/rs_h${FONT_SIZE}_w${WORDS}
		done
	done

	# Run merge_generated_data_directories() in ./TextRecognitionDataGenerator_data_test.py to genereate labels.txt.
else
	echo "Usage: $0 lang dataset-version"
	echo "          lang = en or kr"
fi
