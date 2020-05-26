#!/usr/bin/env bash

DATA_DIR=opennmt_im2txt
IMAGE_DIR=${DATA_DIR}/images
PREPROCESSED_DATA_PREFIX=${DATA_DIR}/preprocessed_data
TRAINED_MODEL_PREFIX=${DATA_DIR}/model
IMAGE_CHANNEL=3
MAX_TEXT_LEN=100
BATCH_SIZE=20
TRAIN_STEPS=100000

for id in {1..10}
do
	# Generate a new dataset.
	python generate_data_for_opennmt.py \
		--data_type simple_word \
		--data_dir ${DATA_DIR} \
		--image_dir ${IMAGE_DIR} \
		--image_shape "64x640x${IMAGE_CHANNEL}" \
		--max_text_len ${MAX_TEXT_LEN} \
		--batch_size ${BATCH_SIZE}

	# Preprocess.
	onmt_preprocess -data_type img \
		-src_dir ${IMAGE_DIR} \
		-train_src ${DATA_DIR}/src-train.txt \
		-train_tgt ${DATA_DIR}/tgt-train.txt \
		-valid_src ${DATA_DIR}/src-val.txt \
		-valid_tgt ${DATA_DIR}/tgt-val.txt \
		-save_data ${PREPROCESSED_DATA_PREFIX} \
		-tgt_seq_length ${MAX_TEXT_LEN} \
		-tgt_words_min_frequency 0 \
		-shard_size 500 \
		-image_channel_size ${IMAGE_CHANNEL}

	# Train.
	if [ $id -eq 1 ]
	then
		TRAINED_MODEL_FILEPATH=
	else
		#TRAINED_MODEL_FILEPATH=${TRAINED_MODEL_PREFIX}_step_$((${TRAIN_STEPS}*(${id}-1))).pt
		TRAINED_MODEL_FILEPATH=${TRAINED_MODEL_PREFIX}_step_$((${TRAIN_STEPS})).pt
	fi
	onmt_train -model_type img \
		-train_from "${TRAINED_MODEL_FILEPATH}" \
		-data ${PREPROCESSED_DATA_PREFIX} \
		-save_model ${TRAINED_MODEL_PREFIX} \
		-gpu_ranks 0 \
		-world_size 1 \
		-batch_size ${BATCH_SIZE} \
		-train_steps ${TRAIN_STEPS} \
		-max_grad_norm 20 \
		-learning_rate 0.1 \
		-word_vec_size 80 \
		-encoder_type brnn \
		-image_channel_size ${IMAGE_CHANNEL}

	# Translate.
	onmt_translate -data_type img \
		-model ${TRAINED_MODEL_FILEPATH} \
		-src_dir ${IMAGE_DIR} \
		-src ${DATA_DIR}/src-val.txt \
		-output ${DATA_DIR}/pred.txt \
		-max_length ${MAX_TEXT_LEN} \
		-beam_size 5 \
		-gpu 0 \
		-verbose

	# Delete the dataset.
	#rm -Rf ${IMAGE_DIR} ${DATA_DIR}
	#rm -Rf ${IMAGE_DIR}
done
