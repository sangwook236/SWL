#!/usr/bin/env bash

DATA_DIR=opennmt_im2txt
IMAGE_DIR=${DATA_DIR}/images
PREPROCESSED_DATA_PREFIX=${DATA_DIR}/preprocessed_data
MODEL_PREFIX=${DATA_DIR}/model
IMAGE_CHANNEL=3
MAX_TEXT_LEN=100
BATCH_SIZE=128
TRAIN_STEPS_PER_DATASET=10000
GPU=0
DATA_TYPES=("simple_word" "random_word" "file_based_word" "simple_text_line")
#DATA_TYPES=("simple_word" "random_word" "file_based_word")

for id in {1..10}
do
	DATA_TYPE=${DATA_TYPES[$(($RANDOM % ${#DATA_TYPES[@]}))]}

	# Generate a new dataset.
	echo ">>>>>>>>>> Generate a new dataset: $DATA_TYPE (Iteration: #$id)."
	python generate_data_for_opennmt.py \
		--data_type ${DATA_TYPE} \
		--data_dir ${DATA_DIR} \
		--image_dir ${IMAGE_DIR} \
		--image_shape "64x640x${IMAGE_CHANNEL}" \
		--max_text_len ${MAX_TEXT_LEN} \
		--batch_size ${BATCH_SIZE}

	# Preprocess.
	echo ">>>>>>>>>> Preprocess: $id."
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
		-image_channel_size ${IMAGE_CHANNEL} \
		-overwrite

	# Train.
	echo ">>>>>>>>>> Train: $DATA_TYPE (Iteration: #$id)."
	TRAIN_STEPS=$((TRAIN_STEPS_PER_DATASET * ${id}))
	MODEL_FILEPATH=${MODEL_PREFIX}_step_${TRAIN_STEPS}.pt

	if [ $id -eq 1 ]
	then
		PRETRAINED_MODEL_FILEPATH=
	else
		PRETRAINED_MODEL_FILEPATH=${MODEL_PREFIX}_step_$((${TRAIN_STEPS_PER_DATASET} * (${id} - 1))).pt
	fi
	CUDA_VISIBLE_DEVICES=${GPU} onmt_train -model_type img \
		-train_from "${PRETRAINED_MODEL_FILEPATH}" \
		-data ${PREPROCESSED_DATA_PREFIX} \
		-save_model ${MODEL_PREFIX} \
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
	echo ">>>>>>>>>> Translate: $DATA_TYPE (Iteration: #$id)."
	onmt_translate -data_type img \
		-model ${MODEL_FILEPATH} \
		-src_dir ${IMAGE_DIR} \
		-src ${DATA_DIR}/src-val.txt \
		-output ${DATA_DIR}/pred.txt \
		-max_length ${MAX_TEXT_LEN} \
		-beam_size 5 \
		-gpu ${GPU} \
		-verbose

	# Delete the dataset.
	echo ">>>>>>>>>> Delete the dataset: $DATA_TYPE (Iteration: #$id)."
	#rm -Rf ${IMAGE_DIR} ${DATA_DIR}
	#rm -Rf ${IMAGE_DIR}
done
