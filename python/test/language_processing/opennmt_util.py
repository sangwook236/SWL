import onmt

# REF [function] >> create_greedy_search_strategy() in ${SWDT_PYTHON_HOME}/rnd/test/language_processing/opennmt_py_test.py.
def create_greedy_search_strategy(batch_size, random_sampling_topk, random_sampling_temp, min_length, max_length, block_ngram_repeat, bos_index, eos_index, pad_index, exclusion_idxs):
	replace_unk = False
	#tgt_prefix = False
	attn_debug = False

	return onmt.translate.greedy_search.GreedySearch(
		pad=pad_index, bos=bos_index, eos=eos_index,
		batch_size=batch_size,
		min_length=min_length, max_length=max_length,
		block_ngram_repeat=block_ngram_repeat,
		exclusion_tokens=exclusion_idxs,
		return_attention=attn_debug or replace_unk,
		sampling_temp=random_sampling_temp,
		keep_topk=random_sampling_topk
	)

# REF [function] >> create_beam_search_strategy() in ${SWDT_PYTHON_HOME}/rnd/test/language_processing/opennmt_py_test.py.
def create_beam_search_strategy(batch_size, scorer, beam_size, n_best, ratio, min_length, max_length, block_ngram_repeat, bos_index, eos_index, pad_index, exclusion_idxs):
	stepwise_penalty = None,
	replace_unk = False
	#tgt_prefix = False
	attn_debug = False

	return onmt.translate.beam_search.BeamSearch(
		beam_size,
		batch_size=batch_size,
		pad=pad_index, bos=bos_index, eos=eos_index,
		n_best=n_best,
		global_scorer=scorer,
		min_length=min_length, max_length=max_length,
		return_attention=attn_debug or replace_unk,
		block_ngram_repeat=block_ngram_repeat,
		exclusion_tokens=exclusion_idxs,
		stepwise_penalty=stepwise_penalty,
		ratio=ratio
	)

# REF [function] >> decode_and_generate() in ${SWDT_PYTHON_HOME}/rnd/test/language_processing/opennmt_py_test.py.
def decode_and_generate(model, decoder_in, memory_bank, batch, src_vocabs, memory_lengths, beam_size, copy_attn, tgt_vocab, tgt_unk_idx, src_map=None, step=None, batch_offset=None):
	if copy_attn:
		# Turn any copied words into UNKs.
		decoder_in = decoder_in.masked_fill(decoder_in.gt(len(tgt_vocab) - 1), tgt_unk_idx)

	# Decoder forward, takes [tgt_len, batch, nfeats] as input
	# and [src_len, batch, hidden] as memory_bank
	# in case of inference tgt_len = 1, batch = beam times batch_size
	# in case of Gold Scoring tgt_len = actual length, batch = 1 batch
	dec_out, dec_attn = model.decoder(decoder_in, memory_bank, memory_lengths=memory_lengths, step=step)

	# Generator forward.
	if not copy_attn:
		if 'std' in dec_attn:
			attn = dec_attn['std']
		else:
			attn = None
		log_probs = model.generator(dec_out.squeeze(0))
		# returns [(batch_size x beam_size) , vocab ] when 1 step
		# or [ tgt_len, batch_size, vocab ] when full sentence
	else:
		attn = dec_attn['copy']
		scores = model.generator(dec_out.view(-1, dec_out.size(2)), attn.view(-1, attn.size(2)), src_map)
		# here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
		if batch_offset is None:
			scores = scores.view(-1, batch.batch_size, scores.size(-1))
			scores = scores.transpose(0, 1).contiguous()
		else:
			scores = scores.view(-1, beam_size, scores.size(-1))
		scores = onmt.modules.copy_generator.collapse_copy_scores(
			scores,
			batch,
			tgt_vocab,
			src_vocabs,
			batch_dim=0,
			batch_offset=batch_offset
		)
		scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
		log_probs = scores.squeeze(0).log()
		# returns [(batch_size x beam_size) , vocab ] when 1 step
		# or [ tgt_len, batch_size, vocab ] when full sentence
	return log_probs, attn

# REF [function] >> translate_batch_with_strategy() in ${SWDT_PYTHON_HOME}/rnd/test/language_processing/opennmt_py_test.py.
def translate_batch_with_strategy(model, decode_strategy, src, batch_size, beam_size, unk_index, tgt_vocab, src_vocabs=[]):
	import torch

	copy_attn = False  # Fixed.
	report_align = False  # Fixed.

	parallel_paths = decode_strategy.parallel_paths  # beam_size.

	enc_states, memory_bank, src_lengths = model.encoder(src, lengths=None)
	if src_lengths is None:
		src_lengths = torch.Tensor(batch_size).type_as(memory_bank).long().fill_(memory_bank.size(0))
	model.decoder.init_state(src, memory_bank, enc_states)

	src_map, target_prefix = None, None
	fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(memory_bank, src_lengths, src_map, target_prefix)
	if fn_map_state is not None:
		model.decoder.map_state(fn_map_state)

	for step in range(decode_strategy.max_length):
		decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

		log_probs, attn = decode_and_generate(
			model,
			decoder_input,
			memory_bank,
			batch=None,  # NOTE [caution] >>
			src_vocabs=src_vocabs,
			memory_lengths=memory_lengths,
			beam_size=beam_size, copy_attn=copy_attn,
			tgt_vocab=tgt_vocab, tgt_unk_idx=unk_index,
			src_map=src_map,
			step=step,
			batch_offset=decode_strategy.batch_offset
		)

		decode_strategy.advance(log_probs, attn)
		any_finished = decode_strategy.is_finished.any()
		if any_finished:
			decode_strategy.update_finished()
			if decode_strategy.done:
				break

		select_indices = decode_strategy.select_indices

		if any_finished:
			# Reorder states.
			if isinstance(memory_bank, tuple):
				memory_bank = tuple(x.index_select(1, select_indices) for x in memory_bank)
			else:
				memory_bank = memory_bank.index_select(1, select_indices)

			memory_lengths = memory_lengths.index_select(0, select_indices)

			if src_map is not None:
				src_map = src_map.index_select(1, select_indices)

		if parallel_paths > 1 or any_finished:
			model.decoder.map_state(lambda state, dim: state.index_select(dim, select_indices))

	results = dict()
	results['scores'] = decode_strategy.scores
	results['predictions'] = decode_strategy.predictions
	results['attention'] = decode_strategy.attention
	if report_align:
		results['alignment'] = self._align_forward(batch, decode_strategy.predictions)
	else:
		results['alignment'] = [[] for _ in range(batch_size)]
	return results

#--------------------------------------------------------------------

class Rare1ImageEncoder(onmt.encoders.encoder.EncoderBase):
	def __init__(self, image_height, image_width, input_channel, output_channel, hidden_size, num_layers=2, bidirectional=True, transformer=None, feature_extractor='VGG', sequence_model='BiLSTM', num_fiducials=0):
		super().__init__()

		import torch
		from rare.modules.transformation import TPS_SpatialTransformerNetwork
		from rare.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
		from rare.modules.feature_extraction import VGG_FeatureExtractor_MixUp, RCNN_FeatureExtractor_MixUp, ResNet_FeatureExtractor_MixUp
		from rare.modules.sequence_modeling import BidirectionalLSTM

		# Transformer.
		if transformer == 'TPS':
			self.transformer = TPS_SpatialTransformerNetwork(F=num_fiducials, I_size=(image_height, image_width), I_r_size=(image_height, image_width), I_channel_num=input_channel)
		else:
			print('No transformer specified.')
			self.transformer = None

		# Feature extraction.
		if feature_extractor == 'VGG':
			self.feature_extractor = VGG_FeatureExtractor(input_channel, output_channel)
		elif feature_extractor == 'RCNN':
			self.feature_extractor = RCNN_FeatureExtractor(input_channel, output_channel)
		elif feature_extractor == 'ResNet':
			self.feature_extractor = ResNet_FeatureExtractor(input_channel, output_channel)
		else:
			raise ValueError("The argument, feature_extractor has to be one of 'VGG', 'RCNN', or 'ResNet': {}".format(feature_extractor))
		feature_extractor_output_size = output_channel  # int(image_height / 16 - 1) * output_channel.
		self.avg_pool = torch.nn.AdaptiveAvgPool2d((None, 1))  # Transform final (image_height / 16 - 1) -> 1.

		# Sequence model.
		if sequence_model == 'BiLSTM':
			#self.sequence_rnn = torch.nn.Sequential(
			#	BidirectionalLSTM(feature_extractor_output_size, hidden_size, hidden_size, batch_first=True),
			#	BidirectionalLSTM(hidden_size, hidden_size, hidden_size, batch_first=True)
			#)
			self.sequence_rnn = torch.nn.LSTM(feature_extractor_output_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
			if bidirectional:
				self.sequence_projector = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
			else:
				self.sequence_projector = torch.nn.Linear(hidden_size, hidden_size)
		else:
			print('No sequence model specified.')
			self.sequence_rnn = None

	# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
	def forward(self, src, lengths=None, device='cuda'):
		# Transform.
		if self.transformer: src = self.transformer(src, device)  # [b, c, h, w].

		# Extract features.
		visual_features = self.feature_extractor(src)  # [b, c_out, h/32, w/4-1].
		visual_features = self.avg_pool(visual_features.permute(0, 3, 1, 2))  # [b, w/4-1, c_out, 1].
		#visual_features = visual_features.permute(0, 3, 1, 2)  # [b, w/4-1, c_out, h/32].
		assert visual_features.shape[3] == 1
		visual_features = visual_features.squeeze(3)  # [b, w/4-1, c_out].
		#visual_features = visual_features.reshape(visual_features.shape[0], visual_features.shape[1], -1)  # [b, w/4-1, c_out * h/32].

		# When batch is not in the first order.
		#visual_features = visual_features.transpose(0, 1)  # [w/4-1, b, c_out * h/32].

		# Sequence model.
		# TODO [check] >> The hidden size is too small?
		#outputs, hiddens = self.sequence_rnn((visual_features, None))  # [b, w/4-1, #directions * hidden size] or [w/4-1, b, #directions * hidden size], ([#directions, b, hidden size], [#directions, b, hidden size]).
		outputs, hiddens = self.sequence_rnn(visual_features)  # [b, w/4-1, #directions * hidden size], ([#layers * #directions, b, hidden size], [#layers * #directions, b, hidden size]).
		outputs = self.sequence_projector(outputs)  # [b, w/4-1, hidden size].
		outputs = outputs.transpose(0, 1)  # [w/4-1, b, hidden size]

		return hiddens, outputs, lengths

#--------------------------------------------------------------------

class Rare2ImageEncoder(onmt.encoders.encoder.EncoderBase):
	def __init__(self, image_height, image_width, input_channel, hidden_size, num_layers, bidirectional=False, num_fiducials=0):
		super().__init__()

		import torch

		assert image_height % 16 == 0, 'image_height has to be a multiple of 16'
		self.image_height = image_height

		# Build a TPS-STN.
		if num_fiducials and num_fiducials > 0:
			import rare.modules.transformation
			self.transformer = rare.modules.transformation.TPS_SpatialTransformerNetwork(F=num_fiducials, I_size=(image_height, image_width), I_r_size=(image_height, image_width), I_channel_num=input_channel)
		else:
			self.transformer = None

		# Build a model.
		# This implementation assumes that input size is h x w.
		self.cnn = torch.nn.Sequential(
			torch.nn.Conv2d(input_channel, 64, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d(2, 2),  # 64 x h/2 x w/2.
			torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d(2, 2),  # 128 x h/4 x w/4.
			torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(True),  # 256 x h/4 x w/4.
			torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 256 x h/8 x w/4+1.
			torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),  # 512 x h/8 x w/4+1.
			torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(True), torch.nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512 x h/16 x w/4+2.
			torch.nn.Conv2d(512, 512, 2, 1, 0), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True)  # 512 x h/16-1 x w/4+1.
		)
		num_features = (image_height // 16 - 1) * 512
		#import rare.crnn_lang
		#self.rnn = torch.nn.Sequential(
		#	rare.crnn_lang.BidirectionalLSTM(num_features, hidden_size, hidden_size),
		#	rare.crnn_lang.BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
		#)
		self.sequence_rnn = torch.nn.LSTM(num_features, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=False)
		if bidirectional:
			self.sequence_projector = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
			#self.sequence_projector = torch.nn.Linear(hidden_size * 2, hidden_size)
		else:
			self.sequence_projector = torch.nn.Linear(hidden_size, hidden_size)

	# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
	def forward(self, src, lengths=None, device='cuda'):
		if self.transformer: src = self.transformer(src, device)  # [B, C, H, W].

		# Conv features.
		conv = self.cnn(src)  # [b, c_out, h/16-1, w/4+1].
		b, c, h, w = conv.size()
		#assert h == 1, 'The height of conv must be 1'
		#conv = conv.squeeze(2)  # [b, c_out, w/4+1].
		conv = conv.reshape(b, -1, w)  # [b, c_out * h/16-1, w/4+1].
		conv = conv.permute(2, 0, 1)  # [w/4+1, b, c_out * h/16-1].

		# RNN features.
		#outputs, hiddens = self.rnn((conv, None))  # [w/4+1, b, hidden size], ([#directions, b, hidden size], [#directions, b, hidden size]).
		outputs, hiddens = self.sequence_rnn(conv)  # [w/4+1, b, #directions * hidden size], ([#layers * #directions, b, hidden size], [#layers * #directions, b, hidden size]).
		outputs = self.sequence_projector(outputs)  # [w/4+1, b, hidden size].

		return hiddens, outputs, lengths

#--------------------------------------------------------------------

class AsterImageEncoder(onmt.encoders.encoder.EncoderBase):
	def __init__(self, image_height, image_width, image_channel, num_classes, hidden_size, num_fiducials=0):
		super().__init__()

		if num_fiducials and num_fiducials > 0:
			import rare.modules.transformation
			self.transformer = rare.modules.transformation.TPS_SpatialTransformerNetwork(F=num_fiducials, I_size=(image_height, image_width), I_r_size=(image_height, image_width), I_channel_num=image_channel)
		else:
			self.transformer = None

		import aster.resnet_aster
		self.aster = aster.resnet_aster.ResNet_ASTER(with_lstm=True, in_height=image_height, in_channels=image_channel, hidden_size=hidden_size)

	# REF [function] >> NMTModel.forward() in https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/models/model.py
	def forward(self, src, lengths=None, device='cuda'):
		if self.transformer: src = self.transformer(src, device)  # [B, C, H, W].

		hiddens, outputs, lengths = self.aster(src, lengths)
		outputs = outputs.transpose(0, 1)  # [B, T, F] -> [T, B, F].

		return hiddens, outputs, lengths
