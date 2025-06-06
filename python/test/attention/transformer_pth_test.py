#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math, copy, time
import numpy as np
import torch
import torchtext
import matplotlib.pyplot as plt

class TransformerModel(torch.nn.Module):
	def __init__(self, num_tokens, d_model, num_heads, dim_feedforward, num_layers, dropout=0.5):
		super(TransformerModel, self).__init__()
		self.model_type = 'Transformer'
		self.src_mask = None
		self.d_model = d_model

		self.pos_encoder = PositionalEncoding1(d_model, dropout)
		encoder_layers = torch.nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
		self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers)
		self.encoder = torch.nn.Embedding(num_tokens, d_model)
		self.decoder = torch.nn.Linear(d_model, num_tokens)

		self.init_weights()

	def forward(self, src):
		if self.src_mask is None or self.src_mask.size(0) != src.size(0):
			device = src.device
			mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
			self.src_mask = mask

		src = self.encoder(src) * math.sqrt(self.d_model)
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src, self.src_mask)
		output = self.decoder(output)
		return output

	def init_weights(self):
		init_range = 0.1
		self.encoder.weight.data.uniform_(-init_range, init_range)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-init_range, init_range)

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

class PositionalEncoding1(torch.nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding1, self).__init__()
		self.dropout = torch.nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)  # [time-steps, batch size (= 1), PE].
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

# REF [site] >> https://pytorch.org/tutorials/beginner/transformer_tutorial.html
def transformer_tutorial():
	TEXT = torchtext.data.Field(
		tokenize=torchtext.data.utils.get_tokenizer('basic_english'),
		init_token='<sos>', eos_token='<eos>',
		lower=True
	)
	train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
	TEXT.build_vocab(train_txt)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def batchify(data, bsz):
		data = TEXT.numericalize([data.examples[0].text])
		# Divide the dataset into bsz parts.
		nbatch = data.size(0) // bsz
		# Trim off any extra elements that wouldn't cleanly fit (remainders).
		data = data.narrow(0, 0, nbatch * bsz)
		# Evenly divide the data across the bsz batches.
		data = data.view(bsz, -1).t().contiguous()
		return data.to(device)

	batch_size = 20
	eval_batch_size = 10
	train_data = batchify(train_txt, batch_size)
	val_data = batchify(val_txt, eval_batch_size)
	test_data = batchify(test_txt, eval_batch_size)

	# Generate input and target sequence.
	bptt = 35
	def get_batch(source, i):
		seq_len = min(bptt, len(source) - 1 - i)
		data = source[i:i + seq_len]  # [time steps, batch size]
		target = source[i + 1:i + 1 + seq_len].view(-1)
		return data, target

	# Initiate an instance.
	num_tokens = len(TEXT.vocab.stoi)  # The size of vocabulary.
	d_model = 200  # Embedding dimension.
	dim_feedforward = 200  # The dimension of the feed-forward network model in torch.nn.TransformerEncoder.
	num_layers = 2  # The number of torch.nn.TransformerEncoderLayer in torch.nn.TransformerEncoder.
	num_heads = 2  # The number of heads in the multi-head attention models.
	dropout = 0.2  # The dropout value.
	model = TransformerModel(num_tokens, d_model, num_heads, dim_feedforward, num_layers, dropout).to(device)

	# Run the model.
	criterion = torch.nn.CrossEntropyLoss()
	lr = 5.0  # Learning rate.
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

	def train():
		model.train()  # Turn on the train mode.
		total_loss = 0.0
		start_time = time.time()
		num_tokens = len(TEXT.vocab.stoi)
		for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
			data, targets = get_batch(train_data, i)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output.view(-1, num_tokens), targets)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()

			total_loss += loss.item()
			log_interval = 200
			if batch % log_interval == 0 and batch > 0:
				cur_loss = total_loss / log_interval
				elapsed = time.time() - start_time
				print('| epoch {:3d} | {:5d}/{:5d} batches | '
					'lr {:02.2f} | ms/batch {:5.2f} | '
					'loss {:5.2f} | ppl {:8.2f}'.format(
						epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
						elapsed * 1000 / log_interval,
						cur_loss, math.exp(cur_loss)
					)
				)
				total_loss = 0
				start_time = time.time()

	def evaluate(eval_model, data_source):
		eval_model.eval()  # Turn on the evaluation mode.
		total_loss = 0.0
		num_tokens = len(TEXT.vocab.stoi)
		with torch.no_grad():
			for i in range(0, data_source.size(0) - 1, bptt):
				data, targets = get_batch(data_source, i)
				output = eval_model(data)
				output_flat = output.view(-1, num_tokens)
				total_loss += len(data) * criterion(output_flat, targets).item()
		return total_loss / (len(data_source) - 1)

	best_val_loss = float('inf')
	epochs = 3  # The number of epochs.
	best_model = None

	for epoch in range(1, epochs + 1):
		epoch_start_time = time.time()
		train()
		val_loss = evaluate(model, val_data)
		print('-' * 89)
		print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
			'valid ppl {:8.2f}'.format(
				epoch, (time.time() - epoch_start_time),
				val_loss, math.exp(val_loss)
			)
		)
		print('-' * 89)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model = model

		scheduler.step()

#--------------------------------------------------------------------

def clones(module, N):
	"Produce N identical layers."
	return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = torch.nn.functional.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn

class LayerNorm(torch.nn.Module):
	"Construct a layernorm module (See citation for details)."

	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = torch.nn.Parameter(torch.ones(features))
		self.b_2 = torch.nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class EncoderDecoder(torch.nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many
	other models.
	"""

	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		"Take in and process masked src and target sequences."
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(torch.nn.Module):
	"Define standard linear + softmax generation step."

	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = torch.nn.Linear(d_model, vocab)

	def forward(self, x):
		return torch.nn.functional.log_softmax(self.proj(x), dim=-1)

class SublayerConnection(torch.nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""

	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = torch.nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(torch.nn.Module):
	"Encoder is made up of self-attn and feed forward (defined below)"

	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		"Follow Figure 1 (left) for connections."
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(torch.nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		"Follow Figure 1 (right) for connections."
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)

class Encoder(torch.nn.Module):
	"Core encoder is a stack of N layers"

	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		"Pass the input (and mask) through each layer in turn."
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)

class Decoder(torch.nn.Module):
	"Generic N layer decoder with masking."

	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)

class MultiHeadedAttention(torch.nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k.
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = torch.nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)

		# 1) Do all the linear projections in batch from d_model => h x d_k.
		query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			for l, x in zip(self.linears, (query, key, value))]

		# 2) Apply attention on all the projected vectors in batch.
		x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

		# 3) 'Concat' using a view and apply a final linear.
		x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)

class PositionwiseFeedForward(torch.nn.Module):
	"Implements FFN equation."

	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = torch.nn.Linear(d_model, d_ff)
		self.w_2 = torch.nn.Linear(d_ff, d_model)
		self.dropout = torch.nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))

class Embeddings(torch.nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = torch.nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding2(torch.nn.Module):
	"Implement the PE function."

	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding2, self).__init__()
		self.dropout = torch.nn.Dropout(p=dropout)

		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
		return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
	"Helper: Construct a model from hyperparameters."
	c = copy.deepcopy
	attn = MultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding2(d_model, dropout)
	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
		torch.nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		torch.nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		Generator(d_model, tgt_vocab))

	# This was important from their code.
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			torch.nn.init.xavier_uniform_(p)
	return model

class Batch:
	"Object for holding a batch of data with mask during training."

	def __init__(self, src, trg=None, pad=0):
		self.src = src
		self.src_mask = (src != pad).unsqueeze(-2)
		if trg is not None:
			self.trg = trg[:, :-1]
			self.trg_y = trg[:, 1:]
			self.trg_mask = self.make_std_mask(self.trg, pad)
			self.ntokens = (self.trg_y != pad).data.sum()

	@staticmethod
	def make_std_mask(tgt, pad):
		"Create a mask to hide padding and future words."
		tgt_mask = (tgt != pad).unsqueeze(-2)
		tgt_mask = tgt_mask & torch.autograd.Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
		return tgt_mask

def run_epoch(data_iter, model, loss_compute):
	"Standard Training and Logging Function"
	start = time.time()
	total_tokens = 0
	total_loss = 0
	tokens = 0
	for i, batch in enumerate(data_iter):
		out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
		loss = loss_compute(out, batch.trg_y, batch.ntokens)
		total_loss += loss
		total_tokens += batch.ntokens
		tokens += batch.ntokens
		if i % 50 == 1:
			elapsed = time.time() - start
			print('Epoch Step: %d Loss: %f Tokens per Sec: %f' % (i, loss / batch.ntokens, tokens / elapsed))
			start = time.time()
			tokens = 0
	return total_loss / total_tokens

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
	"Keep augmenting batch and calculate total number of tokens + padding."
	global max_src_in_batch, max_tgt_in_batch
	if count == 1:
		max_src_in_batch = 0
		max_tgt_in_batch = 0
	max_src_in_batch = max(max_src_in_batch, len(new.src))
	max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
	src_elements = count * max_src_in_batch
	tgt_elements = count * max_tgt_in_batch
	return max(src_elements, tgt_elements)

class NoamOpt:
	"Optim wrapper that implements rate."

	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0

	def step(self):
		"Update parameters and rate"
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optimizer.step()

	def rate(self, step=None):
		"Implement 'lrate' above"
		if step is None:
			step = self._step
		return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
	return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(torch.nn.Module):
	"Implement label smoothing."

	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = torch.nn.KLDivLoss(size_average=False)
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None

	def forward(self, x, target):
		assert x.size(1) == self.size
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 2))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0
		mask = torch.nonzero(target.data == self.padding_idx)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist
		return self.criterion(x, torch.autograd.Variable(true_dist, requires_grad=False))

def subsequent_mask_test():
	mask = subsequent_mask(20)

	plt.figure(figsize=(5,5))
	plt.imshow(mask[0])
	plt.show()

def positional_encoding_test():
	d_model, num_terms = 20, 50
	#d_model, num_terms = 128, 50
	pe = PositionalEncoding2(d_model, dropout=0)

	x = torch.autograd.Variable(torch.zeros(1, num_terms, d_model))
	y = pe.forward(x)
	print('x =', x.shape)
	print('y =', y.shape)

	plt.figure(figsize=(15, 5))
	plt.plot(np.arange(num_terms), y[0, :, 4:8].data.numpy())
	plt.legend(['dim %d' % p for p in [4, 5, 6, 7]])
	plt.show()

	plt.pcolormesh(y[0].data.numpy(), cmap='RdBu')
	plt.xlabel('Depth')
	plt.xlim((0, d_model))
	plt.ylabel('Position')
	plt.colorbar()
	plt.show()

def make_model_test():
	# Small example model.
	tmp_model = make_model(10, 10, 2)

def noam_opt_test():
	# Three settings of the lrate hyperparameters.
	opts = [
		NoamOpt(512, 1, 4000, None),
		NoamOpt(512, 1, 8000, None),
		NoamOpt(256, 1, 4000, None)
	]

	plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
	plt.legend(['512:4000', '512:8000', '256:4000'])
	plt.show()

def label_smoothing_test():
	# Example of label smoothing.
	crit = LabelSmoothing(5, 0, 0.4)
	predict = torch.FloatTensor(
		[[0, 0.2, 0.7, 0.1, 0],
		[0, 0.2, 0.7, 0.1, 0],
		[0, 0.2, 0.7, 0.1, 0]]
	)
	v = crit(torch.autograd.Variable(predict.log()), torch.autograd.Variable(torch.LongTensor([2, 1, 0])))

	# Show the target distributions expected by the system.
	plt.imshow(crit.true_dist)
	plt.show()

def loss_test():
	crit = LabelSmoothing(5, 0, 0.1)
	def loss(x):
		d = x + 3 * 1
		predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],])
		#print(predict)
		return crit(torch.autograd.Variable(predict.log()), torch.autograd.Variable(torch.LongTensor([1]))).data.item()

	plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
	plt.show()

def data_gen(V, batch, nbatches):
	"Generate random data for a src-tgt copy task."
	for i in range(nbatches):
		data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
		data[:, 0] = 1
		src = torch.autograd.Variable(data, requires_grad=False)
		tgt = torch.autograd.Variable(data, requires_grad=False)
		yield Batch(src, tgt, 0)

class SimpleLossCompute:
	"A simple loss compute and train function."

	def __init__(self, generator, criterion, opt=None):
		self.generator = generator
		self.criterion = criterion
		self.opt = opt

	def __call__(self, x, y, norm):
		x = self.generator(x)
		loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
		loss.backward()
		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad()
		return loss.data.item() * norm

def greedy_decode(model, src, src_mask, max_len, start_symbol):
	memory = model.encode(src, src_mask)
	ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
	for i in range(max_len-1):
		out = model.decode(memory, src_mask, torch.autograd.Variable(ys), torch.autograd.Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
		prob = model.generator(out[:, -1])
		_, next_word = torch.max(prob, dim = 1)
		next_word = next_word.data[0]
		ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
	return ys

# REF [site] >> http://nlp.seas.harvard.edu/2018/04/03/attention.html
#	"The Annotated Transformer - Attention Is All You Need".
def first_toy_example():
	# Train the simple copy task.
	V = 11
	criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
	model = make_model(V, V, N=2)
	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	for epoch in range(10):
		model.train()
		run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
		model.eval()
		print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))

	#--------------------
	model.eval()
	src = torch.autograd.Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
	src_mask = torch.autograd.Variable(torch.ones(1, 1, 10))
	print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

class MyIterator(torchtext.data.Iterator):
	def create_batches(self):
		if self.train:
			def pool(d, random_shuffler):
				for p in torchtext.data.batch(d, self.batch_size * 100):
					p_batch = torchtext.data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
					for b in random_shuffler(list(p_batch)):
						yield b
			self.batches = pool(self.data(), self.random_shuffler)
		else:
			self.batches = []
			for b in torchtext.data.batch(self.data(), self.batch_size, self.batch_size_fn):
				self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
	"Fix order in torchtext to match ours"
	src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
	return Batch(src, trg, pad_idx)

class MultiGPULossCompute:
	"A multi-gpu loss compute and train function."

	def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
		# Send out to different gpus.
		self.generator = generator
		self.criterion = torch.nn.parallel.replicate(criterion, devices=devices)
		self.opt = opt
		self.devices = devices
		self.chunk_size = chunk_size

	def __call__(self, out, targets, normalize):
		total = 0.0
		generator = torch.nn.parallel.replicate(self.generator, devices=self.devices)
		out_scatter = torch.nn.parallel.scatter(out, target_gpus=self.devices)
		out_grad = [[] for _ in out_scatter]
		targets = torch.nn.parallel.scatter(targets, target_gpus=self.devices)

		# Divide generating into chunks.
		chunk_size = self.chunk_size
		for i in range(0, out_scatter[0].size(1), chunk_size):
			# Predict distributions.
			out_column = [[torch.autograd.Variable(o[:, i:i+chunk_size].data, requires_grad=self.opt is not None)] for o in out_scatter]
			gen = torch.nn.parallel.parallel_apply(generator, out_column)

			# Compute loss.
			y = [(g.contiguous().view(-1, g.size(-1)), t[:, i:i+chunk_size].contiguous().view(-1)) for g, t in zip(gen, targets)]
			loss = torch.nn.parallel.parallel_apply(self.criterion, y)

			# Sum and normalize loss.
			l = torch.nn.parallel.gather(loss, target_device=self.devices[0])
			l = l.sum()[0] / normalize
			total += l.data[0]

			# Backprop loss to output of transformer.
			if self.opt is not None:
				l.backward()
				for j, l in enumerate(loss):
					out_grad[j].append(out_column[j][0].grad.data.clone())

		# Backprop all loss through transformer.
		if self.opt is not None:
			out_grad = [torch.autograd.Variable(torch.cat(og, dim=1)) for og in out_grad]
			o1 = out
			o2 = torch.nn.parallel.gather(out_grad, target_device=self.devices[0])
			o1.backward(gradient=o2)
			self.opt.step()
			self.opt.optimizer.zero_grad()
		return total * normalize

# Model averaging:
#	The paper averages the last k checkpoints to create an ensembling effect.
def average(model, models):
	"Average models into model"
	for ps in zip(*[m.params() for m in [model] + models]):
		ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))

# REF [site] >> http://nlp.seas.harvard.edu/2018/04/03/attention.html
#	"The Annotated Transformer - Attention Is All You Need".
def real_world_example():
	if True:
		import spacy

		# Download spaCy models:
		#	python -m spacy download en de
		spacy_de = spacy.load('de')
		spacy_en = spacy.load('en')

		def tokenize_de(text):
			return [tok.text for tok in spacy_de.tokenizer(text)]
		def tokenize_en(text):
			return [tok.text for tok in spacy_en.tokenizer(text)]

		BOS_WORD = '<s>'
		EOS_WORD = '</s>'
		BLANK_WORD = '<blank>'
		SRC = torchtext.data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
		TGT = torchtext.data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)

		MAX_LEN = 100
		train, val, test = torchtext.datasets.IWSLT.splits(
			exts=('.de', '.en'), fields=(SRC, TGT),
			filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
		MIN_FREQ = 2
		SRC.build_vocab(train.src, min_freq=MIN_FREQ)
		TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

	# Create our model, criterion, optimizer, data iterators, and paralelization.
	#devices = [0, 1, 2, 3]  # GPUs to use.
	devices = [0, 1]  # GPUs to use.
	if True:
		pad_idx = TGT.vocab.stoi['<blank>']
		model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
		model.cuda()
		criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
		criterion.cuda()
		BATCH_SIZE = 12000
		train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
			repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
			batch_size_fn=batch_size_fn, train=True)
		valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
			repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
			batch_size_fn=batch_size_fn, train=False)
		model_par = torch.nn.DataParallel(model, device_ids=devices)

	# Train the model.
	if False:
		model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
			torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
		for epoch in range(10):
			model_par.train()
			run_epoch((rebatch(pad_idx, b) for b in train_iter),
				model_par,
				MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))
			model_par.eval()
			loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
				model_par,
				MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))
			print(loss)
	else:
		# Download a pretrained model:
		#	wget https://s3.amazonaws.com/opennmt-models/iwslt.pt

		model = torch.load('./iwslt.pt')

	# Shared Embeddings:
	#	When using byte pair encoding (BPE) with shared vocabulary we can share the same weight vectors between the source / target / generator.
	# REF [paper] >> "Using the Output Embedding to Improve Language Models", arXiv 2017.
	# TODO [check] >> Is it the right position?
	if False:
		model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
		model.generator.lut.weight = model.tgt_embed[0].lut.weight

	# Decode the model to produce a set of translations.
	for i, batch in enumerate(valid_iter):
		src = batch.src.transpose(0, 1)[:1]
		src_mask = (src != SRC.vocab.stoi['<blank>']).unsqueeze(-2)
		src, src_mask = src.cuda(), src_mask.cuda()
		out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi['<s>'])
		print('Translation:', end='\t')
		for i in range(1, out.size(1)):
			sym = TGT.vocab.itos[out[0, i]]
			if sym == '</s>': break
			print(sym, end=' ')
		print()
		print('Target:', end='\t')
		for i in range(1, batch.trg.size(0)):
			sym = TGT.vocab.itos[batch.trg.data[i, 0]]
			if sym == '</s>': break
			print(sym, end=' ')
		print()
		break

# REF [site] >> http://nlp.seas.harvard.edu/2018/04/03/attention.html
#	"The Annotated Transformer - Attention Is All You Need".
def OpenNMT_example():
	# Download a pretrained model:
	#	wget https://s3.amazonaws.com/opennmt-models/en-de-model.pt

	model, SRC, TGT = torch.load('./en-de-model.pt')

	model.eval()
	sent = '▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver'.split()
	src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
	src = torch.autograd.Variable(src)
	src_mask = (src != SRC.stoi['<blank>']).unsqueeze(-2)
	out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.stoi['<s>'])
	print('Translation:', end='\t')
	trans = '<s> '
	for i in range(1, out.size(1)):
		sym = TGT.itos[out[0, i]]
		if sym == '</s>': break
		trans += sym + ' '
	print(trans)

	#--------------------
	# Visualize attention.

	import seaborn
	seaborn.set_context(context='talk')

	tgt_sent = trans.split()
	def draw(data, x, y, ax):
		seaborn.heatmap(data,
			xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
			cbar=False, ax=ax)

	for layer in range(1, 6, 2):
		fig, axs = plt.subplots(1, 4, figsize=(20, 10))
		print('Encoder Layer', layer + 1)
		for h in range(4):
			draw(model.encoder.layers[layer].self_attn.attn[0, h].data,
				sent, sent if h == 0 else [], ax=axs[h])
		plt.show()

	for layer in range(1, 6, 2):
		fig, axs = plt.subplots(1, 4, figsize=(20, 10))
		print('Decoder Self Layer', layer+1)
		for h in range(4):
			draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)],
				tgt_sent, tgt_sent if h == 0 else [], ax=axs[h])
		plt.show()
		print('Decoder Src Layer', layer + 1)
		fig, axs = plt.subplots(1, 4, figsize=(20, 10))
		for h in range(4):
			draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)],
				sent, tgt_sent if h == 0 else [], ax=axs[h])
		plt.show()

def main():
	# Use only encoders.
	#	Refer to "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv 2019.
	transformer_tutorial()

	# Use only decoders.
	#	Refer to "Improving Language Understanding by Generative Pre-Training", 2018 (GPT).

	# Use both encoders and decoders.
	#	Refer to "Attention Is All You Need", NIPS 2017 (Transformer).

	#--------------------
	#subsequent_mask_test()
	#positional_encoding_test()
	#make_model_test()
	#noam_opt_test()
	#label_smoothing_test()
	#loss_test()

	#first_toy_example()
	#real_world_example()

	# TODO [fix] >> KeyError: 'unk_index'
	#	https://forum.opennmt.net/t/issue-loading-pretrained-models-for-machine-translation/2982/5
	#	https://github.com/pytorch/text/pull/531
	#	Use a previous torchtext:
	#		pip install --upgrade git+https://github.com/pytorch/text@a63e45e569aa61ca238cca41c49e01dda34466b0
	#OpenNMT_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
