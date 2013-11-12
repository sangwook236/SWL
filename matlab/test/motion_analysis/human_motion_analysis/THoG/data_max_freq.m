function [max_freq, max_val] = data_max_freq(data)

% data is a vector.

vals = unique(data);
freqs = zeros(size(vals));
for kk = 1:length(vals)
	freqs(kk) = length(find(data == vals(kk)));
end;

[max_freq, idx] = max(freqs);
max_val = vals(idx);
