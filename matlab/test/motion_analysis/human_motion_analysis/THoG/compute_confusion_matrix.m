function [confusion_matrix] = compute_confusion_matrix(confusion_matrix, logLikelihoods, testLabels, unique_gesture_count)

[max_LL_val, max_LL_idx] = max(logLikelihoods);
[max_freq, max_freq_idx] = data_max_freq(max_LL_idx);
predicted_idx = max_freq_idx;

% gesture ID: 0 (background), 1, ..., unique_gesture_count.
if predicted_idx < 0 | predicted_idx > unique_gesture_count + 1
	error(sprintf('predicted gesture index error: %d of %d', predicted_idx, unique_gesture_count));
end;

[max_freq, max_freq_idx] = data_max_freq(testLabels);
actual_idx = max_freq_idx;

% gesture ID: 0 (background), 1, ..., unique_gesture_count.
if actual_idx < 0 | actual_idx > unique_gesture_count + 1
	error(sprintf('acutal gesture index error: %d of %d', actual_idx, unique_gesture_count));
end;

confusion_matrix(predicted_idx, actual_idx) = confusion_matrix(predicted_idx, actual_idx) + 1;
