function [ Mu, Kappa, Alpha, IterationStep ] = train_MovM_from_HoG(HoG, HoG_sequence_indexes, HoG_frame_indexes, HoG_bin_width, HoG_scale_factor, num_clusters, init_mu, init_kappa, init_alpha, max_step, tol, resultant_file_path)

numSeqs = length(HoG);
Mu = cell(1, numSeqs);
Kappa = cell(1, numSeqs);
Alpha = cell(1, numSeqs);
IterationStep = cell(1, numSeqs);

if HoG_sequence_indexes(1) > 0
    start_seq_idx = HoG_sequence_indexes(1);
else
	start_seq_idx = 1;
end;
if HoG_sequence_indexes(2) > 0
    end_seq_idx = HoG_sequence_indexes(2);
else
	end_seq_idx = numSeqs;
end;

if start_seq_idx > end_seq_idx || start_seq_idx < 1 || start_seq_idx > numSeqs || end_seq_idx < 1 || end_seq_idx > numSeqs
    eror(sprintf('[SWL] start and/or end indexes of sequence are incorrect - start: %d, end: %d, #sequences: %d)', start_seq_idx, end_seq_idx, numSeqs));
end;

for ii = start_seq_idx:end_seq_idx
    numFrames = size(HoG{ii}, 2);
	Mu{ii} = zeros(num_clusters, numFrames);
	Kappa{ii} = zeros(num_clusters, numFrames);
	Alpha{ii} = zeros(num_clusters, numFrames);
	IterationStep{ii} = zeros(1, numFrames);

	if start_seq_idx == ii && HoG_frame_indexes(1) > 0
	    start_frame_idx = HoG_frame_indexes(1);
	else
		start_frame_idx = 1;
	end;
	if end_seq_idx == ii && HoG_frame_indexes(2) > 0
    	end_frame_idx = HoG_frame_indexes(2);
	else
		end_frame_idx = numFrames;
	end;

    if start_frame_idx > end_frame_idx || start_frame_idx < 1 || start_frame_idx > numFrames || end_frame_idx < 1 || end_frame_idx > numFrames
        error(sprintf('[SWL] start and/or end indexes of frame are incorrect - start: %d, end: %d, #frames: %d)', start_frame_idx, end_frame_idx, numFrames));
    end;

    for jj = start_frame_idx:end_frame_idx
        if sum(HoG{ii}(:,jj)) < eps
    		step = 0;
     		Mu{ii}(:,jj) = 0;
    		Kappa{ii}(:,jj) = 0;
    		Alpha{ii}(:,jj) = 0;
    		IterationStep{ii}(1,jj) = step;
        else
            angleData = HoG_to_angle(HoG{ii}(:,jj), HoG_bin_width, HoG_scale_factor);

    		%----------------------------------------------------------
        	% approach #1: use 2-dim directional vectors.
            %[ cluster, mu_est, kappa_est, alpha_est ] = movmf_sangwook([ cos(angleData) sin(angleData) ], num_clusters);

    		%Mu{ii}(:,jj) = mu_est;
    		%Kappa{ii}(:,jj) = kappa_est;
    		%Alpha{ii}(:,jj) = alpha_est;

    		%----------------------------------------------------------
    		% approach #2: use direction angles, [rad].
            [ mu_est, kappa_est, alpha_est, step ] = em_MovM(angleData, num_clusters, init_mu, init_kappa, init_alpha, max_step, tol);

    		Mu{ii}(:,jj) = mu_est';
    		Kappa{ii}(:,jj) = kappa_est';
    		Alpha{ii}(:,jj) = alpha_est';
    		IterationStep{ii}(1,jj) = step;

    		%----------------------------------------------------------
    		% approach #3: use direction angles, [rad].
    		%[cid, alpha, mu] = circ_clust(angleData', num_clusters, false);
    	end;

        if ~isempty(resultant_file_path)
		    save(resultant_file_path, 'Mu', 'Kappa', 'Alpha', 'IterationStep');
		end;

	    sprintf('seq: %d, frame: %d, iteration: %d', ii, jj, step)
    end;
end;
