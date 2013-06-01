%----------------------------------------------------------
% [ref] Camera Calibration Toolbox for Matlab
%	http://www.vision.caltech.edu/bouguetj/calib_doc/
%	http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
%	http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example.html
%	http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html

%----------------------------------------------------------
%addpath('D:\work_center\sw_dev\matlab\rnd\src\machine_vision\camera_calibration_toolbox_for_matlab\toolbox_calib');

%----------------------------------------------------------
% IR camera calibration (left camera)
%	the coordinate system of the IR camera is taken as the world coordinate system.

calib_gui

% rename result files:
% calib_data.mat -> calib_data_ir.mat
% Calib_Results.mat -> Calib_Results_ir.mat
% Calib_Results.m -> Calib_Results_ir.m

%----------------------------------------------------------
% RGBA camera calibration (right camera)

calib_gui

% rename result files:
% calib_data.mat -> calib_data_rgba.mat
% Calib_Results.mat -> Calib_Results_rgba.mat
% Calib_Results.m -> Calib_Results_rgba.m

%----------------------------------------------------------
% stereo calibration

stereo_gui

% result file:
% Calib_Results_stereo.mat
