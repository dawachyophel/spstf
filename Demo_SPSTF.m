


close all
clear;
Original_image_dir  =    'inputimages/'; 
Sdir = regexp(Original_image_dir, '\', 'split');
fpath = fullfile(Original_image_dir, '*.jpg');
im_dir  = dir(fpath);
im_num = length(im_dir);

method = 'SPSTF';
write_MAT_dir = [method '/Results_Out/'];
write_sRGB_dir  = [write_MAT_dir method '/'];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir);
end
for nSig = [15 25 35 50 75]
    %% Parameters
    Par.innerIter = 2;
    Par.win = 30;
    Par.lambda1 = 0;
    Par.ps = 8;
    Par.outerIter = 10;
    Par.step = 3;
    Par.nlspini = 90;
    Par.nlspgap = 10;
    if 0 < nSig <= 20
        Par.outerIter = 8;
        Par.delta = .07;
        Par.nlspini = 70;
        Par.lambda2 = .9;
    elseif 20 < nSig <= 30
        Par.delta = .06;
        Par.lambda2 = .76;
    elseif 30 < nSig <= 40
        Par.delta = .07;
        Par.lambda2 = .78;
    elseif 40 < nSig <= 60
        Par.nlspini = 120;
        Par.nlspgap = 15;
        Par.delta = .05;
        Par.lambda2 = .72;
    elseif 60 < nSig <= 80
        Par.ps = 9;
        Par.outerIter = 14;
        Par.step = 4;
        Par.nlspini = 140;
        Par.delta = .05;
        Par.lambda2 = .68; % .66
    else
        disp('Tune the above parameters!');
    end
    % record all the results in each iteration
    Par.PSNR = zeros(Par.outerIter, im_num, 'double');
    Par.SSIM = zeros(Par.outerIter, im_num, 'double');
    T512 = [];
    T256 = [];
    for i = 1:im_num
        Par.nlsp = Par.nlspini;  % number of non-local patches
        Par.image = i;
        Par.nSig = nSig/255;
        Par.I =  im2double( imread(fullfile(Original_image_dir, im_dir(i).name)) );
        S = regexp(im_dir(i).name, '\.', 'split');
        randn('seed',0);
        Par.nim =   Par.I + Par.nSig*randn(size(Par.I));
        fprintf('%s :\n',im_dir(i).name);
        PSNR =   csnr( Par.nim*255, Par.I*255, 0, 0 );
        SSIM      =  cal_ssim( Par.nim*255, Par.I*255, 0, 0 );
        fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
        time0 = clock;
        [im_out, Par]  =  SPS_Sigma(Par);
        if size(Par.I,1) == 512
            T512 = [T512 etime(clock,time0)];
            fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
        elseif size(Par.I,1) ==256
            T256 = [T256 etime(clock,time0)];
            fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
        end
        
        %% Transform filtering 
        sigma_s = 0.1;
        sigma_r = 4;
        
        %Filter using the transform filter.
         TFout = TF(im_out, sigma_s, sigma_r);
        
        TFout(TFout>1)=1;
        TFout(TFout<0)=0;
    end
end