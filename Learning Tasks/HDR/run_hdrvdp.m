%Runs the hdrvdp metric on each image in the input directory and saves the result in a txt file
%
%
%
%
%

function [] = run_hdrvdp(gt_dir, pred_dir, output_file)

    if ~exist( 'hdrvdp3', 'file' )
        addpath( fullfile( pwd, 'hdrvdp-3.0.6') );
    end

    filePattern = fullfile(gt_dir, '*.hdr');
    theFiles = dir(filePattern);

    C = {};

    for k = 1 : length(theFiles)
        baseFileName = theFiles(k).name;
        gtFullFileName = fullfile(theFiles(k).folder, baseFileName);
        predFullFileName = fullfile(pred_dir, baseFileName);
        fprintf(1, 'Now treating %s\n', gtFullFileName);
        
        gt = hdrread( gtFullFileName );
        pred = hdrread( predFullFileName );

        res = hdrvdp3( 'detection', pred, gt, 'rgb-native', 30, {} );
        %fprintf(1, 'Res %s\n', res.Q);
        C(k,:) = {gtFullFileName, res.Q * 10};
    end

    T = cell2table(C);

    writetable(T,output_file);


end