function res = hdrvdp3( task, test, reference, color_encoding, pixels_per_degree, options )
% HDR-VDP-3 compute visually significant differences between an image pair.
%
% diff = HDRVDP3( task, test, reference, color_encoding, pixels_per_degree, options )
%
% Parameters:
%   task - the task for which the metric should predict. The options are:
%     'side-by-side' - side-by-side comparison of two images
%     'flicker' - the comparison of two images shown in the same place and
%                 swapped every 0.5 second.
%     'detection' - detection of a single difference in
%                 multiple-alternative-forced-choice task (the same task as
%                 in HDR-VDP-2).
%     'quality' - prediction of image quality (Q and Q_JOD)
%     'civdm' - contrast invariant visual difference metric that can compare LDR
%               and HDR images. See CIVDM section below.
%
%   test - image to be tested (e.g. with distortions)
%   reference - reference image (e.g. without distortions)
%   color_encoding - color representation for both input images. See below.
%   pixels_per_degree - visual resolution of the image. See below.
%   options - cell array with { 'option', value } pairs. See the list of options
%       below. Note if unknown or misspelled option is passed, no warning
%       message is issued.
%
% The function returns a structure with the following fields:
%   P_map - probability of detection per pixel (matrix 0-1)
%   P_det - a single valued probability of detection (scalar 0-1)
%   C_map - threshold normalized contrast map, so that C_max=1
%           corresponds to the detection threshold (P_det=0.5).
%   C_max - maximum threshold normalized contrast, so that C_max=1
%           corresponds to the detection threshold (P_det=0.5).
%   Q     - Quality correlate, which is 10 for the best quality and gets
%           lower for lower quality. Q can be negative in case of very large
%           differences.
%   Q_JOD - Quality correlate in the units of Just Objectionable
%           Differences. 1 JOD means that 75% of observers will select
%           reference image over a test image in a
%           2-alternative-forced-choice test. Refer to section IV.B of
%           https://www.cl.cam.ac.uk/research/rainbow/projects/unified_quality_scale/perezortiz2019unified_quality_scale-large.pdf
%           for the explanation of JOD units.
%
% Test and references images are matrices of size (height, width,
% channel_count). If there is only one channel, it is assumed to be
% achromatic (luminance) with D65 spectra. The values must be given in
% absolute luminance units (cd/m^2). If there are three color channels,
% they are assumed to be RGB of an LCD display with red, green and blue LED
% backlight. If different number of color channels is passed, their spectral
% emission curve should be stored in a comma separated text file and its
% name should be passed as 'spectral_emission' option.
%
% Note that the current version of HDR-VDP does NOT take color differences
% into account. Spectral channels are used to properly compute luminance
% sensed by rods and cones.
%
% COLOR ENCODING:
%
% HDR-VDP-2 requires that the color encoding is specified explicitly to avoid
% mistakes when passing images. HDR-VDP operates on absolue physical units,
% not pixel values that can be found in images. Therefore, it is necessary
% to specify how the metric should interpret input images. The available
% options are:
%
% 'luminance' - images contain absolute luminance values provided in
% photopic cd/m^2. The images must contain exactly one color channel.
%
% 'luma-display' - images contain grayscale pixel values, sometimes known
% as gamma-corrected luminance. The images must contain exactly one color
% channel and the maximum pixel value must be 1 (not 256).
% It corresponds to a gray-scale channel in
% YCrCb color spaces used for video encoding. Because 'luma' alone does not
% specify luminance, HDR-VDP-2 assumes the following display model:
%
% L = 99 * V^2.2 + 1,
%
% where L is luminance and V is luma. This corresponds to a display with
% 2.2 gamma, 100 cd/m^2 maximum luminance and 1 cd/m^2 black level.
% For more flexible display model, use hdrvdp_gog_display_model().
%
% 'sRGB-display' - use this color encoding for standard (LDR) color images.
% This encoding assumes an sRGB display, with 100 cd/m^2 peak luminance and
% 1 cd/m^2 black level. Note that this is different from sRGB->XYZ
% transform, which assumes the peak luminance of 80 cd/m^2 and 1 cd/m^2
% black level. The maximum pixel value must be 1 (not 256).
% For more flexible display model, use hdrvdp_gog_display_model().
% If not specified, 'rgb_display' is set to 'led-lcd-srgb' when using this
% color space.
%
% 'rgb-native' - (if unsure, use this encoding for HDR images) native linear
% RGB color space of the specified display.
% If not specified, 'rgb_display' is set to 'led-lcd-srgb' when using this
% color encoding. Use this color encoding for typical HDR images, scaled in
% absolute units (see the note below). The colors will be close to BT.709
% and, since no color mapping will be done, no colors will be clipped.
%
% 'rgb-bt.709' - linear RGB colour space with BT.709 primaries. This is a
% standard colour spaces used for HD content and most monitors. sRGB colour
% space is using the same primaries.
% If not specified, 'rgb_display' is set to 'led-lcd-srgb' when using this
% color encoding.
%
% 'rgb-bt.2020' - linear RGB colour space with BT.2020 primaries. This is a
% standard colour spaces used for HDR content.
% If not specified, 'rgb_display' is set to 'oled' when using this
% color encoding.
%
% 'XYZ' - input image is provided as ABSOLUTE trichromatic color values in
% CIE XYZ (1931) color space. The Y channel must be equal luminance in
% cd/m^2.
% If not specified, 'rgb_display' is set to 'oled' when using this
% color encoding.
%
% Important note on using linear color spaces: 'rgb-native', 'rgb-bt.709',
% 'rgb-bt.2020', 'XYZ' and 'luminance'.
%
% The values must be in absolute (not relative) units. This means that if a
% monitor has a peak brightness of 400 cd/m^2 (nits), the white color of
% the input image needs to have the RGB values of [400 400 400]. Note that
% most HDR images available are stored in relative units, which needs to be
% mapped to a display by multiplying by appropriate constant. HDR-VDP
% operates on absolute colorimetric values and makes a distinction between
% brighter and darker displays.
%
%
% PIXELS_PER_DEGREE:
%
% This argument specifies the angular resolution of the image in terms of the
% number of pixels per one visual degree. It
% will change with a viewing distance and display resolution. A typical
% value for a stanard resolution computer display is around 30.
% You can use hdrvdp_pix_per_deg() function  to compute the value of this
% parameter.
%
% OPTIONS:
%
% The options must be given as name-value pairs in a cell array.
% Default values are given in square brackets.
%
%   'surround' - absolute luminance / color of the display surround or one
%       of the special settings: 'none', 'mean'. Default is 'none'.
%       The surround luminance affects the glare introduced into the image.
%       This option is useful when simulating displays in bright
%       environments. It should be either a scalar, a vector of the length
%       corresponding to the color channels of the image, or one of:
%       'mean' - use the geometric mean of the reference image
%                (separately for each color channel).
%       'none' (default) - try to avoid the influence of the surround.
%       (this is done by a mirror reflection of the pixels outside the
%       image)
%
%   'age' - (default 24) age of the observer in years.
%       HDR-VDP simulates age-related loss of sensitivity and optical
%       performance. For details, refer to the paper:
%          Mantiuk, R. K., & Ramponi, G. (2018). Age-dependent predictor of
%          visibility in complex scenes. Journal of the Society for
%          Information Display, 1-21. https://doi.org/10.1002/jsid.623
%       It is recommended to use 'cie' MTF with when simulating age-related
%       effects.
%
%   'spectral_emission' - name of the comma separated file that contains
%      spectral emission curves of color channels of reference and test
%      images.
%
%   'mtf' - modulation transfer function of the eye optics. The options
%      are:
%         'hdrvdp' - (default) MTF fitted to glare data
%         'cie99' - age-adaptive glare-spread-function (GSF) from the CIE
%                   report (CIE 135/1-6 Disability Glare)
%         'none' - do not use any glare model
%      MTF models scattering of the light in the eye's optics (glare). The
%      effect is very relevant for high-contrast HDR images. But this is also
%      computationally expensive step of the metric.
%
%   'rgb_display' - use this option to specify one of the
%      predefined emission spectra for typical displays. Available options
%      are:
%        led-lcd-srgb - an LCD display with an LED backlight (rec.709 gamut)
%        led-lcd-wcg - an LCD display with an LED backlight (wide color gamut)
%        oled - an AMOLED display (wide colour gamut)
%        crt - [depreciated] a typical CRT display
%        ccfl-lcd - an LCD display with CCFL backlight (wide color gamut)
%     If the option is left unspecified, either led-lcd-srgb or oled will be used,
%     depending on the selected color encoding (see above). Note that if
%     the image contains colors that are out of color gamut of the
%     specified display, those colors will be clamped and a warning message
%     will be shown if at least 1% of the pixels was clipped.
%
% The following are the most important options used to fine-tune and calibrate
% HDR-VDP:
%   'sensitivity_correction' - relative correction of the absolute
%              sensitivity. Positive value (e.g. 0.2) will make the metric
%              more sensitive, negative values, less sensitive. The default value
%              is 0.
%   'mask_p' - excitation of the visual contrast masking model
%   'mask_q' - inhibition of the visual contrast masking model
%
% CIVDM
%
% When the "target" parametyer is set to 'civdm', the metric runs a modern
% implementation of the dynamic range independent metric from:
%
% T. O. Aydin, R. Mantiuk, K. Myszkowski, and H.-P. Seidel,
% "Dynamic range independent image quality assessment"
% ACM Trans. Graph. (Proc. SIGGRAPH), vol. 27, no. 3, p. 69, 2008.
%
% The dynamic range independent metric can compare HDR and SDR images and
% can be used to evaluate tone-mapping or inverse-tone-mapping. Note that
% this implementation is different from the one presented in the original
% paper in a few aspects:
% - "contrast reversal" map is disabled (always returns 0s) because it
%   prooved to be unreliable predictor of artifacts.
% - the detection model is build on a newer HDR-VDP architecture. The
%   original metric was based on HDR-VDP 1.0
% - the results are processed by morfological openning operator to
%   eliminate spurious predictions (artifact of multi-scale decomposition).
%
% EXAMPLE:
%
% The following example creates a luminance ramp (gradient), distorts it
% with a random noise and computes detection probabilities using HDR-VDP.
% More examples can be found in the "examples" foder.
%
% %The reference image is a luminance gradient from 0.1 to 1000 cd/m^2
% reference = logspace( log10(0.1), log10(1000), 512 )' * ones( 1, 512 );
%
% % The test image is the reference image with added relative (multiplicative)
% % noise.
% noise_contrast = 0.05;
% test = reference .* (1 + (rand( 512, 512 )-0.5)*2*noise_contrast);
%
% % Find the angular resolution in pixels per visual degree:
% % 24" HD monitor seen from 0.5 meters
% ppd = hdrvdp_pix_per_deg( 30, [1920 1200], 0.5 );
%
% % Find the detection threshold for two images shown side by size. The
% % images are shown on the background of 13 cd/m^2. Note that "test" and
% % "reference" images contain absolute luminance in cd/m^2 (nit).
% res = hdrvdp3( 'side-by-side', test, reference, 'luminance', ppd, { 'surround', 13 } );
%
% clf;
%
% % Show the visualization of the differences. Note that the noise is more
% % visible in the brighter part (bottom of the image).
% imshow( hdrvdp_visualize( 'pmap', res.P_map ) );

%
% BUGS and LIMITATIONS:
%
% If you suspect the predictions are wrong, check first the Frequently
% Asked Question section on the HDR-VDP-2 web-site
% (http://hdrvdp.sourceforge.net). If it does not help, post your problem to
% the HDR-VDP discussion group: http://groups.google.com/group/hdrvdp
% (preffered) or send an e-mail directly to the author.
%
% REFERENCES
%
% The metric is described in the paper:
% Mantiuk, Rafal, Kil Joong Kim, Allan G. Rempel, and Wolfgang Heidrich.
% HDR-VDP-2: A Calibrated Visual Metric for Visibility and
%  Quality Predictions in All Luminance Conditions.
% ACM Trans. Graph (Proc. SIGGRAPH) 30, no. 4 (July 01, 2011)
% doi:10.1145/2010324.1964935.
%
% When refering to the metric, please cite the paper above and include the
% version number, for example "HDR-VDP 2.2.0". To check the version number,
% see the ChangeLog.txt. To check the version in the code, call
% hdrvdp_version.txt.
%
% Copyright (c) 2010-2019, Rafal Mantiuk

% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%  * Redistributions of source code must retain the above copyright notice,
%    this list of conditions and the following disclaimer.
%  * Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the  documentation
%    and/or other materials provided with the distribution.
%  * Neither the name of the HDR-VDP nor the names of its contributors may be
%    used to endorse or promote products derived from this software without
%    specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
% DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if( any( size(test) ~= size(reference) ) )
    error( 'reference and test images must be the same size' );
end

if( ~exist( 'options', 'var' ) )
    options = {};
end

if( ~exist( 'reconSpyr', 'file' ) )
    % If matlabPyrTools not in the path, add them now
    
    % Get the path to the hdrvdp directory
    [pathstr, name, ext] = fileparts(mfilename( 'fullpath' ));
    
    addpath( fullfile( pathstr, 'matlabPyrTools_1.4_fixed' ) );
    addpath( fullfile( pathstr, 'utils' ) );
    addpath( fullfile( pathstr, 'data' ) );
    
    % Re-check if everything went OK
    if( ~exist( 'reconSpyr', 'file' ) )
        error( 'Failed to add matlabPyrTools to the path.' );
    end
end

metric_par = hdrvdp_parse_options( task, options );

% The parameters overwrite the options
if( ~isempty( pixels_per_degree ) )
    metric_par.pix_per_deg = pixels_per_degree;
end
if( ~isempty( color_encoding ) )
    metric_par.color_encoding = color_encoding;
end

% Load spectral emission curves
img_channels = size( test, 3 );

switch lower( metric_par.color_encoding )
    case 'luminance'
        if( img_channels ~= 1 )
            error( 'Only one channel must be provided for "luminance" color encoding' );
        end
        check_if_values_plausible( test, metric_par );
    case 'luma-display'
        if( img_channels ~= 1 )
            error( 'Only one channel must be provided for "luma-display" color encoding' );
        end
        test = display_model( test, 2.2, 99, 1 );
        reference = display_model( reference, 2.2, 99, 1 );
    case 'srgb-display'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "sRGB-display" color encoding' );
        end
        if ~isfield( metric_par, 'rgb_display' )
            metric_par.rgb_display = 'led-lcd-srgb';
        end
        test = display_model_srgb( test );
        reference = display_model_srgb( reference );
    case 'rgb-bt.709'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "rgb-bt.709" color encoding' );
        end
        if ~isfield( metric_par, 'rgb_display' )
            metric_par.rgb_display = 'led-lcd-srgb';
        end
        check_if_values_plausible( test(:,:,2), metric_par );
    case 'rgb-bt.2020'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "rgb-bt.2020" color encoding' );
        end
        if ~isfield( metric_par, 'rgb_display' )
            metric_par.rgb_display = 'oled';
        end
        check_if_values_plausible( test(:,:,2), metric_par );
    case 'rgb-native'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "rgb-native" color encoding' );
        end
        if ~isfield( metric_par, 'rgb_display' )
            metric_par.rgb_display = 'led-lcd-srgb';
        end
        check_if_values_plausible( test(:,:,2), metric_par );
    case 'xyz'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "XYZ" color encoding' );
        end
        if ~isfield( metric_par, 'rgb_display' )
            metric_par.rgb_display = 'oled';
        end
        check_if_values_plausible( test(:,:,2), metric_par );
    case 'generic'
        if( isempty( metric_par.spectral_emission ) )
            error( '"spectral_emission" option must be specified when using the "generic" color encoding' );
        end
    otherwise
        error( 'Unknown color encoding "%s"', metric_par.color_encoding );
end

if ischar( metric_par.surround )
    switch metric_par.surround
        case 'none'
            metric_par.pad_value = 'symmetric';
        case 'mean'
            metric_par.pad_value = geomean( reshape( reference, [size(reference,1)*size(reference,2) size(reference,3)] ) );
        otherwise
            error( 'Unrecognized "surround" setting' );
    end
else
    if( length(metric_par.surround) == 1 )
        metric_par.pad_value = repmat( metric_par.surround, [1 img_channels] );
    elseif( length(metric_par.surround) == img_channels )
        metric_par.pad_value = metric_par.surround;
    else
        error( 'The length of the "surround" vector should be 1 or equal to the number of channels' );
    end
end

if( img_channels == 3 && ~isfield( metric_par, 'rgb_display' ) )
    metric_par.rgb_display = 'ccfl-lcd';
end

if( ~isempty( metric_par.spectral_emission ) )
    [tmp, IMG_E] = load_spectral_resp( fullfile(metric_par.base_dir, metric_par.spectral_emission) );
elseif( isfield( metric_par, 'rgb_display' ) )
    [tmp, IMG_E] = load_spectral_resp( sprintf( 'emission_spectra_%s.csv', metric_par.rgb_display ) );
elseif( img_channels == 1 )
    [tmp, IMG_E] = load_spectral_resp( 'd65.csv' );
else
    error( '"spectral_emission" option needs to be specified' );
end

if( img_channels == 1 && size(IMG_E,2)>1 )
    % sum-up spectral responses of all channels for luminance-only data
    IMG_E = sum( IMG_E, 2 );
end

if( img_channels ~= size( IMG_E, 2 ) )
    error( 'Spectral response data is either missing or is specified for different number of color channels than the input image' );
end
metric_par.spectral_emission = IMG_E;


% ==== Tranform from a standard to native (RGB) colour space

M_itu2native = [];
switch lower( metric_par.color_encoding )
    
    case 'rgb-bt.2020'
        M_itu2native = hdrvdp_iturgb2native( 'rgb-bt.2020', IMG_E );
        
    case { 'rgb-bt.709', 'srgb-display' }
        M_itu2native = hdrvdp_iturgb2native( 'rgb-bt.709', IMG_E );
        
    case 'xyz'
        M_itu2native = hdrvdp_iturgb2native( 'xyz', IMG_E );
end
if ~isempty( M_itu2native )
    test = fix_out_of_gamut( hdrvdp_colorspace_transform( test, M_itu2native ), metric_par );
    reference = fix_out_of_gamut( hdrvdp_colorspace_transform( reference, M_itu2native ), metric_par );
end



% ==== Age-related effect ====

if( metric_par.do_asl )
    % Adjust sensitivity for the age related sensitivity loss
    % TODO - this should be wavelength dependent effect
    
    age_lut = [20  30    40    50    60    70];
    ASL_lut = [.0  .078  .156  .234  .312  .39];
    
    asl = interp1( age_lut, ASL_lut, clamp( metric_par.age, age_lut(1), age_lut(end) ) );
    metric_par.sensitivity_correction = metric_par.sensitivity_correction - asl;
end

switch( metric_par.ms_decomp )
    case 'spyr'
        metric_par.mult_scale = hdrvdp_spyr( metric_par.steerpyr_filter );
    case 'lpyr'
        metric_par.mult_scale = hdrvdp_lpyr( );
    case 'lpyr_dec'
        metric_par.mult_scale = hdrvdp_lpyr_dec( );
    otherwise
        error( 'Unknown multiscale decomposition "%s"', metric_par.ms_decomp );
end

% Compute spatially- and orientation-selective bands
% Process reference first to reuse bb_padvalue
[B_R, L_adapt_reference, bb_padvalue, P_ref] = hdrvdp_visual_pathway( reference, 'reference', metric_par, -1 );
[B_T, L_adapt_test, bb_padvalue, P_test] = hdrvdp_visual_pathway( test, 'test', metric_par, bb_padvalue );

band_freq = B_T.get_freqs();


% precompute CSF
CSF.csf_la = logspace( -5, 5, 256 );
CSF.csf_log_la = log10( CSF.csf_la );

CSF.S = zeros( length(CSF.csf_la), B_T.band_count() );
for b=1:B_T.band_count()
    CSF.S(:,b) = hdrvdp_ncsf( band_freq(b), CSF.csf_la', metric_par );
end
L_mean_adapt = (L_adapt_test + L_adapt_reference)./2;
log_La = log10(clamp( L_mean_adapt, CSF.csf_la(1), CSF.csf_la(end) ));

%log_La = log10(clamp( L_adapt, csf_la(1), csf_la(end) ));


if( metric_par.do_civdm )
    res.civdm = civdm( B_T, L_adapt_test, B_R, L_adapt_reference, CSF, metric_par );
else
    
    % Pixels that are actually different
    diff_mask = any((abs( test-reference ) ./ reference) > 0.001, 3);
    if( isfield( metric_par, 'cmp_mask' ) )
        diff_mask = diff_mask .* metric_par.cmp_mask;
    end
    
    if( metric_par.do_quality_raw_data )
        qres = quality_pred_init();
    end
    
    D_bands = B_T;
    
    Q_err = 0; % Error measure used for quality predictions
    
    % For each band
    for b = 1:B_T.band_count()
        
        %masking params
        p = 10.^metric_par.mask_p;
        q = 10.^metric_par.mask_q;
        pf = (10.^metric_par.psych_func_slope)/p;
        
        % accumulate masking activity across orientations (cross-orientation
        % masking)
        do_mask_xo = (B_T.orient_count(b)>1); % No cross-orientation masking of only one band
        
        if do_mask_xo
            mask_xo = zeros( B_T.band_size(b,1) );
            for o=1:B_T.orient_count(b)
                mask_xo = mask_xo + mutual_masking( b, o );
            end
        end
        
        
        log_La_rs = clamp( imresize(log_La,B_T.band_size(b,1)), CSF.csf_log_la(1), CSF.csf_log_la(end) );
        
        % per-pixel contrast sensitivity
        CSF_b = interp1( CSF.csf_log_la, CSF.S(:,b), log_La_rs );
        
        % REMOVED: Transform CSF linear sensitivity to the non-linear space of the
        % photoreceptor response
        %    CSF_b = CSF_b .* 1./hdrvdp_joint_rod_cone_sens_diff( 10.^log_La_rs, metric_par );
        
        for o=1:B_T.orient_count(b)
            
            band_diff = B_T.get_band(b,o) - B_R.get_band(b,o);
            
            if( b == B_T.band_count() )
                % base band
                
                band_freq = B_T.get_freqs();
                
                % Find the dominant frequency and use the sensitivity for
                % that frequency (more robust than fft filtering used
                % before)
                rho_bb = create_cycdeg_image( size(band_diff), band_freq(end)*2*2 );
                band_diff_f = fft2( band_diff );
                band_diff_f(1,1) = 0; % Ignore DC
                ind = find( abs(band_diff_f(:))==max(abs(band_diff_f(:))), 1 );
                bb_freq = rho_bb(ind);
                
                L_mean = 10.^mean( log_La_rs(:) );
                
                N_nCSF = 1./hdrvdp_ncsf( bb_freq, L_mean, metric_par );
            else
                N_nCSF = (1./CSF_b);
            end
            
            %excitation difference
            ex_diff = sign_pow(band_diff, p);
            
            if ~isempty( metric_par.ignore_freqs_lower_than ) && band_freq(b) < metric_par.ignore_freqs_lower_than
                % An option to ignore low-freq bands
                D_bands = D_bands.set_band( b, o, zeros(size(ex_diff)) );
                
            elseif metric_par.do_masking
                
                k_mask_self = 10.^metric_par.mask_self; % self-masking
                k_mask_xo = 10.^metric_par.mask_xo;     % masking across orientations
                k_mask_xn = 10.^metric_par.mask_xn;     % masking across neighboring bands
                
                
                do_mask_xn = (metric_par.mask_xn>-10); % skip if the effect is too weak
                
                self_mask = mutual_masking( b, o );
                
                mask_xn = zeros( size( self_mask ) );
                if( b > 1 && do_mask_xn)
                    mask_xn = max( imresize( mutual_masking( b-1, o ), size( self_mask ) ), 0 );
                end
                if( b < (B_T.band_count()-1) && do_mask_xn )
                    mask_xn = mask_xn + max( imresize( mutual_masking( b+1, o ), size( self_mask ) ), 0 );
                end
                
                % correct activity for this particular channel
                if do_mask_xo
                    band_mask_xo = max( mask_xo - self_mask, 0 );
                else
                    band_mask_xo = 0;
                end
                
                N_mask = (k_mask_self*(abs(self_mask)).^q + ...
                    k_mask_xo*(abs(band_mask_xo)).^q + ...
                    k_mask_xn*(abs(mask_xn)).^q);
                
                D = ex_diff./sqrt( N_nCSF.^(2*p) + N_mask.^2 );
                
                D_bands = D_bands.set_band( b, o, sign_pow(D, pf) );
                
                %fprintf( 1, 'b = %d, o = %d, freq = %g, ampl = %g\n', ...
                %             b, o, band_freq(b), max(abs(D(:))) )
                
            else
                % NO masking
                D = ex_diff./N_nCSF.^p;
                D_bands = D_bands.set_band( b, o, sign_pow(D, pf) );
            end
            
            
            % Quality prediction
            w_f = interp1( metric_par.quality_band_freq, metric_par.quality_band_w, ...
                clamp( band_freq(b), metric_par.quality_band_freq(end), metric_par.quality_band_freq(1) ) );
            epsilon = 1e-12;
            
            % Mask the pixels that are almost identical in test and
            % reference images. Improves predictions for small localized
            % differences.
            diff_mask_b = imresize( double(diff_mask), size( D ) );
            D_p = D .* diff_mask_b;
            
            % Per-band error used for quality predictions
            Q_err = Q_err + minkowski_sum( D_p(:), 0.8 )/B_T.band_count();
            
            if( metric_par.do_quality_raw_data )
                qres = quality_pred_band( qres, D_p, b, o );
            end
            
        end
        
    end
    
    S_map = abs(D_bands.reconstruct( ));
    
    if( metric_par.do_spatial_total_pooling )
        S_map = sum(S_map(:))/(max(S_map(:))+1e-12)*S_map;
    end
    
    % TODO: localized distortions may cause prediction of visibilble differences
    % in other parts of an image because they affect low frequencies. This is
    % especially apparent for super-threshold differences. A mechanism to
    % restrict location of such changes is needed.
    %
    %S_map = S_map .* double(diff_mask);
    
    res.P_map = 1 - exp( log(0.5)*S_map );
    
    % TODO: this seems to be redundant - similar as C_max
    % Find the maximum in a low-pass filetered map - more robust
    si_sigma = 10^metric_par.si_sigma * metric_par.pix_per_deg;
    S_map_si = fast_gauss( S_map, si_sigma, true, 0 );
    res.S_max = max( S_map_si(:) );
    
    % Spatial probability summation - integrate probabilities within a small
    % area
    if( metric_par.do_sprob_sum )
        si_sigma = 10^metric_par.si_sigma * metric_par.pix_per_deg;
        res.P_map = 1 - exp(fast_gauss( log(1-res.P_map+1e-4), si_sigma, true, 0 ));
    end
    
    if( metric_par.do_pixel_threshold )
        m_thr = 0.01;
        is_diff = 1 - exp( log(0.5)*(abs(L_adapt_reference-L_adapt_test)./L_adapt_reference / m_thr).^3.5 );
        res.P_map = res.P_map.*is_diff;
    end
    
    if( metric_par.do_robust_pdet )
        % Since max operator will pick up the maximum prediction error, a more robust estimate is given if
        % P_map is gauss-filtered before computing max.
        rpd_sigma = 0.25 * metric_par.pix_per_deg;
        pm = fast_gauss( res.P_map, rpd_sigma, true, 0 );
        res.P_det = max( pm(:) );
        cm = fast_gauss( S_map, rpd_sigma, true, 0 );
        res.C_max = max( cm(:) );
    else
        res.P_det = max( res.P_map(:) );
        res.C_max = max( S_map(:) );
    end
    
    
    res.C_map = S_map;
    
    quality_floor = -4; % The minimum difference to consider in the log domain
    max_q = 10; % The maximum quality value
    res.Q = max_q+quality_floor-log(exp(quality_floor)+Q_err);
    
    % Mapping the quality correlate to approximate JODs
    jod_pars = [0.5200    1.2812];
    res.Q_JOD = 10-jod_pars(1)*max(10-res.Q,0).^jod_pars(2);
    
    if( metric_par.do_quality_raw_data )
        res.qres = qres;
    end
    
end
    function m = mutual_masking( b, o )
        
        test_band = B_T.get_band(b,o);
        reference_band = B_R.get_band(b,o);
        
        m = min( abs(test_band), abs(reference_band) );
        
        if( metric_par.do_si_gauss )
            % simplistic phase-uncertainty mechanism
            m = fast_gauss( m, 10^metric_par.si_size, false, 0 );
            %m = blur_gaussian( m, 10^metric_par.si_size );
            
        else
            if( metric_par.do_pu_dilate )
                m = imdilate( m, strel('disk', metric_par.pu_dilate) );
            else
                F = ones( metric_par.masking_pool_size, metric_par.masking_pool_size);
                masking_norm = 10^metric_par.masking_norm;
                m = conv2( m.^masking_norm, F/numel(F), 'same').^(1/masking_norm);
            end
        end
        
    end

end

function y = sign_pow( x, e )
y = sign(x) .* abs(x).^e;
end

function L = display_model( V, gamma, peak, black_level )
L = peak * V.^gamma + black_level;
end

function RGB = display_model_srgb( sRGB )
a = 0.055;
thr = 0.04045;

RGB = zeros(size(sRGB));
RGB(sRGB<=thr) = sRGB(sRGB<=thr)/12.92;
RGB(sRGB>thr) = ((sRGB(sRGB>thr)+a)/(1+a)).^2.4;

RGB = 99*RGB + 1;

end

function R = civdm( B_T, L_adapt_test, B_R, L_adapt_reference, CSF, metric_par )
% This is the adaptation of the dynamic range independent quality metric to
% HDR-VDP-2. It is rebranded as a Contrast Invariant Visibility
% Difference Metric to avoid confusion with the original paper.

pf = 10.^metric_par.psych_func_slope;
a = (-log(0.5)).^(1/pf);

%c_sep = 0.6; % separation between visible and invisible in log10 units
%c_sep = 0; %0.3; % separation between visible and invisible in log10 units

%c_vis = 10^(c_sep/2);  % contrast considered as visible
%c_invis = 10^(-c_sep/2); % contrast considered as invisible


P_loss = B_T;
P_ampl = B_T;
P_rev = B_T;

for b = 1:B_T.band_count()
    
    % per-pixel contrast sensitivity, separate for test and reference
    % images as they can differ in the absolute luminance
    log_La_test_rs = clamp( log10(imresize(L_adapt_test,B_T.band_size(b,1))), CSF.csf_log_la(1), CSF.csf_log_la(end) );
    CSF_b_test = interp1( CSF.csf_log_la, CSF.S(:,b), log_La_test_rs );
    
    log_La_ref_rs = clamp( log10(imresize(L_adapt_reference,B_T.band_size(b,1))), CSF.csf_log_la(1), CSF.csf_log_la(end) );
    CSF_b_ref = interp1( CSF.csf_log_la, CSF.S(:,b), log_La_ref_rs );
    
    for o=1:B_T.orient_count(b)
        
        % Skip base-band and frequencies <= 2cpd
        if( b == B_T.band_count() || B_T.band_freqs(b)<=2 )
            test = B_T.get_band(b,o);
            P_loss = P_loss.set_band( b, o, zeros(size(test)) );
            P_ampl = P_ampl.set_band( b, o, zeros(size(test)) );
            P_rev = P_rev.set_band( b, o, zeros(size(test)) );
            continue;
        end
        
        test = B_T.get_band(b,o);
        ref = B_R.get_band(b,o);
        
        % Dilation helps to reduce false predictions due to the size of the
        % band-bass filter.
        test_dil = imdilate( abs(test), strel('disk', 3*2^(b-1)) );
        ref_dil = imdilate( abs(ref), strel('disk', 3*2^(b-1)) );
        
        epsilon = 1e-8;
        P_thr = 0.5;
        % P of test visible = (1-exp(-c))
        
        P_t_v = psych_func(test.*CSF_b_test);
        P_t_v(P_t_v<P_thr) = 0;
        P_t_iv = 1-psych_func(test_dil.*CSF_b_test);
        P_t_iv(P_t_iv<P_thr) = 0;
        
        
        P_r_v = psych_func(ref.*CSF_b_ref);
        P_r_v(P_r_v<P_thr) = 0;
        P_r_iv = 1-psych_func(ref_dil.*CSF_b_ref);
        P_r_iv(P_r_iv<P_thr) = 0;
        
        % To get actual probability (for debugging)
        % pfsview( 1-exp(P_t_v) )
        
        %        DD{b} = (P_r_v + P_t_iv);
        P_loss = P_loss.set_band( b, o, log(1 - P_r_v .* P_t_iv + epsilon) );
        P_ampl = P_ampl.set_band( b, o, log(1 - P_r_iv .* P_t_v + epsilon) );
        %        rev_map = B_T.get_band( b, o ).*B_R.get_band( b, o );
        %        P_rev = P_rev.set_band( b, o, log(1 - P_t_v .* P_r_v .* (rev_map<0) + epsilon) );
        %        pp_rev = max( 0, 1-(local_correlation( test, ref, metric_par.pix_per_deg )/2+0.5) );
        %        pp_rev = (local_correlation( test, ref, metric_par.pix_per_deg ) < 0.05) .* P_t_v .* P_r_v;
        
        % Contrast reversal is disabled in this version - this measures is
        % unreliable because of local interference within the band
        pp_rev = zeros(size(test));
        P_rev = P_rev.set_band( b, o, log(1 - pp_rev ) );
    end
end

R.loss = 1-exp(-abs(P_loss.reconstruct()));
R.ampl = 1-exp(-abs(P_ampl.reconstruct()));
R.rev = 1-exp(-abs(P_rev.reconstruct()));


%R.loss = imopen( R.loss, strel('disk', 3) );
%R.ampl = imopen( R.ampl, strel('disk', 3) );

%     function P = local_correlation( x, y, pix_per_deg )
%
%         sigma = 4; %0.25*pix_per_deg;
%
%         mx = fast_gauss( x, sigma );
%         my = fast_gauss( y, sigma );
%
%         sx = sqrt( fast_gauss( x.^2, sigma ) - mx.^2 );
%         sy = sqrt( fast_gauss( y.^2, sigma ) - my.^2 );
%
%         P = fast_gauss( (x-mx).*(y-my), sigma ) ./ (sx.*sy);
%
%     end

%     function y = norm_pow( x, e )
%         y = (abs(x)/band_norm).^e * band_norm;
%     end
%
%     function P_map_corr = spatial_pooling( P_map, pix_per_deg )
%         A = estimate_area( P_map, pix_per_deg );
%
%         a0 = 114;
%         f0 = 0.65;
%         f = 4;
%         a_b = pi*(1/f)^2;
%
%         C = sqrt((A*f^2) ./ (a0 + A*f0 + A*f^2)) / sqrt((a_b*f^2) ./ (a0 + a_b*f0 + a_b*f^2));
%
%         P_map_corr = clamp( P_map.*C, 0, 1 );
%
%     end
%
%     function A = estimate_area( P_map, pix_per_deg )
%         sigma = 1*pix_per_deg;
%         A = fast_gauss( P_map, sigma, false, 0 ) / pix_per_deg.^2;
%     end

    function P = psych_func( contrast )
        P = 1.0 - exp( -(abs(a*contrast)).^pf);
    end

end

function Y = spatial_summation( X, sigma )
% Essentilally a non-normalized Gaussian filter
%

ksize = round(sigma*6);
h = fspecial( 'gaussian', ksize, sigma );
h = h / max(h(:));
Y = imfilter( X, h, 'replicate' );

end

function check_if_values_plausible( img, metric_par )
% Check if the image is in plausible range and report a warning if not.
% This is because the metric is often misused and used for with
% non-absolute luminace data.

if( ~metric_par.disable_lowvals_warning )
    if( max(img(:)) <= 1 )
        warning( 'hdrvdp:lowvals', [ 'The images contain very low physical values, below 1 cd/m^2. ' ...
            'The passed values are most probably not scaled in absolute units, as requied for this color encoding. ' ...
            'See ''doc hdrvdp'' for details. To disable this warning message, add option { ''disable_lowvals_warning'', ''true'' }.' ] );
    end
end

end


function RGB_out = fix_out_of_gamut( RGB_in, metric_par )
% Report if any color are out of native color gamut and clamp them

min_v = 1e-6;
clamp_pix = nnz( any( RGB_in < min_v, 3 ) );
total_pix = size(RGB_in,1)*size(RGB_in,2);
if clamp_pix > 0
    if ~metric_par.disable_lowvals_warning && (clamp_pix/total_pix>=0.01)
        warning( 'hdrvdp:lowvals', cat( 2, 'Image contains values that are out of the native colour gamut of the display. ', ...
            sprintf( '%.2f percent of pixels will be clamped. ', clamp_pix/total_pix*100 ), ...
            'Consider using different "rgb_display" or "native" color space. ', ...
            'To disable this warning message, add option { ''disable_lowvals_warning'', ''true'' }.' ) );
    end
    RGB_out = max( RGB_in, min_v );
else
    RGB_out = RGB_in;
end

end

function d = minkowski_sum( X, p )

d = sum( abs(X).^p / numel(X) ).^(1/p);

end

