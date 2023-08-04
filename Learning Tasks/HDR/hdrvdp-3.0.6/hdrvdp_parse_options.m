function metric_par = hdrvdp_parse_options( task, options )
% HDRVDP_PARSE_OPTIONS (internal) parse HDR-VDP options and create two
% structures: view_cond with viewing conditions and metric_par with metric
% parameters
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

% Defaults

metric_par.debug = false;

metric_par.base_dir = ''; % Used for calibration
metric_par.threshold_p = 0.75;
metric_par.p_obs = 0.8;

metric_par.ms_decomp = 'spyr';
metric_par.do_new_diff = false;
metric_par.do_sprob_sum = false;

metric_par.mtf = 'hdrvdp';

metric_par.ignore_freqs_lower_than = [];

% Peak contrast from Daly's CSF for L_adapt = 30 cd/m^2
%daly_peak_contrast_sens = 0.006894596;

%metric_par.sensitivity_correction = daly_peak_contrast_sens / 10.^-2.355708;
%metric_par.sensitivity_correction = daly_peak_contrast_sens / 10.^-2.4;

metric_par.view_dist = 0.5;

metric_par.do_pixel_threshold = false;

metric_par.age = 24;

if 1 % With age related effects
    
    metric_par.do_aesl = true; % Empirical (neural) sensitivity loss as a function of age
    metric_par.do_aod = true; % Aging optical density (Pokorny 1987).
    metric_par.do_slum = true; % light reduction due to senile miosis (Unified Formula based on
    % the Stanley and Davies function [Watson & Yellott 2012]
    
    % Below are the experimental aging models, found to perform worse then
    % the once above. Disabled by default. 
    metric_par.do_asl = false; % Age related sensitivity loss - averaged [Sagawa 2001, JOSA, A, 18 (11), 2659-2667, 2001]
    metric_par.do_aslw = false; % Age related sensitivity loss - wavelength dependent (note, do not use both do_asl and do_aslw) [Sagawa 2001, JOSA, A, 18 (11), 2659-2667, 2001]
    metric_par.do_sl = false; % Senile miosis (smaller pupil at old age and low light) [Sloane 1988]
    
else % No aging-relted effects
    
    metric_par.do_asl = false; % Age related sensitivity loss - averaged [Sagawa 2001, JOSA, A, 18 (11), 2659-2667, 2001]
    metric_par.do_aslw = false; % Age related sensitivity loss - wavelength dependent (note, do not use both do_asl and do_aslw) [Sagawa 2001, JOSA, A, 18 (11), 2659-2667, 2001]
    metric_par.do_sl = false; % Senile miosis (smaller pupil at old age and low light) [Sloane 1988]
    metric_par.do_aesl = false; % Empirical (neural) sensitivity loss as a function of age
    metric_par.do_aod = false; % Aging optical density (Pokorny 1987).
    metric_par.do_slum = false; % light reduction due to senile miosis (Unified Formula based on
       
end

%metric_par.aesl_slope_age = -3.85951; % not used anymore
metric_par.aesl_slope_freq = -2.711;
metric_par.aesl_base = -0.125539;
metric_par.do_pu_dilate = false;
metric_par.masking_norm = 0; % The L-p norm used for summing up energe from the band for masking
metric_par.masking_pool_size = 3; % The size of the kernel used for pooling masking signal energy

metric_par.spectral_emission = [];

metric_par.w_low = 0;
metric_par.w_med = 0;
metric_par.w_high = 0;

metric_par.orient_count = 4; % the number of orientations to consider

% Various optional features
metric_par.do_masking = true;
metric_par.noise_model = true;
metric_par.do_quality = false; % Do quality predictions
metric_par.do_quality_raw_data = false; % for development purposes only
metric_par.do_si_gauss = false;
metric_par.si_size = 1.008;
metric_par.si_mode='none';

metric_par.do_civdm = false; % contrast indepenent structural difference metric
metric_par.disable_lowvals_warning = false;

metric_par.steerpyr_filter = 'sp3Filters';

% Psychometric function
metric_par.psych_func_slope = log10(3.5);
%metric_par.beta = metric_par.psych_func_slope-metric_par.mask_p;


metric_par.si_size = -0.034244;

metric_par.pu_dilate = 3;

% Spatial summation
metric_par.si_slope = -0.850147;
metric_par.si_sigma = -0.000502005;
metric_par.si_ampl = 0;

% Cone and rod cvi functions
metric_par.cvi_sens_drop = 0.0704457;
metric_par.cvi_trans_slope = 0.0626528;
metric_par.cvi_low_slope = -0.00222585;

metric_par.rod_sensitivity = 0;
%metric_par.rod_sensitivity = -0.383324;
metric_par.cvi_sens_drop_rod = -0.58342;

% Achromatic CSF
metric_par.csf_m1_f_max = 0.425509;
metric_par.csf_m1_s_high = -0.227224;
metric_par.csf_m1_s_low = -0.227224;
metric_par.csf_m1_exp_low = log10( 2 );

par = [0.061466549455263 0.99727370023777070]; % old parametrization of MTF
metric_par.mtf_params_a = [par(2)*0.426 par(2)*0.574 (1-par(2))*par(1) (1-par(2))*(1-par(1))];
metric_par.mtf_params_b = [0.028 0.37 37 360];

%metric_par.quality_band_freq = [15 7.5 3.75 1.875 0.9375 0.4688 0.2344];
metric_par.quality_band_freq = [60 30 15 7.5 3.75 1.875 0.9375 0.4688 0.2344 0.1172];

%metric_par.quality_band_w = [0.2963    0.2111    0.1737    0.0581   -0.0280    0.0586    0.2302];

% New quality calibration: LDR + HDR datasets - paper to be published
%metric_par.quality_band_w = [0.2832    0.2142    0.2690    0.0398    0.0003    0.0003    0.0002];
metric_par.quality_band_w = [0 0.2832 0.2832    0.2142    0.2690    0.0398    0.0003    0.0003    0 0];

metric_par.quality_logistic_q1 = 3.455;
metric_par.quality_logistic_q2 = 0.8886;

metric_par.calibration_date = '07 April 2020';

metric_par.surround = 'none';

metric_par.ms_decomp='spyr';
metric_par.steerpyr_filter='sp0Filters';

% HDR-VDP-csf refitted on 19/04/2020
metric_par.csf_params = [ ...
   0.699404   1.26181   4.27832   0.361902   3.11914
   1.00865   0.893585   4.27832   0.361902   2.18938
   1.41627   0.84864   3.57253   0.530355   3.12486
   1.90256   0.699243   3.94545   0.68608   4.41846
   2.28867   0.530826   4.25337   0.866916   4.65117
   2.46011   0.459297   3.78765   0.981028   4.33546
   2.5145   0.312626   4.15264   0.952367   3.22389 ];

metric_par.csf_lums = [ 0.0002 0.002 0.02 0.2 2 20 150];

metric_par.csf_sa = [315.98 6.7977 1.6008 0.25534];
metric_par.csf_sr = [1.1732 1.32 1.095 0.5547 2.9899 1.8]; % rod sensitivity function

switch( task )
    case { 'side-by-side', 'sbs' }
        % Fitted to LocVisVC (http://dx.doi.org/10.1109/CVPR.2019.00558) using after all updates in HDR-VDP-3.0.6 (24/05/2020)
        % marking_2020_05_sbs-stdout
        
        metric_par.base_sensitivity_correction = 0.203943775672;
        metric_par.mask_self=1.41846291721;
        metric_par.mask_xn=0.136877512403;
        metric_par.mask_xo=-50;
        metric_par.mask_q=0.108934275615;
        
        metric_par.mask_p=0.3424;
        metric_par.do_sprob_sum=true;
        metric_par.psych_func_slope=0.34;
        
        metric_par.si_sigma=-0.502280453708;
        
        metric_par.do_robust_pdet = true;
        metric_par.do_spatial_total_pooling = false;

    case 'flicker'
        % Fitted to compression_flicker dataset  (23/05/2020)
        % test_results/marking_flicker_peak_mask-all-stdout                
      
        metric_par.base_sensitivity_correction = 0.359225308021 + 0.14178269059; 
        metric_par.mask_self=1.48041073222;
        metric_par.mask_xn=0.00886149078207;
        metric_par.mask_xo=-50;
        metric_par.mask_q=0.11339123107;
        
        metric_par.mask_p=0.3424;      
        metric_par.do_sprob_sum=true;
        metric_par.psych_func_slope=0.591472024776;
        
        metric_par.si_sigma=-0.511779476487;
        
        metric_par.do_robust_pdet = true;
        metric_par.do_spatial_total_pooling = false;
        
    case { 'quality' }        
        % The same parameters as side-by-side

        metric_par.base_sensitivity_correction = 0.203943775672;
        metric_par.mask_self=1.41846291721;
        metric_par.mask_xn=0.136877512403;
        metric_par.mask_xo=-50;
        metric_par.mask_q=0.108934275615;
        
        metric_par.mask_p=0.3424;
        metric_par.do_sprob_sum=true;
        metric_par.psych_func_slope=0.34;
        
        metric_par.si_sigma=-0.502280453708;
        
        if 0 % Old fit
        metric_par.base_sensitivity_correction = -0.0591; 
        metric_par.mask_self=1.35710314938;
        metric_par.mask_xn=-50;
        metric_par.mask_xo=1.23117524266;
        metric_par.mask_q=0.129948514784;
        
        metric_par.mask_p=0.3424;
        metric_par.do_sprob_sum=true;
        metric_par.psych_func_slope=0.5441;
        
        metric_par.si_sigma=-0.502806518922;
        end
        
        metric_par.do_robust_pdet = true;
        metric_par.do_spatial_total_pooling = false;                
        
    case 'detection'
        metric_par.base_sensitivity_correction = -1; %daly_peak_contrast_sens / 10.^-1.3;
        
        % These are standard parameters that work well for Nachmias' data
        metric_par.mask_p=log10(2.2);
        metric_par.psych_func_slope=log10(3.5);
        
        % Fitted on "complexfest", "csfflat", "foley"
        % Calibration from 03/05/2020 (hdrvdp mtf)                
        metric_par.mask_self = 1.73128;
        metric_par.mask_xo = 0.816109;
        metric_par.mask_xn = -50; % No cross-channel masking
        metric_par.mask_q = 0.290256;

        metric_par.do_sprob_sum=false;
        metric_par.do_robust_pdet = false;
        metric_par.do_spatial_total_pooling = true;
        metric_par.steerpyr_filter='sp3Filters';
        
    case 'civdm'
        % Fitted to LocVisVC (http://dx.doi.org/10.1109/CVPR.2019.00558) using 'lpyr' (08/06/2020)        
        % Manually adjusted
        metric_par.base_sensitivity_correction = 0.2501039518897; 
        metric_par.mask_self=1.05317173444;
        metric_par.mask_xn=-0.43899877503;
        metric_par.mask_xo=-50;
        metric_par.mask_p=0.3424;
        metric_par.do_sprob_sum=true;
        metric_par.mask_q=0.108934275615;
        metric_par.psych_func_slope=0.34;
        metric_par.si_sigma=-0.502280453708;
        metric_par.do_robust_pdet = true;
        metric_par.do_spatial_total_pooling = false;
        metric_par.do_civdm = true;
        metric_par.ms_decomp='lpyr';
    otherwise
        error( 'Unknown task "%s"', task );
        
end

metric_par.sensitivity_correction = metric_par.base_sensitivity_correction;

%metric_par.beta = metric_par.psych_func_slope-metric_par.mask_p;

% process options
for i=1:2:length(options)
    
    switch options{i}
        case 'pixels_per_degree'
            metric_par.pix_per_deg = options{i+1};
        case 'viewing_distance'
            metric_par.view_dist = options{i+1};
        case 'sensitivity_correction'
            metric_par.sensitivity_correction = metric_par.base_sensitivity_correction + options{i+1};
        case 'age'
            metric_par.age = options{i+1};
            if ~strcmp( metric_par.mtf, 'cie' )
                warning( 'When ''age'' option is provided, consider switching to ''cie'' MTF: {''mtf'', ''cie''}' );
            end            
        otherwise
            % all other options
            %            if( ~isfield( metric_par, options{i} ) )
            %                error( 'hdrvdp: Unrecognized option "%s"', options{i} );
            %            end
            metric_par.(options{i}) = options{i+1};
    end
    
end


if isfield( metric_par, 'surround_l' )
    error( '"surround_l" option is no longer supported. Use "surround" instead.' );
end

if isfield( metric_par, 'peak_sensitivity' )
    error( '"peak_sensitivity" option is no longer supported. Use "sensitivity_correctin" instead.' );
end

end
