function [bands, L_adapt, bb_padvalue, P] = hdrvdp_visual_pathway( img, name, metric_par, bb_padvalue )
% HDRVDP_VISUAL_PATHWAY (internal) Process image along the visual pathway
% to compute normalized perceptual response
%
% img - image data (can be multi-spectral)
% name - string with the name of this map (shown in warnings and error
%        messages)
% options - cell array with the 'option', value pairs
% bands - CSF normalized freq. bands
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

global hdrvdp_cache;

if( any( isnan( img(:) ) ) )
    warning( 'hdrvdp:BadImageData', '%s image contains NaN pixel values', name );
    img(isnan(img)) = 1e-5;
end

% =================================
% Precompute
%
% precompute common variables and structures or take them from the cache
% =================================

width = size(img,2);
height = size(img,1);
img_sz = [height width]; % image size
img_ch = size(img,3); % number of color channels

rho2 = create_cycdeg_image( img_sz*2, metric_par.pix_per_deg ); % spatial frequency for each FFT coefficient, for 2x image size

% Load spectral sensitivity curves
[lambda, LMSR_S] = load_spectral_resp( 'log_cone_smith_pokorny_1975.csv' );
LMSR_S(LMSR_S==0) = min(LMSR_S(:));
LMSR_S = 10.^LMSR_S;

[~, ROD_S] = load_spectral_resp( 'cie_scotopic_lum.txt' );
LMSR_S(:,4) = ROD_S;

IMG_E = metric_par.spectral_emission;

% =================================
% Precompute photoreceptor non-linearity
% =================================

pn = hdrvdp_get_from_cache( 'pn', [metric_par.rod_sensitivity metric_par.csf_sa], @() create_pn_jnd( metric_par ) );

pn.jnd{1} = pn.jnd{1} * 10.^metric_par.sensitivity_correction;
pn.jnd{2} = pn.jnd{2} * 10.^metric_par.sensitivity_correction;

% =================================
% Optical Transfer Function
% =================================

L_O = zeros( size(img) );
for k=1:img_ch  % for each color channel
    if ~strcmp( metric_par.mtf, 'none' ) 
        % Use per-channel or one per all channels surround value
        if ischar( metric_par.pad_value )
            pad_value = metric_par.pad_value;
        else
            pad_value = metric_par.pad_value( k );
        end
        mtf_filter = hdrvdp_mtf( rho2, metric_par );
        L_O(:,:,k) =  clamp( fast_conv_fft( double(img(:,:,k)), mtf_filter, pad_value ), 1e-5, 1e10 );
    else
        % NO mtf
        L_O(:,:,k) =  img(:,:,k);
    end    
end

if( ~metric_par.debug )
    % memory savings for huge images
    clear mtf_filter;
    clear rho2;
end

%TODO - MTF reduces luminance values

% =================================
% Age-related spectral luminous efficiency loss - wavelength-dependent
% effect
% =================================

if( metric_par.do_aslw )

    lam = 420:50:670;  % a few lambda samples
    beta = - [.0163  .0122  .0055  0  .0060  .0068];  % est. from Fig.4

    % age-related transmission filter. The effect is simulated as an optical
    % filter which reduces light at certain wavelengths.
    
    age_r = max( 15, metric_par.age );    
    trans_filter = 10.^((age_r-25) * interp1( lam, beta, lambda, 'cubic', 'extrap' ));
    %trans_filter = 10.^((metric_par.age-35) * interp1( lam, beta, lambda, 'cubic', 'extrap' ));

    IMG_E = IMG_E .* repmat( trans_filter', [1 size(IMG_E,2)] );

end

if( metric_par.do_aod )

    lam = [400 410 420 430 440 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650];
    TL1 = [0.600 0.510 0.433 0.377 0.327 0.295 0.267 0.233 0.207 0.187 0.167 0.147 0.133 0.120 0.107 0.093 0.080 0.067 0.053 0.040 0.033 0.027 0.020 0.013 0.007 0.000];
    TL2 = [1.000 0.583 0.300 0.116 0.033 0.005 zeros(1,20)];
    
    OD_y = TL1 + TL2;
%    trans_y = 10.^-interp1( lam, OD_y, lambda, 'cubic', 'extrap' );
    
    if metric_par.age <= 60
        OD = TL1 .* (1 + .02*(metric_par.age-32)) + TL2;
    else
        OD = TL1 .* (1.56 + .0667*(metric_par.age-60)) + TL2; 
    end
    
%    trans_filter = min( (10.^-interp1( lam, OD, lambda, 'cubic', 'extrap' ))./trans_y, 1 );

    trans_filter = 10.^-interp1( lam, OD-OD_y, clamp( lambda, lam(1), lam(end)), 'pchip' );
    
    IMG_E = IMG_E .* repmat( trans_filter', [1 size(IMG_E,2)] );

end

% =================================
% Photoreceptor spectral sensitivity
% =================================

% Matrix M_img_lmsr converts from the native color space (of a display)
% into absolute reponse of LMS cones and rods
M_img_lmsr = zeros( img_ch, 4 ); % Color space transformation matrix
for k=1:4
    for l=1:img_ch
        M_img_lmsr(l,k) = trapz( lambda, LMSR_S(:,k).*IMG_E(:,l) * 683.002 );                
    end
end

% Normalization step
% We want RGB=[1 1 1] to result in L+M=1 (L+M is equivalent to luminance)
M_img_lmsr = M_img_lmsr / sum(sum(M_img_lmsr(:,1:2)));

% Convert from the native color space of the display into LMSR responses
R_LMSR = reshape( reshape( L_O, width*height, img_ch )*M_img_lmsr, height, width, 4 );

R_LMSR = clamp( R_LMSR, 1e-8, 1e10 ); % To avoid negative numbers

if( ~metric_par.debug )
    % memory savings for huge images
    clear L_O;
end


%surround_LMSR = metric_par.surround_l * M_img_lmsr;

% =================================
% Adapting luminance
% =================================

L_adapt = hdrvdp_local_adapt( R_LMSR(:,:,1) + R_LMSR(:,:,2), metric_par.pix_per_deg );

% =================================
% Age related pupil contraction at low luminance
% =================================

if( metric_par.do_sl )
    % light reduction due to senile miosis
    L_a = geomean( L_adapt(:) );
    d = 4.6 - 2.8*tanh(0.4 * log10(0.625*L_a));   % pupil diam. (as in barten_csf)
    dslope = -.05 ./ (1 + ((.0025 .* L_a).^.5)); % Watson Fig.15b
    lum_reduction = ( d + dslope.*(clamp( metric_par.age, 20, 80 )-20) ).^2 ./ d.^2;
    
    R_LMSR = R_LMSR * lum_reduction;    
end

if( metric_par.do_slum )
    % light reduction due to senile miosis (Unified Formula based on
    % the Stanley and Davies function [Watson & Yellott 2012]
    
    L_a = geomean( L_adapt(:) );
    area = size(img,1)*size(img,2)/metric_par.pix_per_deg^2;
            
    d_ref = pupil_d_unified( L_a, area, 28 );
    d_age = pupil_d_unified( L_a, area, metric_par.age );
    
    lum_reduction = d_age.^2 ./ d_ref.^2;
    
    R_LMSR = R_LMSR * lum_reduction;    
end



% =================================
% Photoreceptor non-linearity
% =================================

%La = mean( L_adapt(:) );

P_LMR = zeros(height, width, 4);
%surround_P_LMR = zeros(1,4);
for k=[1:2 4] % ignore S - does not influence luminance   
    if( k==4 )
        ph_type = 2; % rod
        ii = 3;
    else
        ph_type = 1; % cone
        ii = k;
    end
    
    P_LMR(:,:,ii) = pointOp( log10( clamp(R_LMSR(:,:,k), 10^pn.Y{ph_type}(1), 10^pn.Y{ph_type}(end)) ), ...
        pn.jnd{ph_type}, pn.Y{ph_type}(1), pn.Y{ph_type}(2)-pn.Y{ph_type}(1), 0 );
    
%    surround_P_LMR(ii) = interp1( pn_Y{ph_type}, pn_jnd{ph_type}, ...
%        log10( clamp(surround_LMSR(k), 10^pn_Y{ph_type}(1), 10^pn_Y{ph_type}(end)) ) );
end

if( ~metric_par.debug )
    % memory savings for huge images
    clear R_LMSR;
end


% =================================
% Remove the DC component, from 
% cone and rod pathways separately
% =================================

% TODO - check if there is a better way to do it
% cones
P_C = P_LMR(:,:,1)+P_LMR(:,:,2);
%mm = mean(mean( P_C ));
%P_C = P_C - mm;
% rods
P_R = P_LMR(:,:,3);
%mm = mean(mean( P_LMR(:,:,3) ));
%P_R = P_LMR(:,:,3) - mm;


% =================================
% Achromatic response
% =================================

P = P_C + P_R;

if( ~metric_par.debug )
    % memory savings for huge images
    clear P_LMR P_C P_R;
end


%surround_P = surround_P_LMR(1)+surround_P_LMR(2)+surround_P_LMR(3);

% =================================
% Multi-channel decomposition
% =================================

bands = metric_par.mult_scale.decompose( P, metric_par.pix_per_deg );

% Remove DC from the base-band - this is important for contrast masking
BB = bands.get_band( bands.band_count(), 1 );
bands = bands.set_band( bands.band_count(), 1, BB - mean(BB(:)) );

    function item = cache_get( item_name, item_signature, compute_func )
        sign_name = [ item_name '_sign' ];
        if( isfield( hdrvdp_cache, sign_name ) && all( hdrvdp_cache.( sign_name ) == item_signature) )  % caching
            item = hdrvdp_cache.( item_name );
        else
            item = compute_func();
            hdrvdp_cache.( sign_name ) = zeros(size(item_signature)); % in case of breaking at this point
            hdrvdp_cache.( item_name ) = item;
            hdrvdp_cache.( sign_name ) = item_signature;
        end
    end

end


function [Y, jnd] = build_jndspace_from_S(l,S)

L = 10.^l;
dL = zeros(size(L));

for k=1:length(L)

    % Different than in the paper because integration is done in the log
    % domain - requires substitution with a Jacobian determinant
    dL(k) = S(k) * log(10);
%
%  This came from:
%    thr = L(k)/S(k);
%    dL(k) = 1/thr * L(k) * log(10);
end

Y = l;
jnd = cumtrapz( l, dL );

end

function pn = create_pn_jnd( metric_par )
% Create lookup tables for intensity -> JND mapping

c_l = logspace( -5, 5, 2048 );

s_A = hdrvdp_joint_rod_cone_sens( c_l, metric_par );
s_R = hdrvdp_rod_sens( c_l, metric_par ) * 10.^metric_par.rod_sensitivity;

% s_C = s_L = S_M
s_C = 0.5 * interp1( c_l, max(s_A-s_R, 1e-3), min( c_l*2, c_l(end) ) );

pn = struct();

[pn.Y{1}, pn.jnd{1}] = build_jndspace_from_S( log10(c_l), s_C );
[pn.Y{2}, pn.jnd{2}] = build_jndspace_from_S( log10(c_l), s_R );

end
