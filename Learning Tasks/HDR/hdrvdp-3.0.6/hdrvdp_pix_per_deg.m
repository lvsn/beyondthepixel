function ppd = hdrvdp_pix_per_deg( display_diagonal_in, resolution, viewing_distance )
% HDRVDP_PIX_PER_DEG - computer pixels per degree given display parameters
% and viewing distance
%
% ppd = hdrvdp_pix_per_deg( display_diagonal_in, resolution,
% viewing_distance )
%
% This is a convenience function that can be used to provide angular
% resolution of input images for the HDR-VDP-2. 
%
% display_diagonal_in - diagonal display size in inches, e.g. 19, 14
% resolution - display resolution in pixels as a vector, e.g. [1024 768]
% viewing_distance - viewing distance in meters, e.g. 0.5
%
% Note that the function assumes 'square' pixels, so that the aspect ratio
% is resolution(1):resolution(2).
%
% EXAMPLE:
% ppd = hdrvdp_pix_per_deg( 24, [1920 1200], 0.5 );
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

ar = resolution(1)/resolution(2); % width/height

height_mm = sqrt( (display_diagonal_in*25.4)^2 / (1+ar^2) );

height_deg = 2 * atand( 0.5*height_mm/(viewing_distance*1000) );

ppd = resolution(2)/height_deg;

end