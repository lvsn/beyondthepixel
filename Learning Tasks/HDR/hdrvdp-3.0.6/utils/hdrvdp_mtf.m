function MTF = hdrvdp_mtf( rho, metric_par )
% Custom-fit MTF of the eye
%
% Copyright (c) 2011, Rafal Mantiuk <mantiuk@gmail.com>

% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
%
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


switch metric_par.mtf 
    case 'none'
        % When MTF disabled, the modulation is 1
        MTF=ones(size(rho));
    case 'cie'
        % Age-adative CIE99 Glare Spread Function
        MTF = hdrvdp_otf_cie99( rho, metric_par.age );        
    case 'hdrvdp'
        % Standard HDR-VDP MTF
        MTF = zeros(size(rho));
        for kk=1:4
            MTF = MTF + metric_par.mtf_params_a(kk) * exp( -metric_par.mtf_params_b(kk) * rho );
        end
    otherwise 
        error( 'Unrecognized MTF function "%s"', metric_par.mtf  );        
end
    
end

        % Lost track where these equations are comming from - not used
        % anymore
%         if( metric_par.do_mtf_aging )            
%             f = linspace( 0, max(rho2(:)), 1024 );            
%             
%             s = .02;
%             a = 1.3;
%             % MTF for 64 years mean observer
%             mtf_64 = (1 - s + a*f.^.5) ./ (1 + a*f.^.5 + .6*f.^2.) + s;
% 
%             s = .0;
%             a = 1.;
%             % MTF for 29 years mean observer
%             mtf_29 = (1 - s + a*f.^.5) ./ (1 + a*f.^.5 + .1*f.^2.) + s;
%             
%             alpha = clamp( (metric_par.age-29)/(64-29), 0, 1 );
%             mtf_a = (1-alpha)*mtf_29 + alpha*mtf_64;
%             
%             mtf_filter = mtf_filter + interp1( f, mtf_a-mtf_29, rho2 );            
%             
%         end            

