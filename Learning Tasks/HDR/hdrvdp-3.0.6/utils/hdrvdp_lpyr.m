classdef hdrvdp_lpyr < hdrvdp_multscale
% Laplacian pyramid
    
    properties
        P;
        
        ppd;
        base_ppd;
        img_sz;
        band_freqs;        
    end
    
    methods
        
%        function ms = hdrvdp_lpyr( )
%        end
        
        function ms = decompose( ms, I, ppd )
            
            ms.ppd = ppd;
            ms.base_ppd = 2^ceil(log2(ppd));
            ms.img_sz = size(I);
            
            % We want the minimum frequency the band of 2cpd or higher
            height = max( ceil(log2(ppd))-2, 1 );
            ms.band_freqs = 2.^-(0:(height)) * ms.base_ppd / 2;
            
            % Resample to fix the frequency of the bands
            I_res = imresize( I, ms.base_ppd/ms.ppd, 'bilinear' );
            
            ms.P = laplacian_pyramid( I_res, height+1 );
        end
        
        function I = reconstruct( ms )
            I = zeros( size(ms.P{1}) );
            for kk=1:length(ms.P)
                I = I + ms.P{kk};
            end
            I = imresize( I, ms.img_sz );            
        end
        
        function B = get_band( ms, band, o )
            B = ms.P{band};
        end
            
        function ms = set_band( ms, band, o, B )
            ms.P{band} = B;
        end
                    
        function bc = band_count( ms )
            bc = length(ms.P);
        end
        
        function oc = orient_count( ms, band )
            oc = 1;
        end
        
        function sz = band_size( ms, band, o )
            sz = size( ms.P{band} );
        end

        function bf = get_freqs( ms )
            bf = ms.band_freqs;
        end
        
    end
    
end