classdef hdrvdp_spyr_fixedbands < hdrvdp_multscale
% Steerable pyramid

    properties
        steerpyr_filter;
        pyr;        
        pind;        
        sz;
        
        ppd;
        base_ppd;
        img_sz;
        band_freqs;
    end
    
    methods
        
        function ms = hdrvdp_spyr_fixedbands( steerpyr_filter )
            ms.steerpyr_filter = steerpyr_filter;
        end
        
        function ms = decompose( ms, I, ppd )
            ms.ppd = ppd;
            ms.base_ppd = 2^ceil(log2(ppd));
            ms.img_sz = size(I);
            
            % We want the minimum frequency the band of 2cpd or higher
            height = min( ceil(log2(ppd))-2, maxPyrHt(ms.img_sz,size(eval(ms.steerpyr_filter))) );
            ms.band_freqs = 2.^-(0:(height+1)) * ms.base_ppd / 2;
            
            % Resample to fix the frequency of the bands
            I_res = imresize( I, ms.base_ppd/ms.ppd, 'bilinear' );
            
            [ms.pyr,ms.pind] = buildSpyr( I_res, height, ms.steerpyr_filter );
            ms.sz = ones( spyrHt( ms.pind ) + 2, 1 );
            ms.sz(2:end-1) = spyrNumBands( ms.pind );
        end
        
        function I = reconstruct( ms )
            I_res = reconSpyr( ms.pyr, ms.pind, ms.steerpyr_filter );
            I = imresize( I_res, ms.img_sz, 'box' );
        end
        
        function B = get_band( ms, band, o )
            band_norm = 2^(band-1);  % This accounts for the fact that amplitudes increase in the steerable pyramid
            oc = min( o, ms.sz(band) );
            B = pyrBand( ms.pyr, ms.pind, sum(ms.sz(1:(band-1)))+oc )/band_norm;            
        end
            
        function ms = set_band( ms, band, o, B )
            band_norm = 2^(band-1);  % This accounts for the fact that amplitudes increase in the steerable pyramid
            ms.pyr(pyrBandIndices(ms.pind,sum(ms.sz(1:(band-1)))+o)) = B*band_norm;
        end
                    
        function bc = band_count( ms )
            bc = length(ms.sz);
        end
        
        function oc = orient_count( ms, band )
            oc = ms.sz(band);
        end
        
        function sz = band_size( ms, band, o )
            sz = ms.pind(sum(ms.sz(1:(band-1)))+o,:);
        end

        function bf = get_freqs( ms )
            bf = ms.band_freqs;
        end

        
    end
    
end