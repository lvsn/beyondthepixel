classdef hdrvdp_spyr < hdrvdp_multscale
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
        
        function ms = hdrvdp_spyr( steerpyr_filter )
            ms.steerpyr_filter = steerpyr_filter;
        end
        
        function ms = decompose( ms, I, ppd )
            ms.ppd = ppd;
            ms.img_sz = size(I);
            
            % We want the minimum frequency the band of 2cpd or higher
            % that is base-band can be 1cpd
            [~,~,lofilt] = eval(ms.steerpyr_filter);
            pyr_height = min( ceil(log2(ppd))-2, maxPyrHt(ms.img_sz,size(lofilt)) );
            ms.band_freqs = 2.^-(0:(pyr_height+1)) * ms.ppd / 2;
                        
            [ms.pyr,ms.pind] = buildSpyr( I, pyr_height, ms.steerpyr_filter );
            ms.sz = ones( spyrHt( ms.pind ) + 2, 1 );
            ms.sz(2:end-1) = spyrNumBands( ms.pind );
        end
        
        function I = reconstruct( ms )
            I = reconSpyr( ms.pyr, ms.pind, ms.steerpyr_filter );
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