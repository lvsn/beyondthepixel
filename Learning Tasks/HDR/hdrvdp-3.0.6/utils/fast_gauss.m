function Y = fast_gauss( X, sigma, do_norm, pad_value )
% Low-pass filter image using the Gaussian filter
% 
% Y = blur_gaussian( X, sigma, do_norm, pad_value  )
%  
% do_norm - normalize the results or not, 'true' by default. If do_norm==true,
% filtering computes weighted average (defult), if do_norm==false, filtering computes a
% weighted sum. 
% pad_value - value passed to padarray (0, 1, 'replicate', etc.)
%
% Use FFT or spatial domain, whichever is faster
%

% Copyright (c) 2016, Rafal Mantiuk <mantiuk@gmail.com>

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

if( ~exist( 'do_norm', 'var' ) )
    do_norm = true;
end

if( ~exist( 'pad_value', 'var' ) )
    pad_value = 'replicate';
end

if( sigma >= 4.3 ) % Experimentally found threshold when FFT is faster

    ks = [size(X,1) size(X,2)]*2;
       
    N = ks(1);
    M = ks(2);
    KX0 = (mod(1/2 + (0:(M-1))/M, 1) - 1/2); 
    KX1 = KX0 * (2*pi);
    KY0 = (mod(1/2 + (0:(N-1))/N, 1) - 1/2);
    KY1 = KY0 * (2*pi);
    
    [XX, YY] = meshgrid( KX1, KY1 );
    
    K = exp( -0.5*(XX.^2 + YY.^2)*sigma^2 );
    
    if( ~do_norm )
        K = K/(sum(K(:))/numel(K));
    end
    
    Y = zeros( size(X) );
    for cc=1:size(X,3)
        Y(:,:,cc) = fast_conv_fft( X(:,:,cc), K, pad_value );
    end        
    
else
    ksize = round(sigma*6);   
    ksize = ksize + 1-mod(ksize,2); % Make sure the kernel size is always an odd number
    h = fspecial( 'gaussian', ksize, sigma );
    if( ~do_norm )
        h = h/h(ceil(size(h,1)/2),ceil(size(h,2)/2));
    end
        
    Y = imfilter( X, h, pad_value );    
    
end

end


function Y = fast_conv_fft( X, fH, pad_value )
% Convolve with a large support kernel in the Fourier domain.
%
% Y = fast_conv_fft( X, fH, pad_value )
%
% X - image to be convolved (in spatial domain)
% fH - filter to convolve with in the Fourier domain, idealy 2x size of X
% pad_value - value to use for padding when expanding X to the size of fH
%
% (C) Rafal Mantiuk <mantiuk@gmail.com>
% This is an experimental code for internal use. Do not redistribute.

pad_size = (size(fH)-size(X));

%mX = mean( X(:) );

fX = fft2( padarray( X, pad_size, pad_value, 'post' ) );

Yl = real(ifft2( fX.*fH, size(fX,1), size(fX,2), 'symmetric' ));

Y = Yl(1:size(X,1),1:size(X,2));

end
