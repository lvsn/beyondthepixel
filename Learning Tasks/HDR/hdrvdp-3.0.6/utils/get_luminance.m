function Y = get_luminance( img )
% Return 2D matrix of luminance values for 3D matrix with an RGB image

%dims = sum(nnz( size(img)>1 ));
dims = find(size(img)>1,1,'last');

if( dims == 3 )
    Y = img(:,:,1) * 0.212656 + img(:,:,2) * 0.715158 + img(:,:,3) * 0.072186;
elseif( dims == 1 || dims == 2 )
    Y = img;
else
    error( 'get_luminance: wrong matrix dimension' );
end

end