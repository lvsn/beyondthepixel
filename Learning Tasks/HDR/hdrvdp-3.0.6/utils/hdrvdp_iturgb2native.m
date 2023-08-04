function M = hdrvdp_iturgb2native( itu_standard, IMG_E )
% Find a transformation matrix from one of the standard ITU color spaces
% into the native color space of the display, given its emission spectra.
% The matrix operates on relative values: RGB=[1 1 1] result in Y=1

[lambda, XYZ_cmf] = load_spectral_resp( 'ciexyz31.csv' );

M_native2XYZ = zeros(3,3);
for rr=1:3
    for cc=1:3
        M_native2XYZ(rr,cc) = trapz( lambda, XYZ_cmf(:,rr).*IMG_E(:,cc) ) * 683.002;
    end
end

% The absolute luminance must not change - we make sure that the row
% responsible for Y sums up to 1
M_native2XYZ = M_native2XYZ / sum(M_native2XYZ(2,:));

switch itu_standard

    case { 'rgb-bt.709', '709', 'rec709' }
        M_iturgb2xyz = [
            0.4124 0.3576 0.1805;
            0.2126 0.7152 0.0722;
            0.0193 0.1192 0.9505];
    case { 'rgb-bt.2020', '2020', 'rec2020' }
        M_iturgb2xyz = [0.6370, 0.1446, 0.1689;
               0.2627, 0.6780, 0.0593;
               0.0000, 0.0281, 1.0610];           
    case 'xyz'
        M_iturgb2xyz = eye(3);
        
    otherwise
        error( 'Unknown RGB color space' );        
end
    
M = M_native2XYZ \ M_iturgb2xyz;

end