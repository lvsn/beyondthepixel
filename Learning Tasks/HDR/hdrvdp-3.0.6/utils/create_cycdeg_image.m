function D = create_cycdeg_image( im_size, pix_per_deg )
% CREATE_CYCDEG_IMAGE (internal) create matrix that contains frequencies,
% given in cycles per degree.
%
% D = create_cycdeg_image( im_size, pix_per_deg )
% im_size     - [height width] vector with image size
% pix_per_deg - pixels per degree for both horizontal and vertical axis
%               (assumes square pixels)
%
% Useful for constructing Fourier-domain filters based on OTF or CSF data.
%
% (C) Rafal Mantiuk <mantiuk@gmail.com>
% This is an experimental code for internal use. Do not redistribute.

nyquist_freq = 0.5 * pix_per_deg;

KX0 = (mod(1/2 + (0:(im_size(2)-1))/im_size(2), 1) - 1/2);
KX1 = KX0 * nyquist_freq * 2;
KY0 = (mod(1/2 + (0:(im_size(1)-1))/im_size(1), 1) - 1/2);
KY1 = KY0 * nyquist_freq * 2;

[XX, YY] = meshgrid( KX1, KY1 );

D = sqrt( XX.^2 + YY.^2 );

end