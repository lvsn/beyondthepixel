function M = hdrvdp_otf_cie99( rho, age, p )
% Modulation transfer function of the eye, computed as a Fourier transform of 
% the CIE99 GSF. 
%
% M = mtf_cie_ft( rho )
% M = mtf_cie_ft( rho, age )
% M = mtf_cie_ft( rho, age, p )
%
% angledeg - angle in visual degrees
% age - age in years (default is 24)
% p - iris pigmentation (default is 0.5)
%  p=0   - Non caucasian
%  p=0.5 - Brown eye
%  p=1   - Green eye
%  p=1.2 - Light blue
%
% This model does not account for a pupil size (assumed that it is readly
% known).
%
% This is a glare spread function from:
% Vos, J.J. and van den Berg, T.J.T.P [CIE Research note 135/1, Disability
% Glare] ISBN 3 900 734 97 6 (1999).

if ~exist( 'age', 'var' )
    age = 24;
end

if ~exist( 'p', 'var' )
    p = 0.5;
end

omega = 2*pi*rho;

M = cie_mtf( omega, age, p )/cie_mtf( 0.0001, age, p );

M(omega<1e-4) = 1; % To avoid NaN for rho=0

end

function M = cie_mtf( omega, age, p )

% The equation was found by applying a Fourier transform using Matlab's
% symbolic toolbox

c1=9.2e6;
c2=0.08;
c3=0.0046;
c4=1.5e5;
c5=0.045;
c6=1.6;
c7=400;
c8=0.1;
c9=3e-8;
c10=1300;
c11=0.8;
c12=2.5e-3;
c13=0.0417;
c14=0.055;

age_m=70;

M = (p*((age^4*c6)/age_m^4 + 1)*(2*c8*c11*besselk(0, c8*abs(omega)) + 2*c8^2*c10*abs(omega).*besselk(1, c8*abs(omega))) ...
    - ((age^4*c2)/age_m^4 - 1)*(2*c1*c3^2*abs(omega).*besselk(1, c3*abs(omega)) + 2*c4*c5^2*abs(omega).*besselk(1, c5*abs(omega))) ...
    + 2*pi*c12*p*dirac(omega) - (2*c9*pi*(c6*age^4 + age_m^4).*dirac(2, omega))/age_m^4 + (c7*c8*pi*exp(-c8*abs(omega))*(c6*age^4 + age_m^4))/age_m^4)/(p*(c13 + (age^4*c14)/age_m^4) + 1);

end

 