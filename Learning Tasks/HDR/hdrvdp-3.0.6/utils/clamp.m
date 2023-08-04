function Y = clamp( X, min, max )
% CLAMP restricts values of 'X' to be within the range from 'min' to 'max'.
%
% Y = clamp( X, min, max )
%  
% (C) Rafal Mantiuk <mantiuk@gmail.com>
% This is an experimental code for internal use. Do not redistribute.

  Y = X;

if( isa( X, 'single' ) )
  Y(X<min) = single(min);
  Y(X>max) = single(max);
else
  Y(X<min) = double(min);
  Y(X>max) = double(max);    
end

end
