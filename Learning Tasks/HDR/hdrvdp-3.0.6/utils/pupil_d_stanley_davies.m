function d = pupil_d_stanley_davies( L, area )

La = L * area;

d = 7.75 - 5.75 * ( (La/846).^0.41 ./ ((La/846)^0.41 + 2) );

end