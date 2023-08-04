function d = pupil_d_unified( L, area, age )

y0 = 28.58; % Reference age from the paper
y = clamp( age, 20, 83 ); % The formula applies to this age-range

d_sd = pupil_d_stanley_davies( L, area );
d = d_sd + (y-y0)*(0.02132-0.009562*d_sd);

end